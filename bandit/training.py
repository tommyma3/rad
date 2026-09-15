"""Shared Accelerate trainer with resumable state and deterministic target sampling."""
import argparse
import json
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import BanditDataset, assert_disjoint
from .model import make_model
from .optimizer_utils import configure_trainability, optimizer_groups
from .utils import SCHEMA, file_digest, load_config, project_path, write_json

ARCHITECTURE_KEYS = ("model", "num_arms", "context_steps", "tf_n_embd", "tf_n_head",
                     "tf_n_layer", "tf_dim_feedforward", "n_compress_tokens",
                     "compress_n_layers", "compress_n_heads", "short_memory_keep",
                     "latent_update_mode", "always_use_latent_prefix")


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Incompatible checkpoint schema")
    model = make_model(payload["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def save_checkpoint(accelerator, model, run_dir, step, config, data_digests, pretrain):
    target = run_dir / f"checkpoint-{step:07d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    staging = run_dir / f".checkpoint-{step:07d}.partial"
    accelerator.save_state(str(staging))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {"schema": SCHEMA, "config": config, "step": step,
                 "phase": "pretrain" if pretrain else "distill", "data_digests": data_digests,
                 "world_size": accelerator.num_processes}
        torch.save({**state, "model": accelerator.unwrap_model(model).state_dict()}, staging / "model.pt")
        write_json(staging / "training.json", state)
        staging.rename(target)
        write_json(run_dir / "latest.json", {"checkpoint": target.name, "step": step})
    accelerator.wait_for_everyone()


def train(config, dataset_dir, run_dir, *, pretrain=False, resume=None, pretrained=None,
          stop_after=None, cpu=False):
    config = dict(config)
    dataset_dir, run_dir = Path(dataset_dir), Path(run_dir)
    if resume and pretrained:
        raise ValueError("Use either resume or pretrained, not both")
    accelerator = Accelerator(
        cpu=cpu, mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        # Short prefixes intentionally skip all compression/gate parameters.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(config["seed"])
    steps = int(config["train_steps"])
    if min(steps, config["batch_size"], config["gradient_accumulation_steps"],
           config["checkpoint_interval"], config["eval_interval"], config["log_interval"],
           config["validation_batches"]) < 1:
        raise ValueError("Training step, batch, and interval settings must be positive")
    context = config["context_steps"] if pretrain or config["model"] == "AD" else None
    train_data = BanditDataset(dataset_dir / "train.hdf5", context, "train")
    validation_data = BanditDataset(dataset_dir / "validation.hdf5", context, "validation")
    assert_disjoint(train_data, validation_data)
    if train_data.collection_config != validation_data.collection_config:
        raise ValueError("Training and validation must use the same collection settings")
    if config["num_arms"] != train_data.num_arms:
        raise ValueError("Model action space does not match the collection")
    # Preserve actual sampled delays/insertion points, including when the model
    # YAML's default environment differs from the collection supplied on the CLI.
    config["collection_config"] = train_data.collection_config
    data_digests = {split: file_digest(dataset_dir / f"{split}.hdf5")
                    for split in ("train", "validation")}
    model = make_model(config)
    if pretrained:
        _, payload = load_checkpoint(pretrained)
        if payload["phase"] != "pretrain":
            raise ValueError("--pretrained requires a compression-pretraining checkpoint")
        if any(config.get(key) != payload["config"].get(key) for key in ARCHITECTURE_KEYS):
            raise ValueError("Pretrained architecture differs from training configuration")
        model.load_state_dict(payload["model"], strict=True)
    configure_trainability(model, pretrain)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config),
                                   betas=(config["beta1"], config["beta2"]),
                                   weight_decay=config["weight_decay"])

    def learning_rate(step):
        warmup = min(int(config["warmup_steps"]), max(0, steps - 1))
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    accelerator.register_for_checkpointing(scheduler)
    model, optimizer = accelerator.prepare(model, optimizer)
    start_step = 0
    if resume:
        metadata = json.loads((Path(resume) / "training.json").read_text(encoding="utf-8"))
        if metadata["schema"] != SCHEMA or metadata["config"] != config or metadata["data_digests"] != data_digests:
            raise ValueError("Resume requires identical configuration and collection files")
        if metadata["phase"] != ("pretrain" if pretrain else "distill"):
            raise ValueError("Cannot resume across training phases")
        if metadata["world_size"] != accelerator.num_processes:
            raise ValueError("Exact resume requires the same process count")
        accelerator.load_state(str(resume))
        start_step = metadata["step"]
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty for fresh training: {run_dir}")
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "data.json", {"directory": str(dataset_dir.resolve()), "digests": data_digests})
    accelerator.wait_for_everyone()
    writer = SummaryWriter(str(run_dir / "tensorboard")) if accelerator.is_main_process else None
    optimizer.zero_grad(set_to_none=True)
    end_step = min(steps, stop_after) if stop_after is not None else steps
    if end_step <= start_step:
        raise ValueError("Requested end step must exceed the resumed step")
    for update in range(start_step, end_step):
        model.train()
        total_loss = torch.zeros((), device=accelerator.device)
        for micro in range(config["gradient_accumulation_steps"]):
            rng = np.random.default_rng(np.random.SeedSequence(
                [config["seed"], update, micro, accelerator.process_index]))
            batch = train_data.sample_batch(config["batch_size"], rng, pretrain)
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            with accelerator.accumulate(model):
                output = model(batch, pretrain=pretrain)
                if not bool(torch.isfinite(output["loss"])):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(output["loss"])
                total_loss += output["loss"].detach() / config["gradient_accumulation_steps"]
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if not accelerator.optimizer_step_was_skipped:
            scheduler.step()
        step = update + 1
        log = {"step": step}
        if step % config["log_interval"] == 0 or step == end_step:
            log["train_loss"] = float(accelerator.reduce(total_loss, reduction="mean"))
            log["lr"] = scheduler.get_last_lr()[0]
        if step % config["eval_interval"] == 0 or step == end_step:
            model.eval()
            metrics = torch.zeros(2, device=accelerator.device)
            with torch.no_grad():
                for batch_index in range(config["validation_batches"]):
                    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 991, batch_index]))
                    batch = validation_data.sample_batch(config["batch_size"], rng, pretrain)
                    batch = {key: value.to(accelerator.device) for key, value in batch.items()}
                    output = model(batch, pretrain=pretrain)
                    metrics += torch.stack((output["loss"], output["accuracy"]))
            metrics /= config["validation_batches"]
            log.update(validation_loss=float(metrics[0]), validation_accuracy=float(metrics[1]))
        if writer and len(log) > 1:
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log) + "\n")
            for key, value in log.items():
                if key != "step":
                    writer.add_scalar(key, value, step)
            print(json.dumps(log), flush=True)
        if step % config["checkpoint_interval"] == 0 or step == end_step:
            save_checkpoint(accelerator, model, run_dir, step, config, data_digests, pretrain)
    if writer:
        writer.close()
    accelerator.end_training()
    return run_dir / f"checkpoint-{end_step:07d}"


def main(default_model="ad_short", pretrain=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_model)
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--dataset", default="datasets/delayed")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--pretrained")
    parser.add_argument("--stop_after", type=int, help="Save early while retaining the planned LR schedule")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.resume:
        config = json.loads((project_path(args.resume) / "training.json").read_text(encoding="utf-8"))["config"]
    else:
        config = load_config(args.config, args.env)
        if pretrain:
            config.update(config.get("pretrain", {}))
    for argument, key in ((args.steps, "train_steps"), (args.batch_size, "batch_size"),
                          (args.seed, "seed"), (args.mixed_precision, "mixed_precision")):
        if argument is not None:
            config[key] = argument
    if pretrain and config["model"] != "RAD":
        raise ValueError("Pretraining requires a RAD config")
    target = train(config, project_path(args.dataset), project_path(args.run_dir), pretrain=pretrain,
                   resume=project_path(args.resume) if args.resume else None,
                   pretrained=project_path(args.pretrained) if args.pretrained else None,
                   stop_after=args.stop_after, cpu=args.cpu)
    print(f"Saved checkpoint: {target}")
