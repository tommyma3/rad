"""Train one isolated evidence-integration run on one device."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bandit.evidence_experiment import (DEFAULT_CONFIG, METHODS, PROTOCOL, EvidenceDataset,
    model_config, prepare_data, read_study, save_payload)
from bandit.model import make_model
from bandit.optimizer_utils import configure_trainability, optimizer_groups
from bandit.utils import project_path, write_json


def train_run(study, data_dir, run_dir, method, seed, *, device="cuda", resume=False, stop_after=None):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Use the sweep launcher: each GPU trains an independent run, without DDP")
    if seed < 0:
        raise ValueError("Training seed must be nonnegative")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; --device cpu is for smoke validation")
    config = model_config(study, method, seed)
    for key in ("train_steps", "batch_size", "eval_interval", "checkpoint_interval", "log_interval"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("This single-device diagnostic uses gradient_accumulation_steps=1")
    precision = config["mixed_precision"]
    if precision not in ("no", "bf16", "fp16") or (device.type != "cuda" and precision != "no"):
        raise ValueError("Mixed precision requires CUDA and must be no, bf16, or fp16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; choose no or fp16")
    manifest = prepare_data(data_dir, study)
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Fresh run requires an empty directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng([seed, 619])
    train_data = EvidenceDataset(data_dir / "train.npz", study)
    validation_data = EvidenceDataset(data_dir / "validation.npz", study)
    model = make_model(config).to(device)
    configure_trainability(model)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config),
        betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"])
    steps, batch_size = config["train_steps"], config["batch_size"]

    def learning_rate(step):
        warmup = min(config["warmup_steps"], max(0, steps - 1))
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    start, best_loss, best_payload = 0, float("inf"), None
    if resume and not (run_dir / "last.pt").exists():
        # A worker may fail before its first checkpoint. Restart that owned run
        # from the same seed, rather than requiring deletion of its artifacts.
        previous = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if previous["config"] != config or previous["study"] != study or previous["data_digests"] != manifest["digests"]:
            raise ValueError("Cannot restart an uncheckpointed run with changed settings/data")
    elif resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if (saved.get("protocol") != PROTOCOL or saved["config"] != config or
                saved["study"] != study or saved["data_digests"] != manifest["digests"] or
                saved["device_type"] != device.type):
            raise ValueError("Resume requires identical study, data, model, seed, and device type")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best_loss = saved["step"], saved["best_validation_loss"]
        best_payload = saved["best_payload"]
        if best_payload is not None:
            save_payload(run_dir / "best-model.pt", best_payload)
        if start == steps:
            if not (run_dir / "best-model.pt").exists():
                raise FileNotFoundError("Completed run is missing its best validation checkpoint")
            if not (run_dir / "model.pt").exists():
                save_payload(run_dir / "model.pt", {key: saved[key] for key in
                    ("protocol", "study", "config", "method", "seed", "step", "data_digests",
                     "model", "best_validation_loss")})
            return run_dir / "best-model.pt"
    end = min(steps, stop_after) if stop_after is not None else steps
    if end <= start:
        raise ValueError("Requested stop must be after the resumed step")
    if resume and (run_dir / "metrics.jsonl").exists():
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        retained = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue  # A hard interruption may leave a partial final line.
            if value["step"] <= start:
                retained.append(line)
        (run_dir / "metrics.jsonl").write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    write_json(run_dir / "config.json", {"protocol": PROTOCOL, "study": study, "config": config,
        "method": method, "seed": seed, "data_digests": manifest["digests"],
        "device": str(device), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    context = config["context_steps"] if method != "rad" else None

    def autocast():
        return (nullcontext() if precision == "no" else torch.autocast("cuda",
            dtype=torch.bfloat16 if precision == "bf16" else torch.float16))

    def payload(step):
        return {"protocol": PROTOCOL, "study": study, "config": config, "method": method,
                "seed": seed, "step": step, "data_digests": manifest["digests"],
                "model": model.state_dict(), "best_validation_loss": best_loss}

    for step in range(start + 1, end + 1):
        model.train()
        batch = train_data.batch(rng.integers(train_data.size, size=batch_size), context_steps=context, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            result = model(batch)
        if not bool(torch.isfinite(result["loss"])):
            raise FloatingPointError("Nonfinite evidence-training loss")
        scaler.scale(result["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        log = {"step": step, "train_loss": float(result["loss"].detach()),
               "train_accuracy": float(result["accuracy"].detach())}
        if step % config["eval_interval"] == 0 or step == end:
            model.eval()
            totals = np.zeros(2)
            with torch.inference_mode():
                for offset in range(0, validation_data.size, batch_size):
                    indices = np.arange(offset, min(offset + batch_size, validation_data.size))
                    batch = validation_data.batch(indices, context_steps=context, device=device)
                    with autocast():
                        output = model(batch)
                    totals += np.array([float(output["loss"]), float(output["accuracy"])]) * len(indices)
            loss, accuracy = totals / validation_data.size
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            log.update(validation_loss=float(loss), validation_accuracy=float(accuracy))
            if loss < best_loss:
                best_loss = float(loss)
                best_payload = {**payload(step), "model": {
                    key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}}
                save_payload(run_dir / "best-model.pt", best_payload)
        if step % config["log_interval"] == 0 or "validation_loss" in log:
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log, allow_nan=False) + "\n")
            print(json.dumps(log), flush=True)
        if step % config["checkpoint_interval"] == 0 or step == end:
            saved = {**payload(step), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "device_type": device.type, "best_payload": best_payload}
            save_payload(run_dir / "last.pt", saved)
            write_json(run_dir / "progress.json", {"step": step, "planned_steps": steps,
                "complete": step == steps, "best_validation_loss": best_loss if np.isfinite(best_loss) else None})
    if end == steps:
        save_payload(run_dir / "model.pt", payload(end))
    return run_dir / "best-model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = train_run(read_study(args.config), project_path(args.dataset), project_path(args.run_dir),
        args.method, args.seed, device=args.device, resume=args.resume, stop_after=args.stop_after)
    print(f"Best checkpoint: {result}", flush=True)


if __name__ == "__main__":
    main()
