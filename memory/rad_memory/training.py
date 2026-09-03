"""Shared Accelerate training loops for AD and RAD."""

from __future__ import annotations

from pathlib import Path
import json
import time

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from .dataset import (
    ADDataset,
    CompressionBucketBatchSampler,
    CompressionPretrainDataset,
    RADDataset,
    collate_trajectories,
)
from .model import AD, RAD
from .optimizer_utils import build_rad_optimizer_param_groups, freeze_reconstruction_decoder_for_finetuning
from .utils import latest_checkpoint, save_checkpoint_atomic, seed_everything


def make_loader(dataset, config: dict, *, batch_size_key: str, shuffle: bool) -> DataLoader:
    workers = int(config.get("num_workers", 0))
    common = {
        "collate_fn": collate_trajectories,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if isinstance(dataset, RADDataset) and config.get("rad_batching_strategy", "compression_buckets") == "compression_buckets":
        sampler = CompressionBucketBatchSampler(
            dataset,
            int(config[batch_size_key]),
            shuffle=shuffle,
            seed=int(config.get("seed", 0)),
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=int(config[batch_size_key]),
        shuffle=shuffle,
        **common,
    )


def _infinite(loader):
    while True:
        yield from loader


def train_distillation(
    config: dict,
    data_root: str | Path,
    run_dir: str | Path,
    *,
    model_kind: str,
    pretrain_checkpoint: str | Path | None = None,
) -> Path:
    seed_everything(int(config.get("seed", 0)))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision=str(config.get("mixed_precision", "no")),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
    )
    config = dict(config) | {"model": model_kind}
    if model_kind == "AD":
        model = AD(config)
        dataset = ADDataset(config, data_root, "train")
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.99))),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
    elif model_kind == "RAD":
        model = RAD(config)
        if pretrain_checkpoint is not None:
            checkpoint = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            allowed_missing = {
                name for name in missing
                if name.startswith(("transformer.", "pred_action.", "latent_"))
            }
            if set(missing) - allowed_missing or unexpected:
                raise ValueError(
                    f"Compression checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
        freeze_reconstruction_decoder_for_finetuning(model)
        dataset = RADDataset(config, data_root, "train")
        optimizer = AdamW(
            build_rad_optimizer_param_groups(model, config),
            betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.99))),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    loader = make_loader(dataset, config, batch_size_key="train_batch_size", shuffle=True)
    total_steps = int(config["train_steps"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.get("warmup_steps", 0)),
        total_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    writer = SummaryWriter(run_dir) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True, default=str)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step = 0
    checkpoint_path = latest_checkpoint(run_dir)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])

    batches = _infinite(loader)
    model.train()
    while step < total_steps:
        batch = next(batches)
        with accelerator.accumulate(model):
            output = model(batch)
            loss = output["loss_total"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if writer is not None and step % int(config.get("summary_interval", 100)) == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/action_accuracy", output["acc_action"].item(), step)
            writer.add_scalar("train/decision_accuracy", output["acc_decision"].item(), step)
            if "num_compressions" in output:
                writer.add_scalar("train/compressions", output["num_compressions"].item(), step)
        if accelerator.is_main_process and step % int(config.get("checkpoint_interval", 1000)) == 0:
            save_checkpoint_atomic(
                {
                    "step": step,
                    "config": config,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                run_dir / f"ckpt-{step}.pt",
            )
    accelerator.wait_for_everyone()
    final_path = run_dir / "final.pt"
    if accelerator.is_main_process:
        save_checkpoint_atomic(
            {
                "step": step,
                "config": config,
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            final_path,
        )
        elapsed = time.perf_counter() - started
        with (run_dir / "run_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "elapsed_seconds": elapsed,
                    "steps": step,
                    "steps_per_second": step / max(elapsed, 1e-9),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in accelerator.unwrap_model(model).parameters()
                        if parameter.requires_grad
                    ),
                    "peak_gpu_memory_bytes": (
                        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                    ),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        if writer is not None:
            writer.close()
    dataset.close()
    return final_path


def train_compression(config: dict, data_root: str | Path, run_dir: str | Path) -> Path:
    seed_everything(int(config.get("seed", 0)))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision=str(config.get("mixed_precision", "no")),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
    )
    model = RAD(config)
    dataset = CompressionPretrainDataset(config, data_root, "train")
    loader = make_loader(dataset, config, batch_size_key="pretrain_batch_size", shuffle=True)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["pretrain_lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_steps = int(config["pretrain_steps"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.get("pretrain_warmup_steps", 0)),
        total_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    batches = _infinite(loader)
    for step in range(1, total_steps + 1):
        batch = next(batches)
        with accelerator.accumulate(model):
            output = model(batch, pretrain_compression=True)
            accelerator.backward(output["loss_total"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.is_main_process and step % int(config.get("checkpoint_interval", 1000)) == 0:
            save_checkpoint_atomic(
                {"step": step, "config": config, "model": accelerator.unwrap_model(model).state_dict()},
                run_dir / f"ckpt-{step}.pt",
            )
    accelerator.wait_for_everyone()
    final_path = run_dir / "pretrain-final.pt"
    if accelerator.is_main_process:
        save_checkpoint_atomic(
            {"step": total_steps, "config": config, "model": accelerator.unwrap_model(model).state_dict()},
            final_path,
        )
    dataset.close()
    return final_path
