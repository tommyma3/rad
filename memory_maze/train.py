"""Train visual Algorithm Distillation on fixed-task source histories."""

from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import ADDataset
from model import AD
from train_common import infinite_batches, latest_checkpoint, make_loader, save_checkpoint_atomic
from utils import checkpoint_payload, get_config, normalize_state_dict, set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    set_all_seeds(int(config.get("seed", 42)))
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
    )
    dataset_root = Path(config.get("dataset_root", "datasets"))
    train_dataset = ADDataset(config, dataset_root, "train")
    test_dataset = ADDataset(config, dataset_root, "test")
    train_loader = make_loader(train_dataset, config, "train_batch_size")
    test_loader = make_loader(test_dataset, config, "test_batch_size", shuffle=False)
    model = AD(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.get("num_warmup_steps", 1000)),
        int(config["train_timesteps"]),
    )
    run_dir = Path(config.get("run_dir", f"runs/AD-{config['source_algorithm']}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(normalize_state_dict(state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    batches = infinite_batches(train_loader)
    while step < int(config["train_timesteps"]):
        step += 1
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = model(next(batches))
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(output["loss_action"])
            accelerator.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()
        if accelerator.is_main_process and step % int(config.get("checkpoint_interval", 5000)) == 0:
            unwrapped = accelerator.unwrap_model(model)
            save_checkpoint_atomic(
                checkpoint_payload(
                    unwrapped,
                    config,
                    step=step,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                run_dir / f"ckpt-{step:09d}.pt",
            )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
