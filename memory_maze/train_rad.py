"""Fine-tune visual RAD with action loss only."""

from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import RADDataset
from model import RAD
from optimizer_utils import build_rad_optimizer_param_groups, freeze_reconstruction_decoder_for_finetuning
from train_common import infinite_batches, make_loader, save_checkpoint_atomic
from utils import checkpoint_payload, get_config, set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrained-compression", type=Path, required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    set_all_seeds(int(config.get("seed", 42)))
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
    )
    dataset = RADDataset(config, Path(config.get("dataset_root", "datasets")), "train")
    loader = make_loader(dataset, config, "train_batch_size")
    model = RAD(config)
    model.load_pretrained_compression(str(args.pretrained_compression))
    freeze_reconstruction_decoder_for_finetuning(model)
    optimizer = AdamW(
        build_rad_optimizer_param_groups(model, config),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.get("num_warmup_steps", 1000)),
        int(config["train_timesteps"]),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    batches = infinite_batches(loader)
    run_dir = Path(config.get("run_dir", f"runs/RAD-{config['source_algorithm']}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    curriculum = sorted(
        config.get("curriculum_schedule", [{"step": 0, "max_compressions": None}]),
        key=lambda stage: int(stage["step"]),
    )
    current_stage = None
    for step in range(1, int(config["train_timesteps"]) + 1):
        stage = max(
            (item for item in curriculum if int(item["step"]) <= step),
            key=lambda item: int(item["step"]),
        )
        stage_id = int(stage["step"])
        if stage_id != current_stage:
            dataset.max_compressions = stage.get("max_compressions")
            accelerator.unwrap_model(model).set_curriculum(stage.get("max_compressions"))
            current_stage = stage_id
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
            save_checkpoint_atomic(
                checkpoint_payload(
                    accelerator.unwrap_model(model),
                    config,
                    step=step,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                run_dir / f"ckpt-{step:09d}.pt",
            )


if __name__ == "__main__":
    main()
