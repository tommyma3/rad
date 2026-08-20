"""Pretrain RAD's recurrent compression bottleneck in token space."""

from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import CompressionPretrainDataset
from model import RAD
from train_common import infinite_batches, make_loader, save_checkpoint_atomic
from utils import checkpoint_payload, get_config, set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    set_all_seeds(int(config.get("seed", 42)))
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "bf16"))
    dataset = CompressionPretrainDataset(
        config, Path(config.get("dataset_root", "datasets")), "train"
    )
    loader = make_loader(dataset, config, "pretrain_batch_size")
    model = RAD(config)
    parameters = (
        list(model.image_encoder.parameters())
        + list(model.embed_action.parameters())
        + list(model.embed_reward.parameters())
        + [model.type_embedding]
        + list(model.compression_transformer.parameters())
        + list(model.reconstruction_decoder.parameters())
    )
    optimizer = AdamW(
        parameters,
        lr=float(config["pretrain_lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.get("pretrain_warmup_steps", 1000)),
        int(config["pretrain_timesteps"]),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    batches = infinite_batches(loader)
    run_dir = Path(config.get("pretrain_run_dir", f"runs/RAD-pretrain-{config['source_algorithm']}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, int(config["pretrain_timesteps"]) + 1):
        with accelerator.autocast():
            output = accelerator.unwrap_model(model).forward_pretrain_compression(next(batches))
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(output["loss_recon"])
        accelerator.clip_grad_norm_(parameters, float(config.get("grad_clip", 1.0)))
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
