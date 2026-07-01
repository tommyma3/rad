# gridworld_test

## RAD Latent-Update Comparison

This experiment compares RAD-only latent update variants on dark key-to-door:

- `replace`: current RAD behavior, replacing old latents with compressor output.
- `residual`: layer-normalized `old_latents + candidate_latents`.
- `multiplicative_gate`: `sigmoid(W old_latents) * candidate_latents`.
- `gru_gate`: gated interpolation between old latents and a candidate state.

Run all variants in parallel on separate GPUs:

```bash
python scripts/run_rad_latent_update_comparison.py --gpus 0 1 2 3
```

The runner first creates or reuses the shared pretraining checkpoint at
`runs/RAD-pretrain-dktd-seed0/pretrain-final.pt`, then launches one fine-tuning
process per variant with `CUDA_VISIBLE_DEVICES` set to the corresponding GPU
index. Variant run directories are:

- `runs/RAD-dktd-seed0-replace`
- `runs/RAD-dktd-seed0-residual`
- `runs/RAD-dktd-seed0-multiplicative_gate`
- `runs/RAD-dktd-seed0-gru_gate`

Inspect commands without running them:

```bash
python scripts/run_rad_latent_update_comparison.py --gpus 0 1 2 3 --dry-run
```

Summarize completed runs:

```bash
python scripts/summarize_rad_latent_update_comparison.py --output runs/rad_latent_update_summary.csv
```
