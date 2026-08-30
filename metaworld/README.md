# Meta-World RAD

This package mirrors the training, compression, checkpoint, and evaluation
structure in `gridworld_test` while using Meta-World ML1 tasks and continuous
actions.

## Data collection

From this directory:

```bash
uv run python collect.py
```

Trajectories are written below `datasets/<task>/`, with the official ML1 test
tasks in the `test/` subdirectory.

## Training

```bash
uv run accelerate launch train.py --config ad_ml1
uv run accelerate launch train_pretrain_compression.py --config rad_pretrain_ml1
uv run accelerate launch train_rad.py --config rad_ml1
```

AD and RAD use interleaved state/action/reward tokens. RAD pretraining and
fine-tuning checkpoints use the `metaworld-sar-v1` format; legacy packed-
transition checkpoints are intentionally rejected.

## Evaluation

```bash
uv run python evaluate.py --ckpt_dir runs/AD-ml1-window-open-v3-seed0
uv run python evaluate_rad.py --ckpt_dir runs/RAD-ml1-window-open-v3-seed0 --use_best
```

Reward arrays are saved to `eval_result.npy` and success arrays to
`eval_success.npy`.

## Latent-update comparison

```bash
uv run python scripts/run_rad_latent_update_comparison.py --gpus 0 1 2 3 --dry-run
uv run python scripts/summarize_rad_latent_update_comparison.py
```

The available variants are `replace`, `residual`,
`multiplicative_gate`, and `gru_gate`.
