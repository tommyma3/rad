# gridworld_test

## Darkroom memory-size ablation

See [MEMORY_SIZE_ABLATION.md](MEMORY_SIZE_ABLATION.md) for the 3/6/15/30/60-latent
sweep with fixed recent-history capacities, a disabled initial null prefix in both
phases, independent pretraining per size/seed, and paired held-out evaluation.

```bash
python scripts/run_memory_size_comparison.py --stage all --dry-run
python scripts/run_memory_size_comparison.py --stage pilot --gpu 0
python scripts/run_memory_size_comparison.py --stage all --gpu 0
```

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

This also saves `runs/rad_latent_update_comparison.pdf` (vector graphics with
embedded TrueType fonts) and a 400-dpi PNG. The single-panel figure compares
average episode reward against evaluation episode for the selected variants,
using each run's `eval_result.npy` array of shape `(environments, episodes)`.
Each point is the mean across evaluation environments, without smoothing or
averaging over episodes. Curves retain their own evaluation lengths; missing
results are reported and skipped. No figure is written if all results are missing.
Colors and line styles distinguish methods, with a shared legend below the axes.
Use `--plot_output figures/latent_updates` to choose the PDF/PNG filename stem,
`--variants replace gru_gate` to select methods, or `--no_plot` for summary only.
Relative output paths are resolved against the `gridworld` directory.

## DPT and IDT baselines (GPT-2)

The baseline protocols follow [dicp/gridworld](https://github.com/jaehyeon-son/dicp/tree/e36b3f713fdca775728525dc621820a1989691f8/gridworld),
with the existing `GPT2Transformer` replacing TinyLlama. The implementations live
in `model/dpt.py`, `model/idt.py`, `baseline_dataset.py`, and `train_baseline.py`.
AD/RAD model definitions, datasets, collators, training entrypoints, GPT-2 code,
and state-dict layouts are unchanged. The model registry adds two new names.
New checkpoints have distinct DPT/IDT run directories and tokenization markers;
AD/RAD weights are not initialization checkpoints for these baselines.

| Method | Tokens and supervision | Context preset (Darkroom / DKTD) |
| --- | --- | --- |
| AD/RAD | Existing separate state, action, reward tokens | Unchanged |
| DPT | Padded query state first, then one packed `(state, one-hot action, reward, next state)` token per transition; every nonempty context prefix predicts the optimal action at the query state | `n_transit=80 / 100`, including one query token |
| IDT | Review transformer sums transition embeddings; high-level GPT-2 uses `(return-to-go, state, reviewed decision)`; low-level GPT-2 uses `(sampled decision, state, action, reward)` | `n_transit=80 / 100` raw transitions, 10 actions per decision, latent dimension 8 |

IDT sorts complete episodes by return within each task/source stream and relabels
return-to-go to the stream's best episode return, as in the reference. It requires
the source timestep count to be divisible by the episode horizon, and both the
horizon and context length to be divisible by `low_per_high`. Training samples
Gaussian decisions; greedy evaluation uses their means. The presets match AD's
GPT-2 width/depth and optimizer settings, rather than claiming equal parameter
counts or compute: IDT has three transformers and DPT uses fewer tokens.

Run from `gridworld/`:

```bash
accelerate launch train_baseline.py --env darkroom --config dpt_dr
accelerate launch train_baseline.py --env darkroom --config idt_dr
accelerate launch train_baseline.py --env dktd --config dpt_dktd
accelerate launch train_baseline.py --env dktd --config idt_dktd
```

Use `--env_split_seed N` for a different train/test task split. The presets assume
histories collected with seed 0 for Darkroom and seed 2 for DKTD; override
`--collection-env-split-seed N` if collection used a different seed. This is
separate from the training/evaluation task split. `--traj-dir`, `--runs-root`,
`--mixed-precision no|fp16|bf16`, and `--train-timesteps` are also supported.
`--config` accepts either a preset name or a YAML file path. The new training
entrypoint supports Accelerate gradient accumulation and optional dynamics
losses (`dynamics: true`); default training uses action loss only.

```bash
accelerate launch train_baseline.py --resume runs/DPT-darkroom-seed0/ckpt-10000.pt
```

Resume restores model, optimizer, scheduler, and update count. It starts a new
shuffled data iteration and does not promise bitwise replay of the original RNG
or dataloader position. Checkpoints are saved atomically and retained at each
checkpoint interval; fresh training refuses a nonempty run directory.

### Existing source histories and task indexing

DPT reads existing `optimal_actions` labels in the reference's `(stream, time)`
layout when available. Otherwise it derives oracle labels from the collector's
task order and validates predicted rewards against recorded true rewards. DKTD
labels reconstruct key possession before each action, resetting it each episode.
An inconsistent collection seed/reward history fails explicitly. Data files are
opened read-only. Both new datasets reconstruct actual terminal next states from
the deterministic transition, since the current collector stores reset states
at terminal steps. Evaluation also keeps terminal transitions separate from
the next episode's reset query.

The current collector numbers its **shuffled** task list, whereas the existing
AD/RAD datasets select file groups as if their IDs were canonical task IDs. The
new baseline datasets and source plotting invert the collection permutation so
their train/test tasks match environment evaluation. Existing AD/RAD data paths
are deliberately unchanged. Consequently, old AD/RAD runs produced by this
collector/dataset combination may overlap evaluation tasks: these plots alone
do not establish a fair held-out comparison. Audit those runs' task splits before
using the comparison as experimental evidence.

### Five-method reward curves

The existing script name remains supported. It now defaults to all five curves:

```bash
python scripts/evaluate_ad_rad_curves.py --dry-run
python scripts/evaluate_ad_rad_curves.py --methods RAD AD DPT IDT SOURCE --eval-seeds 0 1 2 --eval-episodes 100
```

RAD uses `best-model.pt`; AD/DPT/IDT discover `ckpt-*.pt`. All evaluations are
recorded in CSV, but the plot uses only the latest checkpoint per training run
to avoid pooling successive checkpoints as independent trials. Model evaluation
caches include checkpoint identity, requested duration, and action-sampling mode.

The dashed source curve reads true rewards from HDF5 for the same held-out tasks,
sums rewards into episodes per source stream, and is labeled `Source PPO
(training)`. Its x-axis counts source training episodes, while transformer curves
count in-context evaluation episodes; it is not frozen-PPO policy evaluation or
a compute-matched comparison. Bands show standard deviation over task/stream or
task/evaluation trials. Source curves stop at the available complete episodes.

```bash
python scripts/evaluate_ad_rad_curves.py --source-history darkroom=./datasets/history_darkroom_PPO_alg-seed0.hdf5 --source-history dktd=./datasets/history_dktd_PPO_alg-seed0.hdf5
```

Use `--datasets-root` to change the default source directory and
`--collection-env-split-seed` for legacy checkpoint configs lacking collection
metadata. Missing source histories/tasks fail explicitly; omit `SOURCE` from
`--methods` for transformer-only plots. A method without checkpoints is absent
from the plot. Baseline evaluation supports stochastic or `--greedy` policies;
the reference's optional dynamics beam search is not implemented.

Focused CPU validation from the repository root:

```bash
gridworld/.venv/Scripts/python.exe tests/test_gridworld_baselines.py -v
```

These checks cover token causality, gradients, oracle labels, task mapping,
episode handling, strict checkpoint round trips, training/save/resume, and
five-method plotting. They do not establish convergence or multi-GPU correctness.

## Darkroom and DKTD tokenization ablation

See [TOKENIZATION_ABLATION.md](TOKENIZATION_ABLATION.md) for the isolated
`AD_DPT` and `RAD_DPT` query-last variants in both environments, variable-length RAD sampling,
pretraining/training commands, and the four-method evaluator. Existing AD/RAD
training and inference entrypoints remain unchanged.
