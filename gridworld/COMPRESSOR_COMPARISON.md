# Darkroom compressor comparison

Compare Transformer AE, Transformer VAE, and Transformer VQ-VAE using the same
policy, Transformer compressor backbone, decoder, compression schedule, and
GRU-style memory update. This ablates the bottleneck **and its training loss**.

## Fixed protocol

- Configs: `rad_dr_ae`, `rad_dr_vae`, `rad_dr_vq_vae`, inheriting `rad_dr.yaml`.
- Darkroom: 9 x 9, 20 steps/episode, goal split seed 0 (73 train / 8 test).
- Independent training seeds: 0, 1, 2. Each run has its own pretrained compressor.
- Policy: 4 layers, 4 heads, width 64. Compressor: 3 layers, 4 heads, width 64.
- Memory: 15 x 64; 90-token policy capacity; retain 5 recent transitions;
  `gru_gate`; no initial latent policy prefix; gradients through 5 recent rounds.
- Pretraining: 40,000 updates, batch 512, 40-transition windows.
- RAD: 100,000 updates, batch 256, original curriculum and learning rates.
- Source: the first 1,000 steps of 100 streams per task in the existing PPO HDF5.
- One process/GPU per run, sequential launches, gradient accumulation 1.
- All variants run eagerly (`torch_compile: False`) for this first comparison.
  This avoids comparing different compilation/fallback behavior. CUDA performance
  and convergence must still be measured on the training machine.

Shared parameters have identical initialization for a paired training seed.
Additional heads/codebook are initialized without advancing the shared model RNG.
The variants do not have exactly the same parameter count; counts are reported.
Equal token shapes do not imply equal information capacity.

Budgets count successful optimizer updates. AMP overflow retries use the same
batch, dropout RNG, and latent noise (up to ten attempts); retries are logged.

## Bottlenecks and objectives

Let `h` be the Transformer query outputs, each with 64 features.

| Variant | Candidate memory | Pretraining | RAD training |
| --- | --- | --- | --- |
| AE | `h` | embedding reconstruction MSE | action cross-entropy |
| VAE | `mu(h) + exp(logvar(h)/2) * epsilon` | reconstruction + `0.001 * KL` | action + `0.001 * KL` |
| VQ-VAE | nearest codebook vector per token | reconstruction + codebook + `0.25 * commitment` | action + codebook + `0.25 * commitment` |

VAE uses a diagonal Gaussian and standard-normal prior. KL is averaged across
batch, tokens, and dimensions. Its coefficient ramps over the first 5,000
pretraining updates and stays fixed during RAD training. Log variance is clamped
to [-12, 8] for numerical stability. Training samples use a dedicated RNG.
Evaluation reports posterior means as the primary result and sampling separately.

VQ-VAE uses one learned 256 x 64 codebook, Euclidean nearest neighbors,
straight-through encoder gradients, and gradient-based codebook updates. Losses
are mean squared errors: `||q - stopgrad(h)||^2` for the codebook and
`||h - stopgrad(q)||^2` for commitment. There is no EMA or autoregressive prior.
Quantization distances and losses use fp32 even under autocast. Evaluation never
updates the codebook.

The candidate passes through the **unchanged** memory gate. Thus VQ compressor
outputs are quantized, while recurrent stored memory remains continuous.
Requantizing after the gate is outside this experiment.

Fine-tuning averages auxiliary losses over only the compression rounds receiving
gradients. No compression (or zero allowed gradient rounds) contributes zero
auxiliary loss. The decoder is used only during pretraining.

## Dataset identity and compatibility

The collector numbers tasks in shuffled collection order. New comparison configs
explicitly use `dataset_task_mapping: collection_order` and
`collection_env_split_seed: 0`. The original collection seed must be correct.
The loader shares the DPT/IDT task-to-group mapping. Before creating a run, a
read-only audit verifies all selected groups, source lengths/stream counts, and
Darkroom rewards against their reconstructed goals. It requires positive reward
evidence for every task; incorrect or incomplete data fails before training.
The audit hashes the selected states/actions/rewards and records the goal mapping
in checkpoint configuration. Pretraining and RAD source identities must match.
RAD checkpoints also record the exact pretrained file hash, completed update
count, and pretraining settings; the evaluator checks paired pretraining budgets.

Historical configs retain their legacy group selection unless explicitly opted
in. Retrain AE with the new comparison config: historical curves are not the
controlled baseline for this corrected split.

Missing `compressor_type` means legacy AE and retains its state-dict keys.
New variant checkpoints load strictly; cross-variant and mismatched training-seed
or task-split pretrained checkpoints are rejected. Comparison runs require fresh
directories. Interrupted training is not resumed, because exact sampler and latent
RNG continuation is not implemented in this protocol.

DataLoader generators, RAD sampling/window RNGs, VAE noise, and evaluation action
sampling are separate. All variants consume the same data stream for the same
seed and worker count. Keep worker count and hardware settings fixed.

## Run

From `gridworld/`, with the project dependencies and source HDF5 installed:

```bash
# Inspect the exact nine pretrain/train pairs without creating or running jobs.
uv run python scripts/run_compressor_comparison.py --stage train --dry-run

# Run all three short pilots at training seed 0 (100 updates per stage).
uv run python scripts/run_compressor_comparison.py --stage pilot --gpu 0

# After inspecting pilot losses, posterior statistics, and codebook usage:
uv run python scripts/run_compressor_comparison.py --stage train --gpu 0

# All eight goals, 20 evaluation seeds, 100 consecutive episodes, final checkpoint.
uv run python scripts/run_compressor_comparison.py --stage evaluate --gpu 0

# Additionally evaluate the test-selected best-model.pt checkpoints into comparison-best/.
uv run python scripts/run_compressor_comparison.py --stage evaluate --checkpoint both --gpu 0
```

The launcher defaults to `runs/compressor_darkroom`, with pilots under `pilot/`.
Use `--runs-root` for a fresh experiment, `--traj-dir` for external source data,
and `--num-workers` to override worker count consistently. `--stage all` executes
pilot, full training, and evaluation consecutively. Manifests retain exact commands,
exit status, and elapsed time. Existing stage manifests are not overwritten.

Individual entrypoints also accept `--seed`, `--runs_root`, `--run_name`,
`--traj_dir`, `--num_workers`, `--steps`, `--batch_size`, `--cpu`, and `--no_compile`.
`--config` accepts either a config name or an explicit YAML path. `--steps` and
`--batch_size` overrides are for pilots, not the main comparison.

## Evaluation and artifacts

Use the final 100k checkpoint as the controlled result; it is never selected by
test reward. Training saves `best-model.pt` anyway (`save_best_model: True`),
and the evaluator accepts `--checkpoint best` to evaluate it: the checkpoint
that maximized the in-training in-context eval on the eight test goals. Best
curves are secondary, test-selected results — they quantify how much of a
final-checkpoint gap is checkpoint selection, not how a variant trains on
average. Report the two modes side by side, never intermixed.
Evaluation retains memory across episodes and resets it for every evaluation seed.
The evaluator also checks that checkpoint source hashes and shared model/training
settings agree. All eight goals are evaluated with the same sampled-action seeds.

Outputs under `comparison/` (final checkpoint) or `comparison-best/` (best
checkpoint; the loaded step is recorded per run since it varies):

- Raw NPZ: evaluation-seed x goal x episode rewards, goal coordinates, compression
  counts, evaluation seeds, and source checkpoint path for each variant/train seed.
- `per_training_seed.csv`: mean return, episodes 1-10, 1-50, last 20, and 51-100;
  model parameter count, compression time, and peak evaluation GPU allocation.
- `summary.csv`: means, sample standard deviations, and standard errors computed
  over **training seeds**, after averaging evaluation seeds and goals within each.
- `paired_vs_ae.csv`: per-training-seed differences against AE. Mean and sampled
  VAE results remain separate throughout aggregation.
- `adaptation.csv`, `adaptation.png`, `adaptation.pdf`: unsmoothed episode-return
  curves with standard-error bands over training seeds. The 50-episode marker
  identifies the 1,000-step maximum training-window length.
- `protocol.json`: evaluation settings and software version.

Compression latency benchmarks eager fp32 inference for a batch of eight,
including the compressor and memory gate, with warmup and CUDA synchronization.
It excludes policy/environment time. Peak GPU allocation is for evaluation and
this benchmark. Each training run separately writes `pretrain-metrics.json` or
`train-metrics.json` with final losses/diagnostics, parameter counts, elapsed time,
and peak GPU allocation, plus TensorBoard scalar histories. Codebook usage and
perplexity are per compression-call batch (averaged over eligible rounds in RAD),
not a dataset-wide usage estimate.

Check pilot KL/posterior variance for collapse, codebook usage/perplexity for
degeneracy, and all losses for finiteness. Lower reconstruction MSE alone does not
establish better task memory, especially when the token embeddings are trainable.

## Validation

```bash
python -m pytest ../tests/test_gridworld_compressor_comparison.py -q
```

CPU tests cover legacy AE behavior, matched shared initialization, VAE KL and RNG,
VQ gradient paths and codebook updates, both training objectives, recent-round
loss averaging, no-compression cases, checkpoint rejection/round trips and
final/best selection rules, task-map auditing, data-stream independence, and
aggregation units. Small synthetic-history
end-to-end checks validate the training entrypoints and evaluation artifacts;
they do not establish convergence on the real PPO histories.
