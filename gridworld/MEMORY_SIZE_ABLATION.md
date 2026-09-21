# Darkroom long-term memory size ablation

The `darkroom-memory-size-v1` protocol varies `n_compress_tokens` (called
`n_latents` in the launcher/results) over **3, 6, 15, 30, 60**, with training
seeds **0, 1, 2**. Each size/seed gets its own 40,000-update compressor pretraining
and 100,000-update policy training. The baseline is 15 latents of width 64.
The experiment uses the AE compressor and `gru_gate` updates.

## Controlled history and compression schedule

`always_use_latent_prefix: False` applies to **both phases**. Before the first
compression the policy sees only raw history; subsequent calls include the real
compressed memory. No null tokens are prepended to the first compressor input.

The opt-in `first_recent_capacity` and `recurrent_recent_capacity` fields specify
raw-history capacity in environment timesteps, independently of latent count.
Policy training/evaluation use 30 before the first compression and 25 afterwards,
retaining 5 recent timesteps after each compression. Crossing a capacity triggers
compression; reaching it exactly does not. These limits apply to recurrence,
compression counting, gradient truncation, and curriculum-limited truncation.
Dataset compression buckets use the same limits. The 15-latent policy reproduces
the legacy `n_transit=30`, `short_memory_keep=5`, disabled-prefix behavior.

The transformer allocates `max(90, 75 + n_latents)` token positions: 90, 90, 90,
105, and 135 for the default sizes. Width/depth remain fixed. This experiment
holds raw-history access and compression timing fixed; it does not hold total
attention cost or total parameter count fixed. Both costs are reported.

Pretraining reconstructs a **single 40-timestep window directly**, as in the
existing Darkroom pretraining entrypoint. It does not invoke recurrent policy
compression, so the explicit 35/35 recent-history capacities in its config do
not change that training window. The disabled null-prefix setting is saved and
checked during checkpoint transfer.

Existing model configs retain their original capacity rules. This protocol has
its own configuration and identity; it does not reuse compressor-comparison runs.

## Data and matching

Use the existing PPO HDF5 histories collected with seed 0. `collection_order`
mapping and the dataset audit verify group-to-goal rewards and enforce the same
73 training goals and 8 held-out goals. Missing, too-short, or inconsistent
histories fail explicitly. Defaults require 100 source streams and 1,000 source
timesteps per goal. The source file is opened read-only.

Model/training seeds are separate from task split and collection seeds. Dedicated
sampler, window-selection, and DataLoader RNGs pair training data across sizes;
fixed recent capacities make compression-bucket lengths identical as well.
Curriculum, optimizer settings, effective batch size, source histories, and
training updates remain fixed. Keep worker counts and numerical precision equal
across sizes. The first implementation uses one process/GPU per run and disables
compilation. AMP-skipped updates retry the same batch before advancing the budget.

## Running

Run these commands from `gridworld/` with its training dependencies installed.
`--traj-dir` defaults to `datasets/`; `--runs-root` defaults to
`runs/memory_size_darkroom/`.

```bash
# Review the 15 paired pretrain/train runs and final evaluation command.
python scripts/run_memory_size_comparison.py --stage all --dry-run

# End-to-end pilot for every size, first training/evaluation seed only.
# Uses a separate pilot/ directory: 100 updates per phase, batch size 8,
# followed by 10 evaluation episodes. These are pipeline checks, not results.
python scripts/run_memory_size_comparison.py --stage pilot --gpu 0

# Full sweep: each size/seed pretrains, trains, then all policies are evaluated.
python scripts/run_memory_size_comparison.py --stage all --gpu 0
```

Alternatively use `--stage pretrain`, then `--stage train`, then `--stage evaluate`.
The train stage requires existing matching `pretrain-final.pt` files. Size lists,
training seeds, and the runs root must match between stages. `--stage all` performs
all three phases; it does not run a pilot automatically.

Use `--sizes`, `--seeds`, `--eval-seeds`, `--num-workers`, and `--gpu` to select
work. Run disjoint training seeds on separate GPUs with separate runs roots,
then assemble the completed run directories under a common root for evaluation.
Each run is named, for example, `RAD-darkroom-memory15-split0-train0`; pretraining
uses `RAD-pretrain-darkroom-memory15-split0-train0`. Fresh runs/manifests reject
collisions instead of silently resuming or overwriting an experiment. Checkpoint
transfer rejects mismatched sizes, seeds, protocols, task identities, or prefix
settings. A checkpoint from a different size must not be resized or reused.

For a CPU pipeline check add `--cpu --num-workers 0` to the pilot command.
CPU is not supported for a production training launch. `--config` accepts a
custom YAML for diagnostic pilots; keep all variants on the same resolved config.
Check the largest size on the target GPU at production batch sizes (512 pretrain,
256 policy) before committing to the sweep. If they do not fit, use the same lower
batch sizes across every size and record the changed protocol settings.

## Evaluation and artifacts

```bash
python scripts/evaluate_memory_size_comparison.py \
  --runs-root runs/memory_size_darkroom --device cuda
```

Training inherits `save_best_model: True` from regular RAD. The evaluator always
loads `best-model.pt`, selected by the highest mean reward from regular RAD's
in-training test-goal evaluation (the same model used by `evaluate_rad.py --use_best`).
The selected step can precede the final 100,000 updates; the final checkpoint must
still exist as the completed-budget artifact. There is no fallback to final weights.
The evaluator checks the actual 40,000-update pretrained checkpoint against its recorded SHA256.
It validates shared configuration/data identities and loads model state strictly.
Reduced training budgets require `--pilot --checkpoint-step N --pretrain-steps N`.
Here `--checkpoint-step` specifies the training budget, not the selected best step.
Short pilot training caps the online evaluation interval at the pilot budget so
that it also saves a best model. Existing compressor pretraining can be reused;
old policy runs with best-model saving disabled need to be retrained.
Repeated evaluations require a fresh `--output-dir`.

Each policy runs 100 consecutive episodes on all 8 held-out goals with the same
20 evaluation seeds and stochastic action sampling. Memory persists across
episodes of each task and resets for independent evaluation trials. Compression
event boundaries must agree across sizes and seeds.

Artifacts under `comparison/` include:

- `protocol.json`: best-checkpoint selection, evaluation settings, device, and software version.
- `memoryN-trainS.npz`: returns with axes `[evaluation seed, goal, episode]`, goals,
  evaluation seeds, compression counts/events, checkpoint path, selected step, and selection reward.
- `per_training_seed.csv`: overall, first-10, first-50, last-20, and after-50
  episode returns; parameter counts; first/recurrent compression latency;
  evaluation and training runtime; GPU peak memory; final pretraining MSE;
  best-checkpoint path, selected step, and selection reward.
- `summary.csv`: mean, standard deviation, and SEM **across training seeds**,
  after averaging task/evaluation trials within each training run.
- `paired_vs_15.csv` and `paired_summary_vs_15.csv`: paired return differences
  against the 15-latent run with the same training seed (when included).
- `adaptation.csv`, `adaptation.png`, `adaptation.pdf`: one curve per size with
  SEM across training seeds. Single-seed pilots have no uncertainty band.

Pretraining MSE is the final training-batch diagnostic, not a held-out estimate.
Compression timings use eight environments, warmup, and device synchronization.
GPU memory is reported as zero on CPU. Synthetic/CPU pilots establish pipeline
correctness only; convergence and production CUDA memory fit need real GPU runs.

## Focused validation

From the repository root, with training and pytest dependencies available:

```bash
python -m pytest tests/test_gridworld_memory_size_comparison.py \
  tests/test_gridworld_compressor_comparison.py \
  tests/test_gridworld_test_rad_gradient_rounds.py \
  tests/test_gridworld_test_rad_optimizer_groups.py \
  tests/test_gridworld_tokenization.py -q
```

Checks cover baseline equivalence, size-independent recurrence/buckets, identical
sampled data with zero/one workers, real gradients, absent initial null prefixes,
checkpoint transfer/rejection, and training-seed aggregation.
