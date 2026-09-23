# Old-versus-recent evidence integration

This isolated diagnostic tests whether RAD combines compressed old evidence with
recent observations. It supplies controlled exploration histories and trains a
single final arm decision. It does not change the standard bandit environment,
UCB collector, datasets, training entrypoints, or checkpoint schema.

## Protocol

The dedicated configuration is `config/experiments/old_recent.yaml` and the
artifact identifier is `old-recent-evidence-v1`.

1. Draw ten independent arm means from `Uniform[0,1]`, with fresh Gaussian rewards
   of standard deviation 0.3. Randomly partition the arms into two groups of five.
2. Supply 25 early pulls: five observations of each early-group arm, in randomized
   order. These forced exploration actions are context, never prediction targets.
3. Supply 72 distractor transitions, with observation `DISTRACTOR`, independent
   uniform actions, and zero reward.
4. Supply 25 recent pulls: five observations of each other arm, in randomized order.
5. Query one action over all ten arms. There is no target action/reward in the
   input and no opportunity to relearn before this decision.

With `context_steps=50` and `short_memory_keep=5`, RAD compresses after transitions
51 and 97. The recent block starts after transition 97. At the query, all early
evidence resides in latent memory, while all 25 recent observations remain raw,
along with five distractors. There is no compression inside the recent block.
The geometry validator enforces these conditions when the settings change.
`gap_compressions` controls the number of events and derives the gap length;
changing it is a separate study, not a fair isolated gap-length intervention.

The supervised target is the arm with the largest empirical reward mean from
both blocks, computed from the actual observed rewards. All arms have equal
sample counts, so it is also the existing DPT-UCB recommendation (the exploration
bonuses are equal). Ground-truth means never supply training labels or model
inputs. They are used to score decisions and orient the random arm partition so
exactly half of the independent tasks have their true best arm in each block.
The orientation/stratum and task identity are not model inputs.

Train, validation, and test use separate deterministic random streams. Every
method/seed uses the same immutable files and their SHA-256 digests. Defaults
are 10,000/1,000/2,000 tasks. The standard environment's current arm-count setting
is intentionally not inherited: this balanced diagnostic needs equal subsets.

## Training and interventions

The default sweep trains RAD, AD-short, and AD-long for seeds 0, 1, and 2:

| Method | Raw context capacity | Additional recurrent memory |
| --- | ---: | --- |
| RAD | 50 transitions | 15 latent tokens |
| AD-short | 50 transitions | None |
| AD-long | 122 transitions (entire prefix) | None |

All runs start from scratch, including RAD's compressor, with the same number of
optimizer updates and examples per update (20,000 and 64 by default). No separate
compression pretraining is included. RAD uses the existing GRU update, full
gradient propagation through compression, and no initial null latent prefix.
Training seeds and data are matched; parameters, token counts, and compute are
not equal across methods. The decoder is frozen and unused.

Each trained checkpoint is evaluated under three **inference interventions**:

- `both`: the original history.
- `early_only`: replace every recent-block transition with a distractor.
- `recent_only`: replace every early-block transition with a distractor.

Replacement changes the observation, action, and reward. Independent precomputed
filler actions prevent the removed arm identities from leaking through actions.
Sequence length, query position, compression count, and the remaining evidence
are identical across conditions. All models are trained on `both`; the other
conditions involve a distribution shift. They diagnose reliance on information
but are not separately trained optimal policies for partial evidence.

AD-short must have exactly the same output for `both` and `recent_only`, since
the entire removed block is already outside its raw context. This is tested.
The both-block empirical recommendation and the available-block empirical
recommendation are included as references, along with uniform random choice.

## Run independent jobs across GPUs

From the repository root, on a machine with CUDA-enabled Torch installed in the
bandit environment:

```bash
uv run --project bandit python bandit/scripts/run_old_recent_evidence.py --gpus 0 1 2 --seeds 0 1 2
```

The launcher prepares data once and runs up to three jobs concurrently, assigning
one process to each GPU. It queues the remaining method/seed combinations as
devices become free. **Do not wrap this command in `accelerate launch` or
`torchrun`.** This is parallelism across independent training runs, not DDP within
one run. Each child sees one CUDA device and inherited distributed-launch state
is removed. If `CUDA_VISIBLE_DEVICES=3,7` is inherited, `--gpus 0 1` selects those
two devices. GPU UUIDs are supported. No other processes on those GPUs are stopped.

Useful options:

```bash
# Inspect all nine commands without writing files or requiring CUDA.
uv run --project bandit python bandit/scripts/run_old_recent_evidence.py --gpus 0 1 2 --dry-run

# Resume the same study; completed runs are validated and skipped.
uv run --project bandit python bandit/scripts/run_old_recent_evidence.py --gpus 0 1 2 --seeds 0 1 2 --resume

# A separate study with bfloat16 and a different output root.
uv run --project bandit python bandit/scripts/run_old_recent_evidence.py --gpus 0 1 --mixed-precision bf16 --root runs/evidence_bf16

# Bounded CPU pipeline smoke test; not a learning-performance result.
uv run --project bandit python bandit/scripts/run_old_recent_evidence.py --cpu --seeds 0 --steps 2 --batch-size 2 --train-tasks 8 --validation-tasks 4 --test-tasks 4 --eval-interval 1 --checkpoint-interval 1 --root runs/evidence_cpu_smoke
```

`--methods rad` restricts the sweep. `--no-evaluate` runs training only.
`--config` accepts a separate study YAML/JSON. Relative data, config, and output
paths resolve under `bandit/`. Changing overrides on resume is rejected: repeat
the original flags, or pass the saved `study.json` as the configuration.
Failed workers stop the queue, terminate its other active workers, and identify
the failing log. Resume restores model, optimizer, scheduler, AMP scaler, data
sampling RNG, Torch RNG, and the best validation snapshot. A run interrupted
before its first checkpoint restarts from its original seed. Exact CPU resume is
tested; GPU numerical determinism is not established by that test.

## Outputs and interpretation

The default root is `bandit/runs/old_recent_evidence_v1/`:

- `plan.json`, `study.json`, `data/manifest.json`: frozen settings and data hashes.
- `<method>_s<seed>/metrics.jsonl`: training and full-validation metrics.
- `<method>_s<seed>/best-model.pt`: checkpoint selected by both-block validation loss.
- `<method>_s<seed>/model.pt`: completed-budget inference checkpoint.
- `<method>_s<seed>/last.pt`: resumable training state.
- `logs/`: one log per run, evaluation log, and subprocess completion records.
- `evaluation/`: per-task JSONL, summary JSON/CSV, paired intervention effects,
  checkpoint hashes, and a two-panel PNG/vector PDF.

Evaluation runs after all training workers succeed, on the first listed GPU.
It uses the best validation checkpoints. The test set is never used for checkpoint
selection. `evaluation_result.json` records the result directory; interrupted
evaluation attempts are retained and retries use a new directory.

The primary score is first-query expected pseudo-regret,
`max(mu) - sum_a pi(a | history) * mu[a]`, computed exactly without extra reward
noise or action-sampling noise. Greedy regret, optimal-arm probability/accuracy,
and empirical-teacher probability/accuracy are also saved. Summaries split tasks
by whether the true best arm appeared early or recently. Plot error bars are one
standard error across training-seed means when multiple seeds are present; for
one seed, they reflect test-task variability only. `uncertainty_unit` labels this.
There are no separate evaluation seeds because the test evidence and probabilities
are deterministic once the data and checkpoint are fixed.

`paired_effects.json` reports each partial-evidence score minus its matched
both-block score. Positive regret differences favor access to both blocks. A
useful integration result would combine low regret in both best-arm strata with
paired improvements over the relevant partial-evidence controls. These are
questions for the trained results; passing pipeline tests does not establish
RAD's advantage, convergence, or target-GPU memory fit.

To evaluate explicit experiment checkpoints separately:

```bash
uv run --project bandit python bandit/evaluate_evidence.py --checkpoint runs/old_recent_evidence_v1/rad_s0/best-model.pt runs/old_recent_evidence_v1/ad_short_s0/best-model.pt runs/old_recent_evidence_v1/ad_long_s0/best-model.pt --dataset runs/old_recent_evidence_v1/data --output runs/evidence_recheck --device cuda
```

These entrypoints reject legacy checkpoints. The standard bandit loader likewise
does not accept this experiment's independent checkpoint protocol.

## Verification

```bash
uv run --project bandit python -m unittest bandit.tests.test_evidence_experiment -v
```

Tests cover balanced/disjoint task splits, observed-only supervision, equal arm
counts, causal interventions, compression geometry, gradients to early rewards,
AD-short invariance, data tampering, exact CPU resume, all three model families,
paired summaries, duplicate-checkpoint rejection, GPU visibility mapping, actual
subprocess overlap across two simulated device slots, and failure cleanup.
The queue tests do not execute CUDA kernels.
