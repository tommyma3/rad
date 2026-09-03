# RAD on MiniGrid Memory

This package compares bounded-context Algorithm Distillation (AD) with
Recurrent Algorithm Distillation (RAD) on MiniGrid's partially observable
Memory environments. The policy sees a key or ball, loses sight of it in a
corridor, and must later choose the matching branch.

The benchmark distinguishes three lengths:

- MiniGrid's native timeout (`5 * size**2`).
- A calibrated, solvable episode horizon `H`.
- The measured delay from the last cue observation to the branch decision.

All context settings are in environment transitions. Internally each transition
is represented by causal state, action, and reward/boundary tokens.
Each training window supervises its final query action, matching streaming
inference exactly; sliding windows provide supervision throughout episodes.
The same final/decision-window oversampling is applied to every condition so
the rare branch action is not drowned out by corridor-navigation actions.

## 1. Profile the task

Run from this directory after dependency setup completes:

```bash
uv run python -m rad_memory.profile_memory_task \
  --env-id MiniGrid-MemoryS13Random-v0 \
  --random-length \
  --episodes 1000 \
  --output profiles/memory-s13-random.json
```

The privileged profiler is used only to calibrate `H`; it does not provide
privileged observations to AD, RAD, or the recurrent PPO teacher.

For a controlled causal diagnostic with a guaranteed initially visible cue:

```bash
uv run python -m rad_memory.profile_memory_task \
  --env-id MiniGrid-MemoryS13-v0 \
  --controlled --size 13 \
  --episodes 1000 \
  --output profiles/memory-s13-controlled.json
```

## 2. Train and collect the recurrent source learner

```bash
uv run python -m rad_memory.train_teacher \
  --env-id MiniGrid-MemoryS13Random-v0 \
  --random-length \
  --horizon H \
  --seed 0 \
  --run-dir runs/teacher-s13-seed0
```

Collect the final policy and intermediate learning checkpoints into one
append-only history. Give test collection a task spec with `split: test` and a
disjoint seed range.

```bash
uv run python -m rad_memory.collect \
  --task-spec runs/teacher-s13-seed0/task_spec.json \
  --checkpoint runs/teacher-s13-seed0/teacher-checkpoint_100000_steps.zip \
  --checkpoint runs/teacher-s13-seed0/teacher-final.zip \
  --episodes-per-checkpoint 100 \
  --output-root datasets
```

The collector is resumable by `(task, learner_step)`. To create a held-out
artifact from the same teacher, add `--split test --seed 10000`.

## 3. Inspect and run the comparison

The profiler chooses a short context no larger than `0.5H` and, when needed,
shorter so that a substantial fraction of measured cue gaps lie outside it.
The runner compares reactive AD; AD at the RAD-matched short context, exactly
`0.5H`, `1.1H`, and `2H`; compute-matched short AD; RAD with the calibrated
short active context and `2H` effective history; and RAD without compressor
pretraining.

Dry-run first:

```bash
uv run python -m rad_memory.run_context_sweep \
  --profile profiles/memory-s13-random.json \
  --config config/model/memory_base.yaml \
  --seeds 0 1 2 \
  --gpus 0 1 2
```

Add `--execute` to run the sequential dependency-aware matrix. Each condition
is evaluated on the same episode seeds. Summarize completed evaluations with:

```bash
uv run python -m rad_memory.summarize_context_sweep \
  --input-root runs/context-sweep \
  --csv runs/context-sweep/summary.csv \
  --figure runs/context-sweep/success.png
```

The summary includes bootstrap confidence intervals and paired deltas against
AD with the same active context. A RAD advantage should only be claimed on
episodes where the measured cue gap exceeds the active context.

The optional `rad_memory.counterfactual_cue` command swaps key and ball in the
cue-visible prefix while holding the post-cue trajectory fixed. This checks
whether the final branch distribution causally depends on information that is
no longer in the active raw context.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The focused suite checks deterministic environment construction, append-only
artifacts, episode-bounded windows, causal masking, RAD compression, null-latent
gradients, and optimizer-group coverage.
