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

## Fixed-task experiments

Use this workflow for Gridworld-style algorithm distillation. A task is a saved
initial grid, cue/branch objects, success/failure positions, agent pose, view size,
and horizon. Every reset restores that exact state, even if the caller supplies a
different seed. Only episode counters and other transient state restart.
The ordinary MiniGrid action space, partial observations, rewards, and termination
rules are retained. The random variant varies corridor length when constructing
the pool; it does not generate arbitrary corridor topologies.

Run these commands from `memory/` (examples use Bash continuation syntax):

```bash
uv run python -m rad_memory.task_pool \
  --env-id MiniGrid-MemoryS13Random-v0 --horizon 30 \
  --num-tasks 100 --pool-seed 0 --env-split-seed 0 --train-env-ratio 0.8 \
  --output tasks/memory_s13_fixed.json

uv run python -m rad_memory.profile_memory_task \
  --manifest tasks/memory_s13_fixed.json --output profiles/memory-s13-fixed.json

uv run python -m rad_memory.train_task_pool \
  --manifest tasks/memory_s13_fixed.json --source-seeds 0 1 2 \
  --source-algorithm ppo --workers 4 --torch-threads 1 --device cpu \
  --total-timesteps 1000000 --evaluation-interval 50000 \
  --evaluation-episodes 100 --minimum-success-rate 0.9 --required-consecutive-evals 3 \
  --run-dir runs/source-fixed --output-root datasets-fixed

uv run python -m rad_memory.run_context_sweep \
  --profile profiles/memory-s13-fixed.json --manifest tasks/memory_s13_fixed.json \
  --config config/model/memory_fixed_ppo.yaml --data-root datasets-fixed \
  --runs-root runs/fixed-sweep --eval-episodes 20 --trials 3 --execute
```

The pool generator deduplicates full configurations before splitting. Task IDs
are independent of split and learner seed; the manifest fingerprint binds pool
membership and splits. It refuses to overwrite a manifest or return fewer unique
tasks than requested. A small/controlled environment may have fewer distinct
configurations than the requested pool size; reduce `--num-tasks` in that case.
The profiler examines training tasks only. Horizon 30 is an example: check the
profile before source training, and generate a new manifest if it is unsuitable.

Each `(training task, source seed)` gets a fresh learner and one
chronological interaction stream. Its network learns across episodes; its LSTM
state resets each episode when using RecurrentPPO. Test tasks are never source-trained by this command.
Histories contain the actual training actions from exploration onward, rather
than separate rollouts of saved policies. Final incomplete episodes are omitted
and their step count recorded. Interrupted runs remain marked incomplete; fresh
learners cannot append to an existing run. Use a new run/data directory to retry.
`train_teacher --manifest PATH --seed N` also invokes this fixed-task workflow
for one source seed, using its `--checkpoint-interval` for evaluation frequency
and `--validation-episodes` for the assigned-task evaluation count.

Fixed-task CLIs default to standard PPO (`MlpPolicy`): no LSTM, frame stacking,
privileged state, or observation changes. It uses CPU by default and one Torch
thread per worker. `--workers N` runs independent task/seed jobs in separate spawned
processes, each with its own optimizer and history file. `--device` can override
placement. Fixed configurations can make a reactive policy sufficient, but do not
guarantee convergence on every partially observed task; the convergence gate still
applies. Use `--source-algorithm recurrent_ppo` for the previous recurrent learner.

PPO histories live under `train/ppo/`; recurrent histories use
`train/recurrent_ppo/`. Select `memory_fixed_ppo.yaml` for PPO distillation and
`memory_fixed.yaml` for recurrent histories, or override `source_algorithm`.
Run IDs distinguish the algorithms. Use separate source run directories to keep
each pool's aggregate `summary.json` separate. Checkpoints, metadata, per-run
evaluation curves, and the same consecutive-success gate are written for both.

Per-run `evaluations.json`, TensorBoard logs, checkpoints, and `result.json`, plus
the pool run's `summary.json`, expose convergence and learning progress. The
convergence gate evaluates the teacher on its assigned fixed task and requires
the final consecutive evaluations to pass. The dataset rejects incomplete or
unconverged runs. Use separate source seeds on training tasks for validation;
held-out test tasks are reserved for final AD/RAD evaluation.

`memory_fixed.yaml` selects `history_scope: task`. AD, RAD, and compression
pretraining sample across episode boundaries within one source-learning stream.
Termination/truncation tokens precede the next reset observation; terminal
observations remain stored separately and are never treated as observations
requiring another action. Episode-end queries remain represented in sampling.
No window crosses tasks or independent source seeds. The v2 fixed-task format,
manifest membership, and source provenance are checked before training, and
checkpoints record the manifest fingerprint.

Evaluation keeps AD history and RAD compressed memory across episodes of a test
task, clearing it for each new task/trial. `--episodes` means episodes **per task
per trial** in manifest mode. The sweep also evaluates with
`--reset-context-each-episode`; results go to `eval-reset.json`. Each evaluation
reports an adaptation curve by episode index, first-episode success, and overall
return/success. Deterministic policy trials on an identical task repeat the same
trajectory; use `rad_memory.evaluate --sample` for stochastic action trials, and
multiple independently trained AD/RAD seeds for training variability.

For a single trained checkpoint:

```bash
uv run python -m rad_memory.evaluate \
  --checkpoint runs/fixed-sweep/ad-short-h30-seed0/final.pt \
  --manifest tasks/memory_s13_fixed.json --episodes 20 --trials 3 \
  --sample --output runs/fixed-eval.json
```

Remembering a successful route on later episodes is expected in this setting.
Cue-gap statistics describe within-episode observations; they alone do not prove
that a policy uses previous episodes. Use the adaptation curve and reset-context
ablation for that comparison. Fixed-task and legacy checkpoints/artifacts cannot
be silently mixed. Configuration paths in overrides are relative to the working
directory; use absolute paths when launching elsewhere.

## Legacy changing-layout workflow

The commands below retain the original episode-bounded benchmark for existing
experiments. `train_teacher` without `--manifest`, checkpoint `collect`, and specs without a saved
`configuration` use changing layouts. They are not the fixed-task workflow above.

### 1. Profile the task

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

Before launching collection, check that the exact source-learner
hyperparameters converge reliably across training seeds:

```bash
uv run python -m rad_memory.check_recurrent_ppo_convergence \
  --env-id MiniGrid-MemoryS13Random-v0 \
  --random-length \
  --horizon H \
  --seeds 0 1 2 \
  --output-dir runs/recurrent-ppo-convergence-s13
```

The check evaluates a fixed held-out episode set every 50,000 training steps.
It exits successfully only when every seed stays at or above 90% success for
the final three evaluations. `summary.json` records the resolved RecurrentPPO
configuration and every learning-curve point. Use `--save-models` if these
verified final policies should also be retained for collection.

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
