# Delayed adversarial bandit

Standalone uv project for UCB learning histories, Algorithm Distillation (AD),
and Recurrent Algorithm Distillation (RAD). All implementation files live here;
the other environment projects are independent.

## Benchmark contract

One fixed 10-arm task generates 50 genuine pulls, D irrelevant transitions,
then 50 further genuine pulls on the **same task**. A pull's reward is immediate;
the delay is an interruption in useful experience, not delayed reward delivery.

The observation is `BANDIT=0` or `DISTRACTOR=1`. Distractor actions are independently
uniform, and their rewards are zero. The runner supplies these actions for every
method, including during evaluation. They enter the model's history and advance
RAD compression, but are never action-prediction targets. Policies cannot use
their own distractor actions as an external memory channel.

UCB counts, reward sums, tie-breaking RNG, and its genuine-pull clock remain
unchanged throughout the gap. Only a new task resets policy memory. Reward draws
are indexed by genuine pull and arm, using a separate RNG from task generation,
UCB, and distractors. Inserting or extending a gap leaves UCB's 100 genuine
actions and rewards exactly unchanged for fixed seeds.

### Reward distribution

For each new task, draw each arm mean independently:

```text
mu_a ~ Uniform[0, 1]
r_t | a_t=a ~ Normal(mu_a, 0.3^2)
```

The means stay fixed across both phases and across episodes in the convergence
checker. Every genuine pull receives fresh Gaussian noise. Rewards are continuous
and **unclipped**: negative values and values above 1 are valid. The default
`reward_std: 0.3` is in `config/env/adversarial_bandit.yaml` and is recorded in each
task manifest. Distractor rewards remain exactly zero and never update UCB.

The sampler is `iid_uniform_gaussian_v1`. Independently uniform means have no
odd/even bias, so collection and evaluation use the `uniform` task distribution.
The environment filenames retain their existing names for script compatibility.

The artifact schema is now `delayed-bandit-gaussian-sar-v2`. Earlier Bernoulli
collections, manifests, and checkpoints are incompatible: collect new histories
and retrain models for this reward specification. Existing result files are left
as records of the earlier reward setting; use fresh output directories for new runs.

UCB first visits every unpulled arm in random order. Subsequently it maximizes
`reward_sum[a]/count[a] + c*sqrt(log(total_pulls)/count[a])`, with
`c=sqrt(2)` and uniform tie-breaking among exact maximizers.

## Setup

From the repository root:

```powershell
uv sync --project bandit
uv run --project bandit python bandit/collect.py --help
```

The project creates `bandit/.venv` and includes `uv.lock`. Python 3.12 and 3.13
are supported. Torch uses the normal package index; choose a CUDA-enabled Torch
installation appropriate for your server when running GPU experiments. All
scripts work directly or via `python -m bandit.<entrypoint>` from the repo root.
Relative data, config, checkpoint, and output paths are resolved under `bandit/`.

If the machine's global uv cache is inaccessible, set `UV_CACHE_DIR` to a
writable cache directory before invoking uv.

## Collect

```powershell
uv run --project bandit python bandit/collect.py --output datasets/delayed --tasks 10000 --validation_tasks 1000 --seed 0
uv run --project bandit python bandit/collect.py --env adversarial_bandit --output datasets/no_delay --tasks 10000 --validation_tasks 1000 --seed 0
uv run --project bandit python bandit/collect.py --output datasets/mixed --delays 0 25 50 75 100 --pre_steps_choices 40 45 50 55 60 --seed 0
```

The third command varies gap length and insertion point while retaining 100
genuine pulls. Training and validation are independent task manifests drawn from
the same configured training distribution. Evaluation creates a separate manifest.
HDF5 files store complete chronological histories; JSON manifests store exact task
parameters, seeds, phase lengths, sampler version, and configuration. Task metadata
and genuine-pull indices never become model inputs. Fresh collection rejects
nonempty output directories; interrupted `.partial` files are not accepted as data.

## Models and context accounting

Both policies use the same causal observation/action/reward triples and a
categorical action head. `context_steps=K` means **K completed transitions plus
the current query observation**. AD's maximum decision input is `3*K+1` tokens.
RAD additionally has `n_compress_tokens=L` slots, for at most `3*K+1+L` tokens.
For the defaults, those budgets are 151 and 166 tokens respectively.

RAD appends complete transitions. When recent history reaches K+1 transitions,
it compresses the oldest K+1-p and retains p=`short_memory_keep` recent ones.
Therefore K is a maximum capacity, not the number of raw transitions retained
at every action. At a compression event, the compressor sees the previous latent
state (when present) and the outgoing raw transitions; the update mode combines
the old and candidate latent states. All four existing update modes are supported:
`replace`, `residual`, `multiplicative_gate`, and `gru_gate`.

GPT-2 and compression building blocks are copied from Gridworld. The bandit adapter
uses complete-transition boundaries and a separate raw-memory budget. It does not
copy Gridworld's convention of subtracting latent slots from that raw capacity.
Optional null prefix tokens are policy-visible only before recurrent memory exists;
they are not an old recurrent state for first compression.

The sampler chooses genuine target actions uniformly, then batches prefixes of the
same length. AD receives the last K transitions; RAD receives the entire prefix
and processes it recurrently. There is no padding inside the compressor. Every
post-gap target, including the first, is eligible. Current target actions/rewards
are excluded from the prefix, preventing information leakage through compression.

`max_gradient_rounds: null` retains the entire gradient path across the gap.
A finite value is supported but may sever access to the earliest compression;
keep the default for the initial memory experiment.

AD-long has K=300 so its capacity covers the largest initial evaluation history.
Match data, architecture width/depth, seed, and target count across methods.
RAD uses additional parameters and tokens; equal raw capacity is not equal compute.

## Train

```powershell
uv run --project bandit python bandit/train.py --config ad_short --dataset datasets/delayed --run_dir runs/ad_short_s0 --seed 0
uv run --project bandit python bandit/train.py --config ad_long --dataset datasets/delayed --run_dir runs/ad_long_s0 --seed 0
uv run --project bandit python bandit/train_pretrain_compression.py --dataset datasets/delayed --run_dir runs/pretrain_s0 --seed 0
uv run --project bandit python bandit/train_rad.py --dataset datasets/delayed --run_dir runs/rad_s0 --seed 0 --pretrained runs/pretrain_s0/checkpoint-0005000
uv run --project bandit python bandit/train_rad.py --dataset datasets/delayed --run_dir runs/rad_scratch_s0 --seed 0
```

Pretraining reconstructs raw token embeddings from a compressed chunk. Embeddings
and the policy are frozen during this phase; this is not recurrent reconstruction
training. Distillation trains the policy, compressor, and applicable latent modules
with action loss only; the reconstruction decoder is frozen. The pretraining
checkpoint initializes all weights, including the token embeddings it reconstructed.

Use `--steps`, `--batch_size`, `--cpu`, `--threads`, and `--mixed_precision` for
smaller runs. `train_steps` counts optimizer updates; batch size is per process per
accumulation step. Run seeds 0,1,2 (or more) in separate directories for experiments.

On a configured Linux GPU server, for example:

```bash
uv run --project bandit accelerate launch --multi_gpu --num_processes 4 bandit/train_rad.py --run_dir runs/rad_s0 --mixed_precision bf16
```

Train/validation loss and accuracy are recorded in JSONL and TensorBoard. Saved
checkpoints contain model, optimizer, scheduler, RNG, progress, configuration,
process count, and data hashes. `model.pt` is the inference payload. Exact resume
requires unchanged data, settings, and process count:

```powershell
uv run --project bandit python bandit/train.py --run_dir runs/ad_short_s0 --resume runs/ad_short_s0/checkpoint-0001000
```

For interrupted-run testing, `--steps 100 --stop_after 50` preserves the original
100-update schedule, then resume without changing `--steps`. Fresh runs and
existing checkpoints are protected from accidental overwrites.

## Evaluate

```powershell
uv run --project bandit python bandit/scripts/evaluate_delay_sweep.py --checkpoint runs/ad_short_s0/checkpoint-0020000 runs/ad_long_s0/checkpoint-0020000 runs/rad_s0/checkpoint-0020000 --labels AD-short AD-long RAD --output results/delay_sweep --tasks 100 --delays 0 25 50 100 200
```

UCB and random baselines are included by default. Independent uniform-mean tasks
are evaluated using the same manifests, reward potential outcomes, and distractor
streams for all methods and delays. Model actions are sampled by default; use
`--greedy` for argmax evaluation. Use `--device cuda` on GPU.

Each policy generates its own Phase-I evidence in the main `online` protocol.
For an additional controlled retention diagnostic:

```powershell
uv run --project bandit python bandit/evaluate_rad.py --checkpoint runs/rad_s0/checkpoint-0020000 --output results/shared_prefix --manifest results/delay_sweep/manifest.json --shared_prefix
```

This feeds the same UCB Phase-I history into each policy before the gap. It is
recorded as a distinct protocol, not mixed with online results.

Outputs include per-task JSONL, an evaluation manifest and checkpoint provenance,
JSON/CSV summaries, and PNG/PDF delay curves. Metrics include pre/post return,
first 1/5/10 post-gap returns, first optimal action, pseudo-regret, and compression
counts/raw-history lengths at the first post-gap decision. Reward/regret curves
are stored for all 100 genuine pulls. Normalized post-gap scores are optional
derived values; undefined denominators produce null, while raw metrics remain.

Supply multiple checkpoints with the same label to aggregate training runs.
Approximate 95% intervals use run means when several runs are present; for a
single run, they use task variability and are explicitly labeled `ci_unit=task`.
These single-run intervals do not measure training-seed uncertainty.

## Check UCB convergence

```powershell
uv run --project bandit python bandit/test_ucb_convergence.py --pulls 10000 --task-seeds 0 --seeds 0 1 2 --output-dir results/ucb_check
```

The convergence budget is expressed directly as genuine arm pulls. The checker
keeps **one fixed task and the same UCB learner across episode boundaries**,
using fresh reward draws for each configured episode. Delay and distractor steps
do not count toward the pull budget. Collection still starts a fresh learner
for each 100-pull history; a long convergence pass does not establish that 100
pulls suffice for collection.

Every 1,000 arm pulls (and at the end), the checker measures UCB's actual online
performance over the last 1,000 arm pulls. It also evaluates its empirical greedy
arm recommendation on 1,000 independent reward draws without modifying UCB.
The recommendation diagnostic is distinct from exploratory UCB behavior.

By default, each seed must meet both criteria at its **final three checks**:

- Recent mean pseudo-regret `mean(mu_best - mu_selected) <= 0.05`.
- The empirical greedy recommendation's exact expected regret is at most 0.05.

Task means are used only by the evaluator, never by UCB's action selection.
These criteria test finite-budget near-optimality, not a proof of asymptotic
convergence or a requirement to always choose the exact best arm. Fresh-draw
reward estimates and their standard errors are saved as additional diagnostics;
the pass criteria use exact expected gaps to avoid reward-sampling noise.

Configure `--max-mean-regret`, `--max-recommendation-regret`,
`--min-optimal-action-rate`, `--window-pulls`, `--eval-interval`,
`--required-consecutive-evals`, and `--required-seed-fraction` as needed. The seed
fraction must pass separately for every selected task. Use multiple
`--task-seeds`, `--delay 0`, and
`--exploration-coefficient` to vary the experiment.

To check the original **single-history, 100-pull** budget, explicitly use:

```powershell
uv run --project bandit python bandit/test_ucb_convergence.py --pulls 100 --eval-interval 100 --window-pulls 100 --required-consecutive-evals 1 --output-dir results/ucb_100_pulls
```

The script exits **0 for PASS, 1 for FAIL**. Too few checks or an incomplete
rolling window cannot pass. A nonempty output directory is rejected. Outputs
include root/task/seed JSON summaries, per-episode CSV metrics keyed by arm-pull
count, independent
evaluation results, full chronological NPZ histories, final UCB state, and
per-task PNG/PDF learning curves showing every learner seed.

## Validation and experiment progression

```powershell
uv run --project bandit python -m unittest discover -s bandit/tests -v
```

Tests cover frozen UCB state and RNG, no-gap equivalence, distractor independence,
independent uniform means, Gaussian reward moments, task isolation, action masking, causal query inputs, raw-window
exclusion, offline/incremental RAD equivalence, gradients to pre-gap evidence,
phase-specific trainability, exact checkpoint resume, and evaluation outputs.

Start with D=0, then fixed D=100, then mixed delays and insertion offsets. Compare
AD-short, AD-long, RAD, RAD without pretraining, UCB, and random over multiple seeds.
Inspect first-post-gap metrics as well as the full Phase-II return. The expected
RAD advantage is a hypothesis; pipeline tests do not demonstrate convergence or
establish that ordering. CPU smoke tests are separate from full learning experiments
and Linux/CUDA distributed validation.
