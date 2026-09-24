# Bandit training convergence

Train the standard AD-short, AD-long, and RAD models for **100,000 optimizer
updates each**, evaluating online cumulative expected regret at a configurable
interval. The protocol uses **5 arms and 100 genuine pulls**, with 50 pulls before
and 50 after the distractor gap. RAD starts from scratch by default; pass
`--pretrained` to initialize it from compression pretraining. Existing model
defaults and other experiments are unchanged.

From the repository root on a CUDA server:

```bash
uv run --project bandit python bandit/scripts/run_training_convergence.py \
  --gpus 0 1 2 --seeds 0 1 2 \
  --steps 100000 --eval-interval 1000 \
  --eval-tasks 100 --delays 50 \
  --root runs/training_convergence_v1
```

The launcher queues nine independent runs (three methods times three training
seeds), with one run per GPU. Each worker evaluates on its own GPU between training
intervals. GPU count does not change batch size or the number of training updates.
Indices are relative to inherited `CUDA_VISIBLE_DEVICES`; GPU UUIDs also work.
Use fewer or more GPU IDs as available. This is a queue of independent runs,
not distributed training of one model. Worker failures stop the queue and terminate
the launcher's other workers. Logs identify the failed run.

Without `--dataset`, the launcher collects 10,000 training and 1,000 validation
UCB histories in the experiment directory, using the current training gap (50).
All methods share those histories. To reuse compatible data, add
`--dataset datasets/delayed`. Existing data must contain 5 arms, 50+50 pulls,
and fixed gap insertion after pull 50. Data hashes and the resolved configurations
are frozen in `plan.json`. Paths are relative to `bandit/` unless absolute.

AD-short and RAD retain the existing 50-transition raw context capacity;
AD-long retains 300 transitions. The launcher rejects histories that exceed
AD-long's full-history capacity. The standard backbone, optimizers, target
sampling and training schedule are reused. All three train for the requested
update budget; checkpoints are evaluated at their current training step, without
best-checkpoint selection. Equal updates/batch sizes do not mean equal compute.

## Pretrained RAD initialization

Pass a compression-pretraining checkpoint directory or its `model.pt` file:

```bash
uv run --project bandit python bandit/scripts/run_training_convergence.py \
  --gpus 0 1 2 --seeds 0 1 2 --steps 100000 --eval-interval 1000 \
  --pretrained 'runs/pretrain_s{seed}/checkpoint-0020000' \
  --root runs/training_convergence_pretrained_v1
```

`{seed}` selects the corresponding checkpoint for each training seed. A path
without `{seed}` shares one initialization across all RAD runs. Paths resolve
under `bandit/`. The checkpoints must already exist; the launcher does not run
compression pretraining. Checkpoint schema, phase and RAD architecture are
validated before launching or collecting data.

This uses the existing trainer's `--pretrained` behavior: it loads the full RAD
model state, including the compressor, policy and token embeddings from that
pretraining checkpoint. AD-short and AD-long remain initialized from scratch.
All three then receive the full requested distillation budget (100,000 by
default); compression-pretraining updates are additional and excluded from the
plot's training-update axis. Step 0 evaluates RAD after loading the checkpoint.

The frozen plan records each source's absolute path, SHA-256, pretraining step,
seed and dataset hashes. RAD's saved training configuration also records its
source. Source files must remain available and unchanged for resume. Resume
continues the saved model and optimizer without reapplying pretrained weights;
it rejects a different `--pretrained` initialization. Omit `--pretrained` on
resume to use the saved plan. The generated caption identifies pretrained RAD.

## Evaluation and figures

Evaluation runs at update 0, every `--eval-interval` updates, and at the final
update, even if the interval does not divide the training budget. Each point is
the mean, over held-out tasks, of

```text
R_100 = sum(t=1..100) [max_a mu_a - mu_(a_t)].
```

This uses the exact expected arm gaps under the actions selected during each
online rollout. It does not subtract noisy realized rewards. Distractor
transitions enter model history but contribute neither pulls nor regret.
Each task starts with empty policy memory. Rewards remain Gaussian with the
collection's noise level. Evaluation actions are sampled by default; `--greedy`
switches every method to argmax. Task means are never exposed to model inputs.

One frozen `evaluation_manifest.json` specifies held-out tasks, reward potential
outcomes, distractor streams and policy sampling seeds, shared across all methods,
training seeds and training steps. The runner checks task disjointness from both
training and validation. Evaluation does not advance training's Torch RNG state.

Add `--delays 0 50 100 200` for a shared-axis panel per gap length. By default,
the figure contains one compact panel, distinct colors and line styles, a shared
bottom legend, and no smoothing. Curves show mean regret across training seeds;
shading shows one sample standard deviation across the **training-seed task
means**. With one seed, no uncertainty band is drawn. These bands do not quantify
held-out task sampling uncertainty. Convergence here means the learning curve
over training updates, not wall-clock speed or an automatic statistical test.

Outputs under the experiment root:

- `ad_short_s*/`, `ad_long_s*/`, `rad_s*/`: resumable model/optimizer/RNG
  checkpoints at evaluation intervals, validation losses, and TensorBoard logs.
- `evaluations/<method>_s*/step-*.json`: each task's full 100-pull cumulative
  regret curve and its final total, for every training evaluation point.
- `plots/training_convergence.pdf`: vector figure with embedded fonts.
- `plots/training_convergence.png`: 300-dpi preview.
- `plots/summary.csv`, `plots/summary.json`, `plots/caption.txt`: figure values
  and a caption explaining the protocol and uncertainty.
- `logs/`: worker output and GPU job history.

The plotting step requires the complete method/seed/step/task grid. It fails
on missing points instead of silently comparing different runs or horizons.
All periodic checkpoints are retained; choose the interval with disk space and
evaluation cost in mind. Evaluation scales with tasks, delays, and checkpoint
count, and currently processes tasks sequentially within each worker.

## Inspect, resume, or regenerate

```bash
# Print the resolved configurations and commands without launching or writing.
uv run --project bandit python bandit/scripts/run_training_convergence.py --dry-run

# Resume the frozen plan; only GPU assignment and thread count may change.
uv run --project bandit python bandit/scripts/run_training_convergence.py \
  --root runs/training_convergence_v1 --gpus 0 1 2 --resume

# Regenerate figures from the complete recorded evaluations.
uv run --project bandit python bandit/scripts/run_training_convergence.py \
  --root runs/training_convergence_v1 --plot-only

# Bounded execution check, not a convergence experiment.
uv run --project bandit python bandit/scripts/run_training_convergence.py \
  --root runs/convergence_smoke --cpu --seeds 0 --steps 2 \
  --eval-interval 1 --eval-tasks 2 --train-tasks 4 \
  --validation-tasks 2 --batch-size 2 --threads 1
```

Resume reads all scientific settings from the saved plan; new CLI training or
evaluation settings are ignored, while an explicitly supplied `--pretrained`
must match the saved initialization. Changed data or manifests are rejected.
Completed runs are skipped, and incomplete runs resume their most recent complete
checkpoint. Evaluation files beyond that checkpoint are deterministically
recomputed. An interruption before the first checkpoint requires a fresh root.
Fresh launches reject nonempty output directories. Use a distinct root for a new
protocol or budget. `--mixed-precision bf16` is optional on supporting GPUs;
the default uses full precision.

Local CPU tests establish metric accounting, training RNG preservation,
checkpoint resume and the launcher/evaluator/plot pipeline. They do not establish
100k-update convergence, GPU memory fit, or successful multi-GPU execution.

Implementation validation: the three new convergence tests passed, as did a
two-seed CPU smoke run for all three methods and a visual check of its rendered
PDF. The full bandit suite had 37 passes and 8 existing failures (six model tests
generate 10-arm actions against a 5-arm config; two UCB tests assume delay 100
against the configured delay 50). The same eight failures reproduced with the
original trainer. These unrelated fixtures and experiment defaults were left intact.
