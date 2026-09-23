# AD checkpoint to RAD memory transfer

This standalone experiment asks whether a trained AD policy can learn to read
compressed memory. It adds files only: existing AD/RAD models, datasets, configs,
training entrypoints, evaluators, and checkpoints are unchanged. The source AD
checkpoint and HDF5 histories are opened read-only. Nothing searches existing run
directories for an initialization checkpoint.

## Stages

1. **Import AD.** Strictly load a Gridworld SAR AD checkpoint, including its
   Transformer, state/action/reward and type embeddings, and action head. Derive
   the policy width, layers, heads, FFN width, and dropout from that checkpoint.
   For a shorter RAD context, copy the corresponding prefix of the learned
   position table. Longer contexts and incompatible architectures fail explicitly.
2. **Pretrain compression.** Freeze every imported AD parameter. Train a new AE
   compression Transformer (including its query tokens) and reconstruction decoder
   to reconstruct the frozen AD token embeddings with MSE. A window has
   `n_transit` transitions; memory has shape `[batch, n_compress_tokens, tf_n_embd]`.
   Both stages construct exactly the same model shapes; there is no positional
   interpolation or decoder resizing at the stage boundary.
3. **Jointly fine-tune.** Load the complete stage-1 system and create a fresh
   optimizer/scheduler. Train the policy, token embeddings, action head,
   compressor/query tokens, latent-type embedding, and selected recurrent update
   modules on ordinary RAD action loss. The reconstruction decoder, unused update
   modules, and unused null memory tokens stay frozen. There is no reconstruction
   regularizer during policy training. No latent prefix appears before the first
   compression. Existing RAD recurrence and truncated compression gradients are
   reused without changing their behavior.

| Setting | Darkroom | DKTD |
| --- | --- | --- |
| RAD context (transition-equivalent token budget) | 30 | 72 |
| Memory tokens | 15 | 60 |
| Retained recent transitions | 5 | 10 |
| Compressor layers / heads | 3 / 4 | 4 / 4 |
| Compression pretraining updates | 40,000 | 50,000 |
| Joint updates | 20,000 | 20,000 |
| Pretraining / fine-tuning batch size | 512 / 256 | 1,024 / 1,024 |
| Pretraining learning rate | 3e-4 | 3e-4 |
| AD / compressor / latent fine-tuning rates | 3e-5 / 1e-4 / 1e-4 | same |

The fine-tuning curriculum reaches unlimited compressions at 75% of the budget.
Changing `train_timesteps` rescales the stage boundaries. A short pilot must still
leave distinct stages, or use a custom `transfer_curriculum`. The scheduler uses
5% initial warmup and cosine decay to 10% of the base rate. These are initial
experimental settings, not measured convergence claims.

## Running

Run these commands from the repository root in the existing Gridworld Python
environment. Replace the checkpoint and dataset paths with actual inputs; the
example paths are placeholders. No AD training is launched automatically.

```powershell
python gridworld/train_ad_memory_transfer.py --stage all --config gridworld/config/model/rad_ad_transfer_dr.yaml --ad-checkpoint /path/to/ad-checkpoint.pt --dataset-dir /path/to/datasets --run-dir gridworld/runs/ad-memory-transfer/darkroom-seed0 --seed 0 --device cuda
```

For DKTD, use `rad_ad_transfer_dktd.yaml` and a matching DKTD AD checkpoint.
Use a fresh run directory for each training seed. This runner deliberately uses
one process per run; different seeds can run independently on different GPUs.
Default precision is FP32. Optional `--set mixed_precision=bf16` or `fp16` enables
autocasting on CUDA. Successful optimizer updates count toward the budget; FP16
overflows retry the same batch and RNG state at a reduced loss scale.

Run the stages separately:

```powershell
python gridworld/train_ad_memory_transfer.py --stage pretrain --config gridworld/config/model/rad_ad_transfer_dr.yaml --ad-checkpoint /path/to/ad-checkpoint.pt --dataset-dir /path/to/datasets --run-dir gridworld/runs/ad-memory-transfer/darkroom-seed0 --device cuda
python gridworld/train_ad_memory_transfer.py --stage finetune --pretrain-checkpoint gridworld/runs/ad-memory-transfer/darkroom-seed0/pretrain/pretrain-final.pt --run-dir gridworld/runs/ad-memory-transfer/darkroom-seed0 --device cuda
```

Set budgets and other experiment YAML settings during pretraining, for example
`--set train_timesteps=10000 --set train_batch_size=64`. Fine-tuning inherits the
saved configuration, including its seed and budget; it cannot silently switch
to a different memory shape, dataset, or AD source. The single-process sampler
draws one compression bucket per batch and reads curriculum changes immediately.
It uses a separate saved RNG rather than asynchronous DataLoader workers, making
CPU continuation reproducible without worker-prefetch state. This may have
different throughput from the legacy trainers.

For a bounded pilot, use `--stop-after 10` with an individual stage. Resume from
the exact saved stage checkpoint:

```powershell
python gridworld/train_ad_memory_transfer.py --stage pretrain --run-dir gridworld/runs/ad-memory-transfer/darkroom-seed0 --resume gridworld/runs/ad-memory-transfer/darkroom-seed0/pretrain/ckpt-10.pt --device cuda
```

Resume restores model, optimizer, scheduler, AMP scaler, RNG states, update count,
and best selection. It does not reread or reapply the AD weights. It checks the
current dataset against the recorded audit. CPU/GPU changes may affect numerical
results even when state restoration succeeds. Fresh stages reject existing stage
directories; non-experiment directories are always rejected.
Resuming an older checkpoint is rejected when a newer periodic checkpoint exists,
to prevent overwriting later progress.

## Task identity and evaluation

The source checkpoint's task split and `dataset_task_mapping` are inherited.
Legacy HDF5 group IDs represent a shuffled collection order, so the audit maps
the *actual selected groups* back to task identities. Evaluation uses the complement
of both AD and transfer training tasks. For legacy checkpoints these goals can
differ from the old evaluator's nominal test set. This is recorded explicitly.
The original collection seed defaults to 0 for Darkroom and 2 for DKTD; override
it only to match the actual collector. A source checkpoint that records a different
collection seed is rejected unless the experiment setting matches it.

The audit checks presence and available lengths/streams of every training group
and validates reward/task identity on its first stream's first two episodes.
It records the HDF5 path, size, modification time, group/task IDs, and sampling
scope. Source AD training membership is inferred from its saved config; the
original AD dataset bytes and any unrecorded checkpoint-selection history cannot
be verified. This audit is not a full-history content hash or a guarantee that
legacy checkpoint selection never used those tasks.

During fine-tuning, `metrics.jsonl` records training loss, compression counts and
reward versus update. Evaluation starts at step zero, then runs every 1,000
updates and at completion. Best selection considers trained checkpoints only.
`best-model.pt` stores the highest mean reward on the audited held-out tasks using
evaluation seed 0; `ckpt-<budget>.pt` proves completion. This follows ordinary RAD
reward-based best selection, so these are **held-out-task-selected results**, not
an independent final test set. Periodic checkpoints are retained for review.

```powershell
python gridworld/train_ad_memory_transfer.py --stage evaluate --run-dir gridworld/runs/ad-memory-transfer/darkroom-seed0 --ad-checkpoint /path/to/ad-checkpoint.pt --pretrain-checkpoint gridworld/runs/ad-memory-transfer/darkroom-seed0/pretrain/pretrain-final.pt --output-dir gridworld/runs/ad-memory-transfer/darkroom-seed0/evaluation --device cuda --eval-seeds 0 1 2 --episodes 100
```

The evaluator requires the completed final checkpoint and defaults to the exact
`best-model.pt`; it never falls back to final weights when best is missing. Use
`--selection final` for an explicit final-budget comparison. It produces:

- `curves.npz`: `[evaluation_seed, task, episode]` arrays for original AD, RAD
  before joint fine-tuning, fine-tuned RAD, and zero/shuffled-memory interventions.
- `summary.json`: mean/late returns, task audit, contexts, checkpoint steps,
  source hashes/configs, and selection provenance.

Interventions change only memory contents at the policy input; prefix layout,
positions, recent history, and recurrent memory updates remain intact. Slot
shuffling uses its own RNG. All methods use paired tasks and action seeds;
evaluation restores training RNG state. Original AD retains its native context,
which is reported rather than treated as compute-matched.

Optionally pass `--rad-reference /path/to/regular-rad.pt` to include ordinary RAD.
The evaluator rejects reference training-task overlap and records its config,
training step and context. Budget/context differences remain explicit. Use at
least three independent training seeds for training-seed uncertainty; multiple
evaluation seeds of one trained model do not provide that uncertainty.

## Validation

```powershell
python -m unittest discover -s tests -p test_gridworld_ad_memory_transfer.py -v
```

Tests cover AD-logit equivalence before compression, positional prefix transfer,
compiled keys and rejection paths, frozen AD pretraining, joint gradients/updates,
actual task split auditing, both environments' two-stage synthetic smoke runs,
exact CPU resume, checkpoint selection, memory interventions, and input immutability.
Synthetic histories and tiny models validate plumbing only. Target-GPU memory fit,
real-data learning, and benefit over AD/regular RAD require actual experiments.
