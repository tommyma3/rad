# Darkroom tokenization ablation

This experiment adds `AD_DPT` and `RAD_DPT`. Existing `AD`, `RAD`, `DPT`, and
`IDT` classes, model registration, datasets, training entrypoints, inference
methods, configuration files, and checkpoints are unchanged. Continue to train
AD with `train.py` and RAD with `train_rad.py`, using their existing procedures.

## Representation and objective

For each completed transition, a single linear layer embeds
`concat(state_coordinates, one_hot(action), reward, terminal_next_state)`.
The same layer embeds a query by zero-padding the current state coordinates.

* `AD_DPT`: `[e(transition), ..., e(transition), query]`.
* `RAD_DPT`: `[latent_tokens, e(transition), ..., e(transition), query]`.

The action head reads only the final query output. The target is the action
recorded in the PPO history, **not an oracle action**. The query's action,
reward, and next state are excluded from the input. There are no dynamics
losses, planning, or DPT-style repeated optimal-action labels. The reference for
the packed transition fields is
[DICP's DPT](https://github.com/jaehyeon-son/dicp/blob/main/gridworld/model/dpt.py);
this experiment deliberately puts the query at the end.

Both new variants retain the corresponding legacy backbone widths, layers,
learning rates, source streams, training duration, and batch-size settings by
including `ad_dr.yaml` or `rad_dr.yaml`. Legacy dense SAR action losses and RAD
compression buckets are untouched. New examples supervise one endpoint each;
therefore equal batch sizes/updates do not imply equal numbers of action targets
or equal training compute. Record that distinction when interpreting results.

## Explicit context accounting

`policy_token_budget` is the **total number of policy-transformer slots**,
including the query and any latent prefix. `n_transit` is inherited legacy
metadata; it does not override the packed policy budget. History lengths and
`short_memory_keep` count completed environment transitions. Latent slots count
vectors, and need not be divisible by three in the new model.

| Preset | Policy slots | Latent slots | Complete transitions before compression | Complete recent transitions after compression |
| --- | ---: | ---: | ---: | ---: |
| `ad_dpt_dr` | 80 | 0 | 79 (sliding window) | n/a |
| `rad_dpt_dr` | 30 | 15 | up to 29 | 5 immediately after compression, up to 14 before the next |

The untouched SAR presets allocate 240 slots for AD and 90 for RAD. These new
presets explicitly allocate one packed slot instead of three SAR slots for each
nominal context step. RAD retains all 15 latent vectors and five recent
transitions; its latent prefix consequently occupies a larger fraction of the
packed window. **These are not equal-token-compute or exactly equal-compression-
schedule runs.** To study a different resource constraint, create an additional
ablation YAML overriding `policy_token_budget`; do not modify legacy presets.

For budget `B`, latent count `M`, retained transitions `K`, and no initial null
prefix, first compression happens after `B` completed transitions. Thereafter
compression happens every `B - M - K` appended transitions. The query always
reserves one slot. Invalid capacities are rejected. Set
`always_use_latent_prefix: true` to reserve the null prefix before compression
as well; this also changes the first boundary.

## Recurrent memory and variable-length batches

The persistent state is `(latent_tokens, recent_transition_embeddings,
compression_count)`. The query is rebuilt for each decision and is never added
to memory. Compression reads the previous latent state plus the oldest complete
transition embeddings, retaining the most recent `short_memory_keep` transitions.
The first compression bypasses latent updating; later updates use the same RAD
gate math. Latent-prefix policy positions cannot read recent tokens or the query.
Historical policy tokens are causal; the final query sees all available memory.

Training and inference call the same `append_transitions` implementation. Offline
training detaches all but the latest `max_gradient_rounds` compression rounds.
The training curriculum restricts which histories are sampled; evaluation uses
unlimited recurrent compression rather than truncating history at a training cap.

`EndpointBatchSampler` samples multiple historical lengths per logical batch.
For RAD it uses the inherited curriculum's category probabilities (short 0–50,
medium 51–150, long 151–400, very long 401+), discards impossible categories,
and renormalizes their weights. Within a category it samples a compression depth
and then a length, covering different positions in the refill cycle. The new
preset starts at length zero to include query-only and short-history decisions.
Lengths never exceed the available recorded history: a 1,000-step source admits
at most 999 past transitions plus a recorded query target.

The collator groups sampled examples by their **exact** history length and
returns unpadded microbatches. Their mean losses are weighted by example count
and accumulated into one optimizer update. This is not the old compression-bucket
sampler: recent-window lengths vary within and across logical batches. Padding
is never passed to the compressor. `lengths_per_batch` defaults to four.

Sampling is a deterministic function of training seed, update, and process rank.
DDP ranks share length choices but sample different source windows. The sampler
encodes lengths in worker indices, so prefetching does not use stale curriculum
state. Checkpoints save model/optimizer/scheduler/scaler state and per-rank RNG
state. Exact resume requires the same world size and precision.

## Source histories and evaluation splits

`TransitionDataset` reuses the legacy `ADDataset` reader's selected groups,
streams, observations, actions, and rewards. It does not relabel actions or
change which goals the baseline sees. Completed next states are recovered from
Darkroom's deterministic transition function because the collector's terminal
`next_states` can contain reset observations. Online inference likewise stores
the terminal next state and uses the reset observation as the next query. Memory
persists across episodes on one task and resets for each evaluation run.

The legacy reader selects shuffled HDF5 group IDs directly. Those IDs need not
equal canonical goal IDs, so its training goals may overlap the evaluation
goals. This ablation preserves that selection for a like-for-like data comparison.
The new evaluator records actual training/evaluation goal IDs and their overlap
using the collection seed (Darkroom default 0). It prints any overlap instead of
labeling those results strictly held out. A corrected-split experiment would
require separately approved baseline data changes and retraining.

## Commands

Run from `gridworld`, using its installed environment. These commands train new
models; they do not launch or alter the legacy baselines.

```powershell
accelerate launch train_tokenization.py --config ad_dpt_dr --seed 42
accelerate launch train_tokenization.py --config rad_dpt_dr --phase pretrain --seed 42
accelerate launch train_tokenization.py --config rad_dpt_dr --seed 42 --pretrain-ckpt runs/tokenization/RAD_DPT-pretrain-darkroom-seed42/ckpt-40000.pt
```

Without `--pretrain-ckpt`, RAD_DPT trains from scratch. No automatic checkpoint
discovery occurs. If the comparison's legacy RAD uses compression pretraining,
use the new representation's separate pretraining run too. It reconstructs
packed transition embeddings using MSE against detached embedding targets,
training the packed embedding, compressor, and reconstruction decoder. The
inherited pretrain window is 40 transitions and the budget is 40,000 updates.
Fine-tuning imports only those three modules. SAR/DPT policy checkpoints and
incompatible tokenization markers or shapes are rejected.

Repeat with explicit training seeds, e.g. 42, 43, 44. `--seed` controls model/data
sampling; it does **not** change `env_split_seed` or the source task split. Keep
legacy training seeds paired using the existing baseline configuration workflow.
New run directories include the training seed and live under `runs/tokenization`.

```powershell
accelerate launch train_tokenization.py --resume runs/tokenization/RAD_DPT-darkroom-seed42/ckpt-50000.pt
```

A fresh run rejects a nonempty destination. For a short smoke run, use a separate
`--runs-root`, a small YAML with `torch_compile: false`, and `--updates 2`.
CPU validation can use `--mixed-precision no` with `accelerate launch --cpu`.

Supply checkpoints explicitly for the four-way comparison:

```powershell
python scripts/evaluate_tokenization.py --checkpoint AD=runs/AD-darkroom-seed0 --checkpoint RAD=runs/RAD-darkroom-seed0/best-model.pt --checkpoint AD_DPT=runs/tokenization/AD_DPT-darkroom-seed42 --checkpoint RAD_DPT=runs/tokenization/RAD_DPT-darkroom-seed42 --episodes 100 --eval-seeds 0 1 2 3 4
```

Repeat `--checkpoint METHOD=PATH` for independent training seeds. Directories
select the largest **numeric** `ckpt-N.pt`; an explicitly supplied file is used
as-is. Use a consistent checkpoint-selection rule across methods. The evaluator
calls legacy AD/RAD inference unchanged, and loads new models only through their
isolated registry. `--allow-partial` supports smoke tests with fewer methods.

Outputs are `comparison.png`, per-checkpoint raw return arrays (`.npz`), and
`metrics.json`, under `runs/tokenization/comparison`. Metrics include episode
return, cumulative reward, final-ten-episode return, elapsed time per vector
step (including environment/compression work), CUDA peak allocated memory,
compression count, resource settings, and the split audit. Bands are approximate
95% intervals over independent training-seed mean curves; repeated rollout seeds
are averaged within each training seed. A single training seed has no band.

## Validation

```powershell
python -m unittest discover -s ../tests -p test_gridworld_tokenization.py -v
python -m unittest discover -s ../tests -p test_gridworld_baselines.py -v
```

Tests cover query placement, causal masks, source-action labels, replay/online
memory parity, gradient-round truncation, unpadded microbatch weighting,
variable-length/curriculum sampling, terminal/reset handling, pretraining,
checkpoint compatibility, exact training resume, and evaluator artifacts.
Legacy model/train/dataset/config sources remain unchanged. Full convergence and
multi-GPU execution require experiment runs on the intended training hardware.
