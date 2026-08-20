# AD and RAD on fixed Memory Maze tasks

This project ports the causal state/action/reward Algorithm Distillation (AD)
and Recurrent Algorithm Distillation (RAD) design from `gridworld_test` and
`metaworld` to 64x64 image observations from Memory Maze.

## Task contract

One task is one exact maze problem:

- fixed maze topology;
- fixed positions and color identities of all objects;
- fixed agent starting position and orientation;
- the original Memory Maze episode horizon and target-switching behavior.

Every reset reconstructs the same geometry, object placement, and starting
pose. Reaching the requested object yields +1 and selects another requested
object without ending the episode. Post-reset target progression is reseeded
independently, so repeated attempts share the task while retaining the
original stochastic sequence of target requests.

A fresh source learner trains on one task from random initialization until its
moving return converges. The complete chronological behavior history is saved.
AD and RAD are trained on histories from many training tasks and evaluated by
interacting repeatedly with disjoint held-out tasks. Transformer/RAD context is
retained across episodes of the same task and cleared between tasks.

## Source algorithms

- `ppo`: Stable-Baselines3 PPO with a feed-forward `CnnPolicy`; it intentionally
  has no recurrence, frame stacking, or external memory.
- `dreamer_tbtt`: an in-repository PyTorch DreamerV2 implementation with the
  Memory Maze paper's 2048-dimensional recurrent RSSM state, 32x32 discrete
  stochastic state, sequence length 48, imagination horizon 15, eight parallel
  actors, and sequential TBTT replay. Each replay slot carries a detached RSSM
  state to its next chunk and resets it only at episode boundaries.

PPO and Dreamer histories, AD/RAD configs, and checkpoints remain separate.

## Artifact contracts

Task manifests are JSON files whose stable ID hashes the maze layout, ordered
object positions, start position, and start direction. Source histories use
`memory-maze-trajectories-v1` HDF5 files. Images are stored once per episode as
`T+1` uint8 frames, while actions, rewards, boundaries, and learner steps use
length `T`. Dataset readers are lazy and may cross episode boundaries only
inside one task history.

Model checkpoints use `memory-maze-sar-v1`; incompatible checkpoints are
rejected rather than partially loaded.

## Server workflow

Run commands from this directory.

```bash
uv sync

# Create fixed train/test tasks. Repeat for maze sizes 11, 13, and 15.
uv run python generate_tasks.py --maze-size 9 --n-train 1000 --n-test 100 --output tasks

# Train fresh source learners on every task.
uv run python scripts/collect_all_tasks.py --config config/algorithm/ppo.yaml --tasks tasks/9x9/train
uv run python scripts/collect_all_tasks.py --config config/algorithm/dreamer_tbtt.yaml --tasks tasks/9x9/train

# AD, compression pretraining, and RAD fine-tuning remain source-specific.
uv run accelerate launch train.py --config config/model/ad_ppo.yaml
uv run accelerate launch train_pretrain_compression.py --config config/model/rad_pretrain_ppo.yaml
uv run accelerate launch train_rad.py --config config/model/rad_ppo.yaml --pretrained-compression runs/RAD-pretrain-memory-maze-ppo/ckpt-000050000.pt

# Optional latent-update comparison; the same runner supports dreamer_tbtt.
uv run python scripts/run_rad_latent_update_comparison.py --source ppo \
  --pretrained-compression runs/RAD-pretrain-memory-maze-ppo/ckpt-000050000.pt

# Evaluate while retaining context across repeated attempts of each unseen task.
uv run python evaluate.py --checkpoint runs/RAD-memory-maze-ppo/ckpt-000100000.pt \
  --task tasks/9x9/test/task-000000.json --episodes 100 --output eval/rad-ppo.json
```

Generate and collect both train and test task histories because held-out action
loss evaluation uses test histories. Interactive evaluation itself uses only
the task manifests and model checkpoint.

## Required server validation

The local implementation handoff intentionally does not execute these checks:

```bash
uv run python -m unittest \
  ../tests/test_memory_maze_task_contract.py \
  ../tests/test_memory_maze_dataset_contract.py \
  ../tests/test_memory_maze_models.py \
  ../tests/test_memory_maze_dreamer_tbtt.py -v

MUJOCO_GL=egl uv run python generate_tasks.py \
  --maze-size 9 --n-train 2 --n-test 1 --output /tmp/memory-maze-tasks
```

Follow those with short PPO, Dreamer update, AD forward/training, compression
pretraining, RAD fine-tuning, and unseen-task evaluation smoke runs before
starting long source-learning or distillation jobs. Paper-scale results require
100M environment steps and multiple seeds and are not implied by smoke tests.
