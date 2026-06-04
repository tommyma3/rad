import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn.functional as F
import math

from env import map_dark_states
from functools import partial
import matplotlib.pyplot as plt


class CurriculumAwareLRScheduler(LambdaLR):
    """
    Learning rate scheduler that provides warmup at each curriculum stage transition.
    
    This ensures the model has adequate learning rate to adapt when harder tasks
    (more compressions) are introduced in the curriculum.
    
    Schedule per stage:
    - Mini-warmup: LR ramps up over `stage_warmup_steps`
    - Cosine decay: LR decays to `min_lr_ratio * base_lr` over the stage duration
    
    When a new stage starts, LR jumps back up and warms up again.
    """
    
    def __init__(
        self,
        optimizer,
        curriculum_steps: list,  # List of step numbers where curriculum changes
        total_steps: int,
        initial_warmup_steps: int = 1000,
        stage_warmup_steps: int = 500,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: The optimizer to schedule
            curriculum_steps: List of steps where curriculum stage changes (e.g., [0, 25000, 60000, 100000])
            total_steps: Total training steps
            initial_warmup_steps: Warmup steps at the very beginning
            stage_warmup_steps: Warmup steps at each curriculum stage transition
            min_lr_ratio: Minimum LR as ratio of base LR (e.g., 0.1 means min LR is 10% of base)
            last_epoch: The index of last epoch (for resuming)
        """
        self.curriculum_steps = sorted(curriculum_steps)
        self.total_steps = total_steps
        self.initial_warmup_steps = initial_warmup_steps
        self.stage_warmup_steps = stage_warmup_steps
        self.min_lr_ratio = min_lr_ratio
        
        # Precompute stage boundaries
        self.stage_boundaries = self._compute_stage_boundaries()
        
        super().__init__(optimizer, self._lr_lambda, last_epoch=last_epoch)
    
    def _compute_stage_boundaries(self):
        """Compute start/end steps for each curriculum stage."""
        boundaries = []
        for i, start in enumerate(self.curriculum_steps):
            if i + 1 < len(self.curriculum_steps):
                end = self.curriculum_steps[i + 1]
            else:
                end = self.total_steps
            boundaries.append((start, end))
        return boundaries
    
    def _get_stage_info(self, step):
        """Get current stage index and progress within stage."""
        for i, (start, end) in enumerate(self.stage_boundaries):
            if start <= step < end:
                return i, start, end
        # Default to last stage
        return len(self.stage_boundaries) - 1, self.stage_boundaries[-1][0], self.stage_boundaries[-1][1]
    
    def _lr_lambda(self, step):
        """Compute LR multiplier for given step."""
        stage_idx, stage_start, stage_end = self._get_stage_info(step)
        stage_duration = stage_end - stage_start
        steps_into_stage = step - stage_start
        
        # Determine warmup steps for this stage
        if stage_idx == 0:
            warmup_steps = self.initial_warmup_steps
        else:
            warmup_steps = self.stage_warmup_steps
        
        # Phase 1: Warmup
        if steps_into_stage < warmup_steps:
            # Linear warmup from min_lr_ratio to 1.0
            progress = steps_into_stage / warmup_steps
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * progress
        
        # Phase 2: Cosine decay
        decay_steps = stage_duration - warmup_steps
        if decay_steps <= 0:
            return 1.0
        
        decay_progress = (steps_into_stage - warmup_steps) / decay_steps
        decay_progress = min(decay_progress, 1.0)  # Clamp to [0, 1]
        
        # Cosine decay from 1.0 to min_lr_ratio
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay


def get_curriculum_aware_scheduler(
    optimizer,
    curriculum,
    total_steps,
    initial_warmup_steps=1000,
    stage_warmup_steps=500,
    min_lr_ratio=0.1,
):
    """
    Create a curriculum-aware LR scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        curriculum: List of (step, max_compressions, length_dist) tuples
        total_steps: Total training steps
        initial_warmup_steps: Warmup steps at the very beginning
        stage_warmup_steps: Warmup steps at each curriculum stage transition
        min_lr_ratio: Minimum LR as ratio of base LR
        
    Returns:
        CurriculumAwareLRScheduler instance
    """
    curriculum_steps = [item[0] for item in curriculum]
    
    return CurriculumAwareLRScheduler(
        optimizer=optimizer,
        curriculum_steps=curriculum_steps,
        total_steps=total_steps,
        initial_warmup_steps=initial_warmup_steps,
        stage_warmup_steps=stage_warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def get_traj_file_name(config):
    if config["env"] == 'metaworld':
        task = config['task']
    else:
        task = config['env']

    path = f'history_{task}_{config["alg"]}_alg-seed{config["alg_seed"]}'

    return path

def ad_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long), num_classes=5)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    if 'target_next_states' in batch[0].keys():
        res['target_next_states'] = map_dark_states(torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False), grid_size=grid_size)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
        
    return res

def get_data_loader(dataset, batch_size, config, shuffle=True):
    collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=config['num_workers'], persistent_workers=True)

def log_in_context(values: np.ndarray, max_reward: int, episode_length: int, tag: str, title: str, xlabel: str, ylabel: str, step: int, success=None, writer=None) -> None:
    steps = np.arange(1, len(values[0])+1) * episode_length
    mean_value = values.mean(axis=0)
    
    plt.plot(steps, mean_value)
    
    if success is not None:
        success_rate = success.astype(np.float32).mean(axis=0)

        for i, (xi, yi) in enumerate(zip(steps, mean_value)):
            if (i+1) % 10 == 0:
                plt.annotate(f'{success_rate[i]:.2f}', (xi, yi))
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-max_reward * 0.05, max_reward * 1.05)
    writer.add_figure(f'{tag}/mean', plt.gcf(), global_step=step)
    plt.close()

def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch