"""Train DPT/IDT without changing the AD/RAD entrypoints or collators.

Run from gridworld: accelerate launch train_baseline.py --config dpt_dr
"""

import argparse
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from baseline_dataset import BASELINE_DATASET, get_baseline_data_loader
from model import MODEL
from model.baseline_common import validate_tokenization
from utils import checkpoint_state_dict, get_config, normalize_compiled_state_dict


def total_loss(output, config):
    loss = output['loss_action']
    if config.get('dynamics', False):
        loss = loss + config.get('dynamics_strength', 1.0) * (output['loss_reward'] + output['loss_next_state'])
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dpt_dr', help='Preset name or YAML file path')
    parser.add_argument('--env', default='darkroom', choices=['darkroom', 'dktd'])
    parser.add_argument('--env_split_seed', type=int)
    parser.add_argument('--collection-env-split-seed', type=int, help='Seed used by collect.py for task ordering; defaults to env_split_seed')
    parser.add_argument('--traj-dir', default='./datasets')
    parser.add_argument('--runs-root', default='./runs')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--train-timesteps', type=int)
    parser.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'])
    args = parser.parse_args()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        config = dict(checkpoint['config'])
        for key, override in (('env_split_seed', args.env_split_seed), ('collection_env_split_seed', args.collection_env_split_seed)):
            if override is not None and override != config.get(key):
                raise ValueError(f'Cannot change {key} when resuming')
    else:
        checkpoint = None
        config = get_config(f'config/env/{args.env}.yaml')
        config.update(get_config(f'config/algorithm/ppo_{args.env}.yaml'))
        model_config = Path(args.config) if Path(args.config).suffix in ('.yaml', '.yml') else Path(f'config/model/{args.config}.yaml')
        config.update(get_config(model_config))
        if args.env_split_seed is not None:
            config['env_split_seed'] = args.env_split_seed
        if args.collection_env_split_seed is not None:
            config['collection_env_split_seed'] = args.collection_env_split_seed
    method = config['model']
    if method not in BASELINE_DATASET:
        raise ValueError('train_baseline.py accepts only DPT/IDT configs')
    validate_tokenization(config, method)
    config['baseline_tokenization'] = f'gridworld-{method.lower()}-gpt2-v1'
    if args.train_timesteps is not None:
        config['train_timesteps'] = args.train_timesteps
    if config['train_timesteps'] < 1:
        raise ValueError('train_timesteps must be positive')
    config['mixed_precision'] = args.mixed_precision or config.get('mixed_precision', 'no')
    accelerator = Accelerator(mixed_precision=config['mixed_precision'],
                              gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1))
    config['device'] = accelerator.device
    config['traj_dir'] = args.traj_dir
    set_seed(config.get('seed', 42))
    run_dir = args.resume.parent if args.resume else Path(args.runs_root) / f"{method}-{config['env']}-seed{config['env_split_seed']}"
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} is nonempty; use --resume or a different --runs-root')
    config['log_dir'] = str(run_dir)
    dataset_type = BASELINE_DATASET[method]
    train = dataset_type(config, args.traj_dir, 'train', config['train_n_stream'], config['train_source_timesteps'])
    test = dataset_type(config, args.traj_dir, 'test', 1, config['train_source_timesteps'])
    train_loader = get_baseline_data_loader(train, config['train_batch_size'], config, True)
    test_loader = get_baseline_data_loader(test, config['test_batch_size'], config, False)
    model = MODEL[method](config)
    optimizer = AdamW(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    # Scheduler is stepped once per synchronized optimizer update on each rank.
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0
    if checkpoint:
        model.load_state_dict(normalize_compiled_state_dict(checkpoint['model']), strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        step = checkpoint['step']
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    writer = None
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / 'config.yaml').open('w') as stream:
            yaml.safe_dump({**config, 'device': str(config['device'])}, stream)
        writer = SummaryWriter(str(run_dir))
    accelerator.wait_for_everyone()

    def save():
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            target = run_dir / f'ckpt-{step}.pt'
            temporary = target.with_suffix('.pt.tmp')
            torch.save({'step': step, 'config': config, 'model': checkpoint_state_dict(accelerator.unwrap_model(model)),
                        'optimizer': optimizer.state_dict(), 'lr_sched': scheduler.state_dict()}, temporary)
            temporary.replace(target)
        accelerator.wait_for_everyone()

    try:
        model.train()
        optimizer.zero_grad()
        while step < config['train_timesteps']:
            for batch in train_loader:
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        output = model(batch)
                        loss = total_loss(output, config)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
                    continue
                scheduler.step()
                step += 1
                if step % config['summary_interval'] == 0 and writer:
                    writer.add_scalar('train/loss', loss.item(), step)
                    for key, value in output.items():
                        writer.add_scalar(f'train/{key}', value.item(), step)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)
                if step % config['eval_interval'] == 0:
                    model.eval()
                    sums = {}
                    count = 0
                    with torch.inference_mode(), accelerator.autocast():
                        for batch in test_loader:
                            metrics = model(batch)
                            size = len(batch['states'])
                            count += size
                            for key, value in metrics.items():
                                sums[key] = sums.get(key, 0) + value.detach() * size
                    count = accelerator.reduce(torch.tensor(count, device=accelerator.device), reduction='sum')
                    for key, value in sums.items():
                        value = accelerator.reduce(value, reduction='sum') / count
                        if writer:
                            writer.add_scalar(f'test/{key}', value.item(), step)
                    model.train()
                if step % config['ckpt_interval'] == 0 or step == config['train_timesteps']:
                    save()
                if step >= config['train_timesteps']:
                    break
    finally:
        if writer:
            writer.close()
        accelerator.end_training()


if __name__ == '__main__':
    main()
