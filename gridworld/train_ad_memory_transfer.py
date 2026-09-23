"""Standalone, single-process AD -> compressor pretraining -> RAD fine-tuning."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from ad_memory_transfer import (PROTOCOL, PROJECT, RAD, AD, EvaluationRAD, audit_dataset,
    atomic_save, capture_rng, configure_stage, digest, evaluate_policy, initialize_ad,
    load_source, read_checkpoint, resolve_config, restore_rng, validate_config, write_json)
from dataset import CompressionPretrainDataset, RADDataset
from optimizer_utils import build_rad_optimizer_param_groups
from train_rad import rad_collate_fn


def load_settings(filename):
    """Experiment includes are relative to the containing file, independent of cwd."""
    filename = Path(filename).resolve()
    settings = yaml.safe_load(filename.read_text(encoding='utf-8'))
    parent = settings.pop('include', None)
    return dict(load_settings(filename.parent / parent), **settings) if parent else settings


def make_optimizer(model, cfg, stage):
    groups = ([dict(params=[p for p in model.parameters() if p.requires_grad], lr=cfg['pretrain_lr'])]
              if stage == 'pretrain' else build_rad_optimizer_param_groups(model, cfg))
    optimizer = torch.optim.AdamW(groups, betas=(cfg['beta1'], cfg['beta2']), weight_decay=cfg['weight_decay'])
    budget = cfg['pretrain_timesteps'] if stage == 'pretrain' else cfg['train_timesteps']
    warmup = max(1, int(budget * cfg['warmup_fraction']))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(1, budget - warmup))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def apply_curriculum(model, dataset, cfg, step):
    stage = cfg['curriculum_schedule'][0]
    for candidate in cfg['curriculum_schedule']:
        if step >= candidate['step']:
            stage = candidate
    model.set_curriculum(stage['max_compressions'])
    dataset.update_max_compressions(stage['max_compressions'])
    dataset.update_length_distribution(stage['length_distribution'])


def sample_batch(dataset, cfg, stage, rng):
    size = cfg['pretrain_batch_size'] if stage == 'pretrain' else cfg['train_batch_size']
    if stage == 'pretrain':
        indices = [rng.randrange(len(dataset)) for _ in range(size)]
    else:
        bucket = dataset.sample_compression_bucket()
        indices = [(bucket, rng.randrange(len(dataset))) for _ in range(size)]
    return rad_collate_fn([dataset[i] for i in indices], cfg['grid_size'], cfg['num_actions'])


def stage_directory(run_dir, stage, resume=None):
    run_dir = Path(run_dir).resolve()
    marker = run_dir / 'transfer-experiment.json'
    if run_dir.exists():
        import json
        if not marker.exists() or json.loads(marker.read_text()) != {'protocol': PROTOCOL}:
            raise ValueError(f'Refusing an existing non-transfer directory: {run_dir}')
    elif resume:
        raise ValueError('Resume requires its original experiment directory')
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(marker, dict(protocol=PROTOCOL))
    destination = run_dir / stage
    if resume:
        if Path(resume).resolve().parent != destination or not destination.is_dir():
            raise ValueError('Resume checkpoint must belong to this experiment stage directory')
        resumed_step = read_checkpoint(resume)['step']
        saved_steps = [int(p.stem.split('-')[-1]) for p in destination.glob('ckpt-*.pt')]
        if saved_steps and resumed_step < max(saved_steps):
            raise ValueError('A newer checkpoint exists; resume the latest update to avoid overwriting progress')
    else:
        destination.mkdir(exist_ok=False)
    return destination


def validate_stage_checkpoint(checkpoint, stage, complete=False):
    if checkpoint.get('protocol') != PROTOCOL or checkpoint.get('stage') != stage:
        raise ValueError(f'Expected a {PROTOCOL} {stage} checkpoint')
    cfg = checkpoint['config']
    validate_config(cfg)
    budget = cfg['pretrain_timesteps'] if stage == 'pretrain' else cfg['train_timesteps']
    if not 0 <= checkpoint['step'] <= budget or (complete and checkpoint['step'] != budget):
        raise ValueError(f'{stage} checkpoint has not completed its configured budget')


def train_stage(cfg, stage, run_dir, device, source=None, pretrain=None, resume=None, stop_after=None):
    """All state is stage-local, including optimizer, scaler, sampler RNG and best selection."""
    validate_config(cfg)
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    data_rng = random.Random(cfg['seed'] + 10007)
    device = torch.device(device)
    cfg = deepcopy(cfg)
    cfg['device'] = str(device)
    if device.type == 'cpu' and cfg['mixed_precision'] != 'no':
        raise ValueError('Use mixed_precision=no for CPU runs')
    if cfg['mixed_precision'] not in ('no', 'fp16', 'bf16'):
        raise ValueError('mixed_precision must be no, fp16, or bf16')
    current_audit = audit_dataset(cfg)
    if 'dataset_audit' in cfg and cfg['dataset_audit'] != current_audit:
        raise ValueError('Dataset identity or task membership changed since previous stage')
    cfg['dataset_audit'] = current_audit
    model = RAD(cfg).to(device)
    if resume:
        checkpoint = read_checkpoint(resume)
        validate_stage_checkpoint(checkpoint, stage)
        expected = dict(checkpoint['config'], device=str(device))
        if expected != cfg:
            raise ValueError('Resume config differs from checkpoint config')
        model.load_state_dict(checkpoint['model'], strict=True)
    elif stage == 'pretrain':
        if source is None:
            raise ValueError('Fresh pretraining requires an explicit AD checkpoint')
        initialize_ad(model, source)
    else:
        if pretrain is None:
            raise ValueError('Fresh fine-tuning requires an explicit completed pretraining checkpoint')
        checkpoint = read_checkpoint(pretrain)
        validate_stage_checkpoint(checkpoint, 'pretrain', complete=True)
        expected = dict(checkpoint['config'], device=str(device),
                        pretrain_source=dict(path=str(Path(pretrain).resolve()), sha256=digest(pretrain)))
        if expected != cfg:
            raise ValueError('Fine-tuning config does not match the pretrained system')
        model.load_state_dict(checkpoint['model'], strict=True)
    configure_stage(model, stage)
    optimizer, scheduler = make_optimizer(model, cfg, stage)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['mixed_precision'] == 'fp16')
    dataset_type = CompressionPretrainDataset if stage == 'pretrain' else RADDataset
    dataset = dataset_type(cfg, cfg['traj_dir'], 'train', cfg['train_n_stream'], cfg['train_source_timesteps'])
    dataset.rng = data_rng
    if len(dataset) <= 0:
        raise ValueError('Training dataset is empty')
    if stage == 'finetune' and max(dataset._available_compression_buckets()) < 2:
        raise ValueError('Fine-tuning data must support repeated compression')
    step, best_reward, best_step = 0, -float('inf'), 0
    if resume:
        for key in ('optimizer', 'lr_sched', 'scaler', 'rng', 'best_eval_reward', 'best_step'):
            if key not in checkpoint:
                raise ValueError(f'Resume checkpoint is missing {key}')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, best_reward, best_step = checkpoint['step'], checkpoint['best_eval_reward'], checkpoint['best_step']
        restore_rng(checkpoint['rng'], data_rng)
    destination = stage_directory(run_dir, stage, resume)
    write_json(destination / 'config.json', cfg)
    budget = cfg['pretrain_timesteps'] if stage == 'pretrain' else cfg['train_timesteps']
    end = budget if stop_after is None else min(budget, step + stop_after)

    def save(filename, eval_reward=None):
        payload = dict(protocol=PROTOCOL, stage=stage, config=cfg, step=step,
                       model=model.state_dict(), optimizer=optimizer.state_dict(), lr_sched=scheduler.state_dict(),
                       scaler=scaler.state_dict(), rng=capture_rng(data_rng),
                       best_eval_reward=best_reward, best_step=best_step)
        if eval_reward is not None:
            payload.update(eval_reward=eval_reward, selection='mean reward on audited held-out tasks',
                           selection_seed=0)
        atomic_save(destination / filename, payload)

    def record(record):
        import json
        with (destination / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(record, flush=True)

    # This is a diagnostic baseline; best-model selection starts after a training update.
    if stage == 'finetune' and not resume:
        rewards, compressions = evaluate_policy(model, cfg)
        np.save(destination / 'initial-rewards.npy', rewards)
        record(dict(step=0, eval_reward=float(rewards.mean()), compressions=compressions))
    failures = 0
    while step < end:
        if stage == 'finetune':
            apply_curriculum(model, dataset, cfg, step)
        configure_stage(model, stage)
        retry_state = capture_rng(data_rng)
        batch = sample_batch(dataset, cfg, stage, data_rng)
        optimizer.zero_grad(set_to_none=True)
        dtype = torch.float16 if cfg['mixed_precision'] == 'fp16' else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg['mixed_precision'] != 'no'):
            output = model(batch, pretrain=stage == 'pretrain')
            loss = output['loss_recon' if stage == 'pretrain' else 'loss_action']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at {stage} update {step + 1}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm) and not scaler.is_enabled():
            raise FloatingPointError(f'Nonfinite gradient at {stage} update {step + 1}')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            restore_rng(retry_state, data_rng)
            failures += 1
            if failures >= 20:
                raise FloatingPointError('20 consecutive AMP overflows; no successful update')
            continue
        failures = 0
        step += 1
        scheduler.step()
        if step == 1 or step % cfg['log_interval'] == 0 or step == end:
            record(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                        compressions=output.get('num_compressions', 0)))
        if stage == 'finetune' and (step % cfg['eval_interval'] == 0 or step == budget):
            rewards, compressions = evaluate_policy(model, cfg)
            reward = float(rewards.mean())
            if not math.isfinite(reward):
                raise FloatingPointError('Nonfinite evaluation reward')
            np.save(destination / f'eval-{step}.npy', rewards)
            record(dict(step=step, eval_reward=reward, compressions=compressions))
            if reward > best_reward:
                best_reward, best_step = reward, step
                save('best-model.pt', reward)
        if step % cfg['ckpt_interval'] == 0 or step == end:
            save(f'ckpt-{step}.pt')
    if stage == 'pretrain' and step == budget:
        save('pretrain-final.pt')
        return destination / 'pretrain-final.pt'
    return destination / f'ckpt-{step}.pt'


def evaluate_experiment(run_dir, ad_path, pretrain_path, output_dir, device, seeds, episodes=None,
                        selection='best', rad_reference=None):
    """Paired AD / before-transfer / after-transfer curves and memory interventions."""
    run_dir = Path(run_dir)
    pretrain = read_checkpoint(pretrain_path)
    validate_stage_checkpoint(pretrain, 'pretrain', complete=True)
    cfg = deepcopy(pretrain['config'])
    final_path = run_dir / 'finetune' / f"ckpt-{cfg['train_timesteps']}.pt"
    final = read_checkpoint(final_path)
    validate_stage_checkpoint(final, 'finetune', complete=True)
    selected_path = run_dir / 'finetune' / 'best-model.pt' if selection == 'best' else final_path
    selected = read_checkpoint(selected_path)
    validate_stage_checkpoint(selected, 'finetune')
    expected = dict(cfg, device=final['config']['device'],
                    pretrain_source=dict(path=str(Path(pretrain_path).resolve()), sha256=digest(pretrain_path)))
    if final['config'] != expected or selected['config'] != final['config']:
        raise ValueError('Selected/pretrained/final checkpoint provenance differs')
    if selection == 'best' and ('eval_reward' not in selected or selected['step'] < 1):
        raise ValueError('Best checkpoint is missing selection provenance')
    if digest(ad_path) != cfg['source_ad']['sha256']:
        raise ValueError('AD checkpoint does not match the experiment source')
    cfg['device'] = str(device)
    if episodes is not None:
        cfg['eval_episodes'] = episodes
    source = load_source(ad_path)
    source_cfg = dict(source['config'], device=str(device))
    ad = AD(source_cfg).to(device)
    ad.load_state_dict(source['model'], strict=True)
    models = [('AD', ad, 'intact')]
    for label, checkpoint in [('RAD_before_finetune', pretrain), ('RAD_finetuned', selected)]:
        model = EvaluationRAD(cfg).to(device)
        model.load_state_dict(checkpoint['model'], strict=True)
        models.append((label, model, 'intact'))
        if label == 'RAD_finetuned':
            models.extend([(label + '_zero', model, 'zero'), (label + '_shuffle', model, 'shuffle')])
    reference_info = None
    if rad_reference:
        reference = read_checkpoint(rad_reference)
        rcfg = dict(reference['config'], device=str(device))
        for key in ('env', 'grid_size', 'num_actions', 'horizon'):
            if rcfg[key] != cfg[key]:
                raise ValueError(f'Reference RAD differs in {key}')
        # Record overlap rather than presenting unmatched legacy results as held-out.
        from baseline_dataset import collection_task_ids
        from compressor_experiment import select_dataset_groups
        if rcfg.get('collection_env_split_seed', cfg['collection_env_split_seed']) != cfg['collection_env_split_seed']:
            raise ValueError('Reference RAD collection seed differs')
        rcfg['collection_env_split_seed'] = cfg['collection_env_split_seed']
        order = collection_task_ids(cfg['grid_size'], 2 if cfg['env'] == 'darkroom' else 4,
                                    cfg['collection_env_split_seed'])
        trained = {order[g] for g in select_dataset_groups(rcfg, 'train')}
        overlap = trained.intersection(cfg['dataset_audit']['eval_task_ids'])
        if overlap:
            raise ValueError('Reference RAD trained on evaluation tasks; use a split-matched reference')
        model = RAD(rcfg).to(device)
        model.load_state_dict(reference['model'], strict=True)
        models.append(('RAD_reference', model, 'intact'))
        reference_info = dict(path=str(Path(rad_reference).resolve()), sha256=digest(rad_reference),
                              step=reference['step'], n_transit=rcfg['n_transit'], config=rcfg)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, arrays = [], {}
    for label, model, intervention in models:
        curves, counts = [], []
        for seed in seeds:
            rewards, count = evaluate_policy(model, cfg, seed, intervention)
            curves.append(rewards)
            counts.append(count)
        arrays[label] = np.stack(curves)
        rows.append(dict(method=label, mean_reward=float(arrays[label].mean()),
                         late_20_reward=float(arrays[label][..., -20:].mean()),
                         total_compressions=counts, context_timesteps=model.n_transit))
    np.savez_compressed(output_dir / 'curves.npz', **arrays)
    write_json(output_dir / 'summary.json', dict(protocol=PROTOCOL, results=rows,
        eval_seeds=seeds, episodes=cfg['eval_episodes'], dataset_audit=cfg['dataset_audit'],
        source_ad=cfg['source_ad'], selected_checkpoint=str(selected_path.resolve()),
        checkpoint_selection=selection, selected_step=selected['step'],
        selection_eval_reward=selected.get('eval_reward'), reference=reference_info,
        note='Best is selected on the audited held-out tasks; evaluation seeds are not training-seed replicates.'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['pretrain', 'finetune', 'all', 'evaluate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--ad-checkpoint', type=Path)
    parser.add_argument('--pretrain-checkpoint', type=Path)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Exact checkpoint from this stage/run; no AD reinitialization')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=YAML_VALUE')
    parser.add_argument('--stop-after', type=int, help='Stop after this many additional successful updates')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--selection', choices=['best', 'final'], default='best')
    parser.add_argument('--rad-reference', type=Path)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process per run; launch separate seeds on separate GPUs')
    if args.threads < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('threads and stop-after must be positive')
    if args.episodes is not None and args.episodes < 1:
        parser.error('episodes must be positive')
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('Evaluation seeds must be distinct')
    torch.set_num_threads(args.threads)
    if args.stage == 'evaluate':
        if not args.ad_checkpoint or not args.pretrain_checkpoint or not args.output_dir:
            parser.error('evaluate requires ad-checkpoint, pretrain-checkpoint, and a fresh output-dir')
        print(evaluate_experiment(args.run_dir, args.ad_checkpoint, args.pretrain_checkpoint,
              args.output_dir, args.device, args.eval_seeds, args.episodes, args.selection, args.rad_reference))
        return
    if args.resume:
        if args.stage == 'all' or args.set or args.config or args.ad_checkpoint or args.pretrain_checkpoint or args.dataset_dir:
            parser.error('Resume uses checkpoint config only; specify one stage, run-dir, resume, and device')
        checkpoint = read_checkpoint(args.resume)
        validate_stage_checkpoint(checkpoint, args.stage)
        train_stage(checkpoint['config'], args.stage, args.run_dir, args.device,
                    resume=args.resume, stop_after=args.stop_after)
        return
    if args.stage == 'finetune':
        if not args.pretrain_checkpoint or args.set or args.config or args.ad_checkpoint or args.dataset_dir:
            parser.error('finetune requires only pretrain-checkpoint, run-dir, and device; settings were fixed at pretraining')
        checkpoint = read_checkpoint(args.pretrain_checkpoint)
        validate_stage_checkpoint(checkpoint, 'pretrain', complete=True)
        cfg = checkpoint['config']
    else:
        if not args.ad_checkpoint or not args.dataset_dir or not args.config:
            parser.error('pretrain/all require config, ad-checkpoint, and dataset-dir')
        if args.stage == 'all' and args.stop_after:
            parser.error('Use a single stage with stop-after; all requires completed pretraining')
        settings = load_settings(args.config)
        for override in args.set:
            key, separator, value = override.partition('=')
            if not separator or key not in settings:
                parser.error(f'Unknown or malformed setting: {override}')
            settings[key] = yaml.safe_load(value)
        source = load_source(args.ad_checkpoint)
        cfg = resolve_config(settings, source, args.ad_checkpoint, args.dataset_dir, args.seed)
        args.pretrain_checkpoint = train_stage(cfg, 'pretrain', args.run_dir, args.device,
                                              source=source, stop_after=args.stop_after)
        if args.stage == 'pretrain':
            return
        cfg = read_checkpoint(args.pretrain_checkpoint)['config']
    cfg = dict(cfg, pretrain_source=dict(path=str(args.pretrain_checkpoint.resolve()),
                                        sha256=digest(args.pretrain_checkpoint)))
    train_stage(cfg, 'finetune', args.run_dir, args.device, pretrain=args.pretrain_checkpoint,
                stop_after=args.stop_after)


if __name__ == '__main__':
    main()
