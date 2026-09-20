"""Launch the Darkroom memory-size ablation, one process on the selected GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from memory_size_experiment import PROTOCOL, SIZES, run_name


def training_commands(args, pilot=False):
    root = args.runs_root / 'pilot' if pilot else args.runs_root
    seeds = args.seeds[:1] if pilot else args.seeds
    phases = (True, False) if pilot or args.stage == 'all' else (args.stage == 'pretrain',)
    commands = []
    for seed in seeds:
        for size in args.sizes:
            common = ['--env', 'darkroom', '--config', args.config, '--n_latents', str(size),
                      '--seed', str(seed), '--env_split_seed', '0', '--runs_root', str(root),
                      '--traj_dir', str(args.traj_dir)]
            if args.cpu:
                common += ['--cpu']
            if args.num_workers is not None:
                common += ['--num_workers', str(args.num_workers)]
            for pretrain in phases:
                script = 'train_pretrain_compression.py' if pretrain else 'train_rad.py'
                command = [args.python, str(PROJECT / script), *common,
                           '--run_name', run_name(size, seed, pretrain)]
                if not pretrain:
                    command += ['--pretrain_ckpt', str(root / run_name(size, seed, True) / 'pretrain-final.pt')]
                if pilot:
                    command += ['--steps', str(args.pilot_steps), '--batch_size', str(args.pilot_batch_size),
                                '--no_compile']
                commands.append(command)
    return commands


def evaluation_command(args, pilot=False):
    root = args.runs_root / 'pilot' if pilot else args.runs_root
    command = [args.python, str(PROJECT / 'scripts/evaluate_memory_size_comparison.py'),
               '--runs-root', str(root), '--sizes', *map(str, args.sizes),
               '--train-seeds', *map(str, args.seeds[:1] if pilot else args.seeds),
               '--eval-seeds', *map(str, args.eval_seeds[:1] if pilot else args.eval_seeds),
               '--episodes', str(args.pilot_episodes if pilot else args.episodes),
               '--device', 'cpu' if args.cpu else 'cuda']
    if pilot:
        command += ['--pilot', '--checkpoint-step', str(args.pilot_steps),
                    '--pretrain-steps', str(args.pilot_steps), '--benchmark-repeats', '3']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'pretrain', 'train', 'evaluate', 'all'], default='pilot')
    parser.add_argument('--sizes', nargs='+', type=int, default=list(SIZES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs/memory_size_darkroom')
    parser.add_argument('--traj-dir', type=Path, default=PROJECT / 'datasets')
    parser.add_argument('--config', default='rad_dr_memory_size')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--pilot-steps', type=int, default=100)
    parser.add_argument('--pilot-batch-size', type=int, default=8)
    parser.add_argument('--pilot-episodes', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for values in (args.sizes, args.seeds, args.eval_seeds):
        if len(set(values)) != len(values):
            parser.error('Size and seed lists must not contain duplicates')
    if any(size <= 0 or size % 3 for size in args.sizes):
        parser.error('Sizes must be positive multiples of three')
    if min(args.pilot_steps, args.pilot_batch_size, args.pilot_episodes, args.episodes) < 1:
        parser.error('Budgets must be positive')
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.cpu and args.stage not in ('pilot', 'evaluate'):
        parser.error('--cpu is intended for pilots/evaluation')
    args.runs_root = args.runs_root.resolve()
    args.traj_dir = args.traj_dir.resolve()
    if Path(args.config).suffix in ('.yaml', '.yml'):
        args.config = str(Path(args.config).resolve())
    pilot = args.stage == 'pilot'
    commands = []
    if args.stage != 'evaluate':
        commands += training_commands(args, pilot)
    if args.stage in ('pilot', 'evaluate', 'all'):
        commands.append(evaluation_command(args, pilot))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    root = args.runs_root / 'pilot' if pilot else args.runs_root
    manifest_path = root / f'{args.stage}-manifest.json'
    manifest = dict(protocol=PROTOCOL, stage=args.stage, gpu=args.gpu, sizes=args.sizes,
                    training_seeds=args.seeds[:1] if pilot else args.seeds,
                    commands=commands, completed=[])
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('x') as stream:
            json.dump(manifest, stream, indent=2)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        start = time.monotonic()
        result = subprocess.run(command, cwd=PROJECT, env=environment)
        manifest['completed'].append(dict(command=command, seconds=time.monotonic() - start,
                                          returncode=result.returncode))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        result.check_returncode()


if __name__ == '__main__':
    main()
