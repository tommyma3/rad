"""Launch paired Darkroom compressor runs on one GPU, with explicit provenance."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from compressor_experiment import PROTOCOL, VARIANTS


def run_name(variant, seed, pretrain=False):
    return f"{'RAD-pretrain' if pretrain else 'RAD'}-darkroom-{variant}-seed{seed}"


def training_commands(args, pilot=False):
    root = args.runs_root / 'pilot' if pilot else args.runs_root
    seeds = args.seeds[:1] if pilot else args.seeds
    commands = []
    for seed in seeds:
        for variant in args.variants:
            common = ['--env', 'darkroom', '--config', f'rad_dr_{variant}',
                      '--seed', str(seed), '--env_split_seed', '0',
                      '--runs_root', str(root), '--traj_dir', str(args.traj_dir)]
            if args.cpu:
                common += ['--cpu']
            if args.num_workers is not None:
                common += ['--num_workers', str(args.num_workers)]
            for pretrain in (True, False):
                script = 'train_pretrain_compression.py' if pretrain else 'train_rad.py'
                command = [args.python, str(PROJECT / script), *common,
                           '--run_name', run_name(variant, seed, pretrain)]
                if not pretrain:
                    checkpoint = root / run_name(variant, seed, True) / 'pretrain-final.pt'
                    command += ['--pretrain_ckpt', str(checkpoint)]
                if pilot:
                    command += ['--steps', str(args.pilot_steps), '--batch_size', str(args.pilot_batch_size),
                                '--no_compile']
                commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot', 'train', 'evaluate', 'all'], default='pilot')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--runs-root', type=Path, default=PROJECT / 'runs/compressor_darkroom')
    parser.add_argument('--traj-dir', type=Path, default=PROJECT / 'datasets')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--pilot-steps', type=int, default=100)
    parser.add_argument('--pilot-batch-size', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--checkpoint', choices=['final', 'best', 'both'], default='final',
                        help='Evaluate the final 100k checkpoint, the test-selected best-model.pt, or both')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('Seeds and variants must be unique')
    if args.pilot_steps < 1 or args.pilot_batch_size < 1:
        parser.error('Pilot budgets must be positive')
    if args.cpu and args.stage not in ('pilot', 'evaluate'):
        parser.error('--cpu is intended for pilots/evaluation, not the full training budget')
    args.runs_root = args.runs_root.resolve()
    args.traj_dir = args.traj_dir.resolve()
    commands = []
    if args.stage in ('pilot', 'all'):
        commands += training_commands(args, pilot=True)
    if args.stage in ('train', 'all'):
        commands += training_commands(args)
    if args.stage in ('evaluate', 'all'):
        checkpoint_kinds = ('final', 'best') if args.checkpoint == 'both' else (args.checkpoint,)
        for kind in checkpoint_kinds:
            commands.append([args.python, str(PROJECT / 'scripts/evaluate_compressor_comparison.py'),
                             '--runs-root', str(args.runs_root), '--variants', *args.variants,
                             '--train-seeds', *map(str, args.seeds), '--device', 'cpu' if args.cpu else 'cuda',
                             '--checkpoint', kind])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    manifest = dict(protocol=PROTOCOL, stage=args.stage, gpu=args.gpu, checkpoint=args.checkpoint,
                    commands=commands, completed=[])
    manifest_name = f'{args.stage}-manifest.json'
    if args.stage == 'evaluate' and args.checkpoint in ('best', 'both'):
        manifest_name = f'evaluate-{args.checkpoint}-manifest.json'
    manifest_path = args.runs_root / manifest_name
    if not args.dry_run:
        args.runs_root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            raise FileExistsError(f'Manifest already exists: {manifest_path}; choose a fresh runs root')
        manifest_path.write_text(json.dumps(manifest, indent=2))
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
