"""
Launch RAD latent-update comparison runs.

Example:
    python scripts/run_rad_latent_update_comparison.py --env dktd --gpus 0 1 2 3 --dry-run
    python scripts/run_rad_latent_update_comparison.py --env darkroom --gpus 0 5 6 7
"""

import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path


VARIANTS = ['replace', 'residual', 'multiplicative_gate', 'gru_gate']

# Per-environment comparison settings. The dktd comparison ran at seed 2;
# darkroom runs at seed 0.
ENV_SETTINGS = {
    'dktd': {
        'config_stem': 'rad_dktd',
        'pretrain_config': 'rad_dktd',
        'default_seed': 2,
    },
    'darkroom': {
        'config_stem': 'rad_dr',
        'pretrain_config': 'rad_dr',
        'default_seed': 0,
    },
}


def variant_config_name(env, variant):
    return f'{ENV_SETTINGS[env]["config_stem"]}_{variant}'


def variant_run_name(env, seed, variant):
    return f'RAD-{env}-seed{seed}-{variant}'


def pretrain_run_name(env, seed):
    return f'RAD-pretrain-{env}-seed{seed}'


def split_command(command):
    return shlex.split(command, posix=os.name != 'nt')


def has_option(command, option):
    return any(token == option or token.startswith(f'{option}=') for token in command)


def accelerate_launch_index(command):
    for index in range(len(command) - 1):
        executable = Path(command[index]).name
        if executable == 'accelerate' and command[index + 1] == 'launch':
            return index
    return None


def allocate_free_port(used_ports):
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('', 0))
            port = sock.getsockname()[1]
        if port not in used_ports:
            used_ports.add(port)
            return port


def add_main_process_port(command, used_ports):
    if has_option(command, '--main_process_port'):
        return command
    launch_index = accelerate_launch_index(command)
    if launch_index is None:
        return command
    port = allocate_free_port(used_ports)
    insert_at = launch_index + 2
    return command[:insert_at] + ['--main_process_port', str(port)] + command[insert_at:]


def format_command(command, env):
    prefix = ''
    if env and env.get('CUDA_VISIBLE_DEVICES') is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
    return prefix + ' '.join(command)


def run_command(command, cwd, dry_run, env=None):
    printable = format_command(command, env)
    print(f'\n$ {printable}', flush=True)
    if dry_run:
        return
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    subprocess.run(command, cwd=cwd, env=process_env, check=True)


def start_command(command, cwd, env):
    process_env = os.environ.copy()
    process_env.update(env)
    printable = format_command(command, env)
    print(f'\n$ {printable}', flush=True)
    return subprocess.Popen(command, cwd=cwd, env=process_env)


def wait_for_processes(processes, stage):
    failed = []
    for variant, gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((variant, gpu, return_code))
    if failed:
        details = ', '.join(f'{variant}@gpu{gpu}: {return_code}' for variant, gpu, return_code in failed)
        raise RuntimeError(f'Parallel {stage} failed: {details}')


def existing_pretrain_checkpoint(runs_root, pretrain_run_name):
    return runs_root / pretrain_run_name / 'pretrain-final.pt'


def run_pretrain(args, project_dir):
    env = {'CUDA_VISIBLE_DEVICES': args.pretrain_gpu}
    command = add_main_process_port(split_command(args.launcher), args.used_ports) + [
        'train_pretrain_compression.py',
        '--config',
        args.pretrain_config,
        '--env',
        args.env,
    ]
    run_command(command, project_dir, args.dry_run, env=env)


def build_train_command(args, variant, pretrain_ckpt):
    return split_command(args.launcher) + [
        'train_rad.py',
        '--config',
        variant_config_name(args.env, variant),
        '--env',
        args.env,
        '--pretrain_ckpt',
        str(pretrain_ckpt),
    ]


def build_eval_command(args, variant, runs_root):
    ckpt_dir = runs_root / variant_run_name(args.env, args.seed, variant)
    command = [args.python, 'evaluate_rad.py', '--ckpt_dir', str(ckpt_dir), '--eval_episodes', str(args.eval_episodes)]
    if args.use_best:
        command.append('--use_best')
    return command


def run_variants(args, project_dir, runs_root, pretrain_ckpt):
    if len(args.gpus) < len(args.variants):
        raise ValueError(
            f'Need at least one GPU per variant: got {len(args.gpus)} GPUs for '
            f'{len(args.variants)} variants'
        )

    if not args.skip_train:
        train_processes = []
        for variant, gpu in zip(args.variants, args.gpus):
            command = add_main_process_port(build_train_command(args, variant, pretrain_ckpt), args.used_ports)
            env = {'CUDA_VISIBLE_DEVICES': gpu}
            if args.dry_run:
                run_command(command, project_dir, args.dry_run, env=env)
            else:
                train_processes.append((variant, gpu, start_command(command, project_dir, env)))
        wait_for_processes(train_processes, 'training')

    if not args.skip_eval:
        eval_processes = []
        for variant, gpu in zip(args.variants, args.gpus):
            command = build_eval_command(args, variant, runs_root)
            env = {'CUDA_VISIBLE_DEVICES': gpu}
            if args.dry_run:
                run_command(command, project_dir, args.dry_run, env=env)
            else:
                eval_processes.append((variant, gpu, start_command(command, project_dir, env)))
        wait_for_processes(eval_processes, 'evaluation')


def parse_args():
    parser = argparse.ArgumentParser(description='Run RAD latent-update comparison experiments.')
    parser.add_argument('--env', default='dktd', choices=['darkroom', 'dktd'], help='Comparison environment.')
    parser.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    parser.add_argument('--seed', type=int, default=None, help='Env split seed; defaults to 2 for dktd, 0 for darkroom.')
    parser.add_argument('--runs_root', default='./runs', help='Run directory root, relative to gridworld_test by default.')
    parser.add_argument('--pretrain_config', default=None, help='Defaults to the env base RAD config (rad_dktd or rad_dr).')
    parser.add_argument('--pretrain_run_name', default=None, help='Defaults to RAD-pretrain-<env>-seed<seed>.')
    parser.add_argument('--gpus', nargs='+', required=True, help='GPU indexes for variants, e.g. --gpus 0 1 2 3.')
    parser.add_argument('--pretrain_gpu', default=None, help='GPU index for shared pretraining. Defaults to the first --gpus entry.')
    parser.add_argument('--launcher', default='uv run', help='Launcher for training scripts.')
    parser.add_argument('--python', default=sys.executable, help='Python executable for evaluation.')
    parser.add_argument('--eval_episodes', type=int, default=100)
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.set_defaults(use_best=True)
    parser.add_argument('--use_best', action='store_true', dest='use_best')
    parser.add_argument('--no_use_best', action='store_false', dest='use_best')
    parser.add_argument('--dry-run', action='store_true', dest='dry_run')
    return parser.parse_args()


def main():
    args = parse_args()
    args.used_ports = set()
    if args.seed is None:
        args.seed = ENV_SETTINGS[args.env]['default_seed']
    if args.pretrain_config is None:
        args.pretrain_config = ENV_SETTINGS[args.env]['pretrain_config']
    if args.pretrain_run_name is None:
        args.pretrain_run_name = pretrain_run_name(args.env, args.seed)
    if args.pretrain_gpu is None:
        args.pretrain_gpu = args.gpus[0]
    project_dir = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = project_dir / runs_root

    pretrain_ckpt = existing_pretrain_checkpoint(runs_root, args.pretrain_run_name)
    if not args.skip_pretrain and not pretrain_ckpt.exists():
        run_pretrain(args, project_dir)
    else:
        reason = 'requested' if args.skip_pretrain else f'checkpoint exists at {pretrain_ckpt}'
        print(f'Skipping pretrain: {reason}')

    if not args.dry_run and not args.skip_train and not pretrain_ckpt.exists():
        raise FileNotFoundError(f'Pretrain checkpoint not found: {pretrain_ckpt}')

    run_variants(args, project_dir, runs_root, pretrain_ckpt)


if __name__ == '__main__':
    main()
