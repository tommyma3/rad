"""Count average inference FLOPs per action prediction for gridworld AD/RAD.

Drives the real inference path (model.evaluate_in_context on a darkroom
DummyVecEnv with num_envs=1) with a FlopCounter attached. Random weights: FLOPs
depend only on tensor shapes, not values. Run from the gridworld directory with
its venv:

    .venv/bin/python count_flops.py --config rad_dr
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from utils import get_config
from stable_baselines3.common.vec_env import DummyVecEnv
from flops_counter import FlopCounter

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent / 'flops_results'


def attach_counter(counter, model):
    transformer = getattr(model, 'ad_transformer', None) or model.transformer
    compressor = getattr(model, 'compression_transformer', None)
    gates = [getattr(model, name, None) for name in
             ('latent_gru_gate', 'latent_gru_candidate', 'latent_multiplicative_gate')]
    counter.attach(model, 'other', exclude=[transformer, compressor, *gates])
    counter.attach(transformer, 'transformer')
    counter.attach(compressor, 'compressor')
    for gate in gates:
        counter.attach(gate, 'latent_update')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='darkroom', choices=['darkroom', 'dktd'])
    parser.add_argument('--config', default='rad_dr',
                        choices=['ad_dr', 'rad_dr', 'ad_dktd', 'rad_dktd'])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(f'config/env/{args.env}.yaml')
    config.update(get_config(f'config/algorithm/ppo_{args.env}.yaml'))
    config.update(get_config(f'config/model/{args.config}.yaml'))
    config['device'] = device
    config.setdefault('mixed_precision', 'no')

    model = MODEL[config['model']](config).to(device)
    model.requires_grad_(False)
    model.eval()

    counter = FlopCounter()
    attach_counter(counter, model)

    _, test_env_args = SAMPLE_ENVIRONMENT[args.env](config)
    if args.env == 'darkroom':
        envs = DummyVecEnv([make_env(config, goal=test_env_args[0])])
    else:
        envs = DummyVecEnv([make_env(config, key=test_env_args[0][:2],
                                     goal=test_env_args[0][2:])])

    eval_timesteps = config['horizon'] * args.episodes
    start = time.time()
    with torch.no_grad():
        model.evaluate_in_context(vec_env=envs, eval_timesteps=eval_timesteps, sample=False)
    wall_time = time.time() - start
    envs.close()

    n_actions = eval_timesteps  # num_envs == 1: one action predicted per step
    compress_layers = int(config.get('compress_n_layers', 0)) or None
    result = {
        'env': 'gridworld' if args.env == 'darkroom' else 'gridworld-dktd',
        'model': config['model'],
        'config': args.config,
        'device': str(device),
        'episodes': args.episodes,
        'eval_timesteps': eval_timesteps,
        'total_flops': counter.total,
        'by_tag': dict(counter.by_tag),
        'n_actions': n_actions,
        'flops_per_action': counter.total / n_actions,
        'n_transformer_forwards': n_actions,
        'n_compress_calls': (len(counter.compress_context_lens) // compress_layers
                             if compress_layers else 0),
        'mean_ad_seq_len': (sum(counter.ad_seq_lens) / len(counter.ad_seq_lens)
                            if counter.ad_seq_lens else 0.0),
        'mean_sq_ad_seq_len': (sum(t * t for t in counter.ad_seq_lens) / len(counter.ad_seq_lens)
                               if counter.ad_seq_lens else 0.0),
        'mean_compress_context_len': (sum(counter.compress_context_lens) / len(counter.compress_context_lens)
                                      if counter.compress_context_lens else 0.0),
        'wall_time_s': wall_time,
        'arch': {
            'd_model': config['tf_n_embd'],
            'n_layers': config['tf_n_layer'],
            'n_heads': config['tf_n_head'],
            'ffn': config['tf_dim_feedforward'],
            'n_transit': config['n_transit'],
            'n_compress_tokens': config.get('n_compress_tokens', 0),
            'compress_n_layers': config.get('compress_n_layers', 0),
            'compress_n_heads': config.get('compress_n_heads', 0),
            'short_memory_keep': config.get('short_memory_keep', 0),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"gridworld_{config['model']}_{args.config}.json"
    out_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
