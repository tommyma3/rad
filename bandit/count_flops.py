"""Count average inference FLOPs per action prediction for bandit AD/RAD.

Drives the real inference path (ModelPolicy.action/observe over generate_history
rollouts) with a FlopCounter attached. Random weights: FLOPs depend only on
tensor shapes, not values. Run from the bandit directory with its venv:

    .venv/bin/python count_flops.py --model RAD
"""

import argparse
import json
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from bandit.utils import load_config
from bandit.model import MODEL
from bandit.evaluation import ModelPolicy
from bandit.rollout import generate_history
from bandit.env import BanditTask
from bandit.flops_counter import FlopCounter

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
    parser.add_argument('--config', choices=['ad_short', 'ad_long', 'rad'],
                        default='ad_short')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--delay', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config, 'delayed_adversarial_bandit')

    model = MODEL[config['model']](config)
    model.requires_grad_(False)
    model.eval()

    counter = FlopCounter()
    attach_counter(counter, model)

    means = tuple(np.linspace(0.1, 0.9, config['num_arms']))
    task = BanditTask(task_id='flops-probe', means=means, distribution='uniform', seed=0)

    policy = ModelPolicy(model, seed=args.seed, sample=False)
    n_actions = 0
    start = time.time()
    for episode in range(args.episodes):
        history, _ = generate_history(
            task, policy=policy, delay=args.delay,
            pre_steps=config['pre_steps'], post_steps=config['post_steps'],
            reward_seed=args.seed + episode, learner_seed=1000 + episode,
            distractor_seed=2000 + episode,
            exploration_coefficient=config['exploration_coefficient'])
        # loss_mask marks genuine bandit steps (== model queries); 'bandit_steps'
        # is the cumulative pull index, not a per-step indicator.
        n_actions += int(history['loss_mask'].sum())
    wall_time = time.time() - start

    compress_layers = int(config.get('compress_n_layers', 0)) or None
    result = {
        'env': 'bandit',
        'model': config['model'],
        'config': args.config,
        'device': str(model.device),
        'episodes': args.episodes,
        'delay': args.delay,
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
            'n_transit': config['context_steps'],
            'n_compress_tokens': config.get('n_compress_tokens', 0),
            'compress_n_layers': config.get('compress_n_layers', 0),
            'compress_n_heads': config.get('compress_n_heads', 0),
            'short_memory_keep': config.get('short_memory_keep', 0),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"bandit_{config['model']}_{args.config}.json"
    out_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
