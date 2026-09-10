"""GPT-2 construction and rollout helpers exclusive to DPT/IDT."""

import numpy as np
import torch

from .gpt2 import GPT2Transformer


def validate_tokenization(config, method):
    expected = f'gridworld-{method.lower()}-gpt2-v1'
    if config.get('baseline_tokenization', expected) != expected:
        raise ValueError(f'{method} requires baseline_tokenization={expected}')


def make_transformer(config, max_tokens):
    width = config['tf_n_embd']
    transformer = GPT2Transformer(
        d_model=width, n_heads=config.get('tf_n_head', 4),
        n_layers=config.get('tf_n_layer', 4), max_seq_length=max_tokens,
        dim_feedforward=config.get('tf_dim_feedforward', 4 * width),
        dropout=config.get('tf_dropout', 0.1),
    )
    if config.get('gradient_checkpointing', False):
        transformer.enable_gradient_checkpointing()
    return transformer


def run_transformer(transformer, tokens):
    if not torch.is_grad_enabled():
        transformer = getattr(transformer, '_orig_mod', transformer)
    return transformer(tokens, use_causal_mask=True)


def choose_action(logits, sample):
    if sample:
        return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
    return logits.argmax(-1)


def terminal_states(observations, dones, infos):
    """Keep terminal transitions separate from VecEnv's reset observations."""
    result = np.asarray(observations).copy()
    for i, done in enumerate(dones):
        if done:
            result[i] = infos[i]['terminal_observation']
    return result


def episode_array(episodes):
    # Fixed-horizon Gridworld normally completes these synchronously.
    count = min(map(len, episodes))
    return np.asarray([row[:count] for row in episodes], dtype=np.float64)
