"""Isolated query-last, packed-transition AD/RAD tokenization ablations.

The legacy AD, RAD and DPT implementations are deliberately not registered or
modified here. A history contains completed transitions only; queries are never
persisted or compressed. Capacities in this module count transformer tokens.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline_common import choose_action, episode_array, make_transformer, terminal_states
from .compressed_ad import RAD
from .compression import CompressionTransformer, ReconstructionDecoder


TOKENIZATION = 'gridworld-packed-transition-query-last-v1'


def validate_config(config):
    if config.get('tokenization') != TOKENIZATION:
        raise ValueError(f'Tokenization ablations require tokenization={TOKENIZATION}')
    if config.get('env') not in ('darkroom', 'dktd') or config.get('dynamics', False):
        raise ValueError('This ablation supports action-only Darkroom and DKTD training')
    if int(config['policy_token_budget']) < 2:
        raise ValueError('policy_token_budget must reserve a query and history')


@dataclass(frozen=True)
class MemorySchedule:
    policy_tokens: int
    latent_tokens: int
    keep: int
    null_prefix: bool = False

    def __post_init__(self):
        if self.latent_tokens < 1 or not 0 <= self.keep <= self.capacity(True):
            raise ValueError('Require positive latent slots and 0 <= keep <= post-compression capacity')
        if self.capacity(True) < 1:
            raise ValueError('Policy capacity must include latent slots, a transition, and a query')

    def capacity(self, has_latent):
        return self.policy_tokens - 1 - (self.latent_tokens if has_latent or self.null_prefix else 0)

    def state_after(self, length):
        """Return (compression count, recent transitions) for a complete history."""
        if length < 0:
            raise ValueError('History length cannot be negative')
        first = self.capacity(False) + 1
        if length < first:
            return 0, length
        period = self.capacity(True) - self.keep + 1
        rounds, offset = divmod(length - first, period)
        return 1 + rounds, self.keep + offset


class ADTransition(nn.Module):
    """AD with [packed history, current-state query] and recorded-action labels."""

    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        self.max_seq_length = int(config['policy_token_budget'])
        width = config['tf_n_embd']
        self.embed_context = nn.Linear(2 * config['dim_states'] + config['num_actions'] + 1, width)
        self.ad_transformer = make_transformer(config, self.max_seq_length)
        self.pred_action = nn.Linear(width, config['num_actions'])
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.))

    @property
    def device(self):
        return self.embed_context.weight.device

    def transition_tokens(self, states, actions, rewards, next_states):
        if actions.ndim == states.ndim - 1:
            actions = F.one_hot(actions.long(), self.config['num_actions'])
        fields = torch.cat((states.float(), actions.float(), rewards.float().unsqueeze(-1), next_states.float()), -1)
        return self.embed_context(fields)

    def query_token(self, states):
        return self.embed_context(F.pad(states.float(), (0, self.embed_context.in_features - states.shape[-1]))).unsqueeze(1)

    def empty_memory(self, batch_size):
        return None, self.embed_context.weight.new_empty(batch_size, 0, self.config['tf_n_embd']), 0

    def append_transitions(self, memory, tokens, gradient_start=0):
        _, recent, _ = memory
        return None, torch.cat((recent, tokens), 1)[:, -(self.max_seq_length - 1):], 0

    def replay(self, tokens):
        return self.append_transitions(self.empty_memory(tokens.shape[0]), tokens)

    def policy_logits(self, query_states, memory):
        _, recent, _ = memory
        tokens = torch.cat((recent, self.query_token(query_states)), 1)
        transformer = self.ad_transformer if torch.is_grad_enabled() else getattr(self.ad_transformer, '_orig_mod', self.ad_transformer)
        return self.pred_action(transformer(tokens, use_causal_mask=True)[:, -1])

    def forward(self, batch, pretrain=False):
        if pretrain:
            raise ValueError('Compression pretraining requires RAD_DPT')
        batch = {key: value.to(self.device) for key, value in batch.items()}
        if 'context_lengths' in batch and not torch.all(batch['context_lengths'] == batch['states'].shape[1]):
            raise ValueError('Use exact-length microbatches; padded histories cannot enter memory')
        tokens = self.transition_tokens(*(batch[key] for key in ('states', 'actions', 'rewards', 'next_states')))
        memory = self.replay(tokens)
        logits = self.policy_logits(batch['query_states'], memory)
        return {
            'loss_action': self.loss_fn(logits, batch['target_actions'].long()),
            'acc_action': (logits.argmax(-1) == batch['target_actions']).float().mean(),
            'num_compressions': logits.new_tensor(memory[2]),
            'recent_length': logits.new_tensor(memory[1].shape[1]),
            'history_length': logits.new_tensor(tokens.shape[1]),
        }

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        if beam_k:
            raise ValueError('Tokenization ablations use action prediction without beam search')
        observations = vec_env.reset()
        memory = self.empty_memory(vec_env.num_envs)
        totals = np.zeros(vec_env.num_envs)
        episodes = [[] for _ in totals]
        events = []
        for step in range(eval_timesteps):
            states = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
            actions = choose_action(self.policy_logits(states, memory), sample)
            observations, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            next_states = torch.as_tensor(terminal_states(observations, dones, infos), device=self.device)
            reward_tensor = torch.as_tensor(rewards, device=self.device)
            token = self.transition_tokens(states[:, None], actions[:, None], reward_tensor[:, None], next_states[:, None])
            old_count = memory[2]
            memory = self.append_transitions(memory, token)
            if memory[2] != old_count:
                events.append(step)
            totals += rewards
            for index, done in enumerate(dones):
                if done:
                    episodes[index].append(totals[index])
                    totals[index] = 0
        return {'reward_episode': episode_array(episodes), 'compression_events': events,
                'total_compressions': memory[2]}


class RADTransition(ADTransition):
    """Recurrent packed-transition memory with a transient query at the end."""

    # Reuse the existing architecture's math without changing its implementation.
    _update_latent_tokens = RAD._update_latent_tokens
    _compress_sequence = RAD._compress_sequence
    _module_for_current_grad_mode = RAD._module_for_current_grad_mode
    _get_attention_mask_for_latent = RAD._get_attention_mask_for_latent
    _forward_ad_transformer = RAD._forward_ad_transformer

    def __init__(self, config):
        super().__init__(config)
        self.n_compress_tokens = int(config['n_compress_tokens'])
        self.always_use_latent_prefix = config.get('always_use_latent_prefix', False)
        self.schedule = MemorySchedule(self.max_seq_length, self.n_compress_tokens,
                                       int(config['short_memory_keep']), self.always_use_latent_prefix)
        self.max_gradient_rounds = int(config.get('max_gradient_rounds', 2))
        if self.max_gradient_rounds < 0:
            raise ValueError('max_gradient_rounds must be nonnegative')
        self.max_compressions = config.get('max_compressions')
        self.latent_update_mode = config.get('latent_update_mode', 'replace')
        if self.latent_update_mode not in RAD.LATENT_UPDATE_MODES:
            raise ValueError(f'Unknown latent_update_mode: {self.latent_update_mode}')
        width = config['tf_n_embd']
        self.latent_type_embedding = nn.Parameter(torch.empty(1, 1, width))
        self.null_latent_tokens = nn.Parameter(torch.empty(1, self.n_compress_tokens, width))
        self.latent_residual_norm = nn.LayerNorm(width)
        self.latent_multiplicative_gate = nn.Linear(width, width)
        self.latent_gru_gate = nn.Linear(2 * width, width)
        self.latent_gru_candidate = nn.Linear(2 * width, width)
        options = dict(d_model=width, n_heads=config['compress_n_heads'], n_layers=config['compress_n_layers'],
                       dim_feedforward=config.get('tf_dim_feedforward', 4 * width))
        # The same allocation also supports separate compression pretraining.
        max_context = max(int(config.get('max_context_length', self.max_seq_length)),
                          int(config.get('pretrain', {}).get('n_transit', self.max_seq_length)))
        self.compression_transformer = CompressionTransformer(
            **options, n_compress_tokens=self.n_compress_tokens,
            max_context_length=max_context + self.n_compress_tokens + 1)
        self.reconstruction_decoder = ReconstructionDecoder(**options, max_seq_length=max_context)
        nn.init.trunc_normal_(self.latent_type_embedding, std=.02)
        nn.init.trunc_normal_(self.null_latent_tokens, std=.02)
        nn.init.zeros_(self.latent_multiplicative_gate.weight)
        nn.init.constant_(self.latent_multiplicative_gate.bias, config.get('latent_gate_init_bias', 4.))
        nn.init.zeros_(self.latent_gru_gate.weight)
        nn.init.constant_(self.latent_gru_gate.bias, config.get('latent_gru_init_bias', -2.))
        nn.init.zeros_(self.latent_gru_candidate.weight)
        nn.init.zeros_(self.latent_gru_candidate.bias)
        self.configure_phase(False)

    def configure_phase(self, pretrain):
        """Exclude inactive modules from optimizers and distributed reduction."""
        for name, parameter in self.named_parameters():
            if pretrain:
                active = name.startswith(('embed_context.', 'compression_transformer.', 'reconstruction_decoder.'))
            else:
                active = not name.startswith('reconstruction_decoder.')
                if name.startswith('null_latent_tokens'):
                    active = self.always_use_latent_prefix
                for prefix, mode in (('latent_residual_norm.', 'residual'),
                                     ('latent_multiplicative_gate.', 'multiplicative_gate'),
                                     ('latent_gru_', 'gru_gate')):
                    if name.startswith(prefix):
                        active = self.latent_update_mode == mode
            parameter.requires_grad_(active)

    def append_transitions(self, memory, tokens, gradient_start=0):
        latent, recent, count = memory
        cursor = 0
        while cursor < tokens.shape[1]:
            capacity = self.schedule.capacity(latent is not None)
            take = min(tokens.shape[1] - cursor, capacity - recent.shape[1] + 1)
            recent = torch.cat((recent, tokens[:, cursor:cursor + take]), 1)
            cursor += take
            if recent.shape[1] > capacity:
                prefix_length = recent.shape[1] - self.schedule.keep
                prefix = recent[:, :prefix_length]
                compress_input = torch.cat((latent, prefix), 1) if latent is not None else prefix
                latent = self._compress_sequence(compress_input, count >= gradient_start, latent)
                recent = recent[:, prefix_length:]
                count += 1
        return latent, recent, count

    def replay(self, tokens):
        rounds, _ = self.schedule.state_after(tokens.shape[1])
        if self.training and self.max_compressions is not None and rounds > self.max_compressions:
            raise ValueError('Sampled history exceeds the active compression curriculum')
        return self.append_transitions(self.empty_memory(tokens.shape[0]), tokens,
                                       max(0, rounds - self.max_gradient_rounds))

    def policy_logits(self, query_states, memory):
        latent, recent, _ = memory
        tokens = torch.cat((recent, self.query_token(query_states)), 1)
        if latent is None and self.always_use_latent_prefix:
            latent = self.null_latent_tokens.expand(tokens.shape[0], -1, -1)
        if latent is not None:
            tokens = torch.cat((latent, tokens), 1)
        return self.pred_action(self._forward_ad_transformer(tokens, latent is not None)[:, -1])

    def forward(self, batch, pretrain=False):
        if not pretrain:
            return super().forward(batch)
        fields = [batch[key].to(self.device) for key in ('states', 'actions', 'rewards', 'next_states')]
        tokens = self.transition_tokens(*fields)
        if not tokens.shape[1]:
            raise ValueError('Compression pretraining needs a nonempty transition window')
        latent = self.compression_transformer(tokens)
        reconstructed = self.reconstruction_decoder(latent, tokens.shape[1])
        return {'loss_recon': F.mse_loss(reconstructed, tokens.detach())}


TOKENIZED_MODELS = {'AD_DPT': ADTransition, 'RAD_DPT': RADTransition}
