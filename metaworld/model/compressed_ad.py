"""
Recurrent reward-aware SAD-style Algorithm Distillation.

RAD uses the same s_t, a_t, r_t tokenization as gridworld_test AD and adds a
compression transformer for long histories. Config fields such as n_transit and
short_memory_keep are environment-timestep counts; internally they are converted
to token counts by multiplying by three.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .compression import CompressionTransformer, ReconstructionDecoder
from .gpt2 import GPT2Transformer


class RAD(nn.Module):
    """RAD over interleaved state/action/reward tokens."""

    LATENT_UPDATE_MODES = ('replace', 'residual', 'multiplicative_gate', 'gru_gate')

    def __init__(self, config):
        super(RAD, self).__init__()

        self.config = config
        self._initial_device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = 3 * self.n_transit
        self.mixed_precision = config['mixed_precision']
        self.obs_dim = config['dim_obs']
        self.action_dim = config['dim_actions']
        self.learn_var = config.get('learn_var', False)

        self.n_compress_tokens = config.get('n_compress_tokens', 15)
        if self.n_compress_tokens % 3 != 0:
            raise ValueError('n_compress_tokens must be a multiple of 3 for s/a/r token consistency')

        default_keep = max(1, self.n_compress_tokens // 3)
        self.short_memory_keep = config.get('short_memory_keep', default_keep)
        self.short_memory_keep_tokens = 3 * self.short_memory_keep
        self.compress_n_layers = config.get('compress_n_layers', 2)
        self.compress_n_heads = config.get('compress_n_heads', 4)
        self.max_gradient_rounds = config.get('max_gradient_rounds', 2)
        self.max_compressions = config.get('max_compressions', None)
        self.max_context_tokens = 3 * config.get('max_context_length', self.n_transit)
        self.always_use_latent_prefix = config.get('always_use_latent_prefix', False)
        self.latent_update_mode = config.get('latent_update_mode', 'replace')
        if self.latent_update_mode not in self.LATENT_UPDATE_MODES:
            raise ValueError(
                f'Unknown latent_update_mode: {self.latent_update_mode}. '
                f'Expected one of {self.LATENT_UPDATE_MODES}'
            )

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        tf_dropout = config.get('tf_dropout', 0.1)

        self.ad_transformer = GPT2Transformer(
            d_model=tf_n_embd,
            n_heads=tf_n_head,
            n_layers=tf_n_layer,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
        )
        if config.get('gradient_checkpointing', False):
            self.ad_transformer.enable_gradient_checkpointing()

        self.embed_state = nn.Linear(self.obs_dim, tf_n_embd)
        self.embed_action = nn.Linear(self.action_dim, tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, tf_n_embd))
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, tf_n_embd))
        self.null_latent_tokens = nn.Parameter(torch.zeros(1, self.n_compress_tokens, tf_n_embd))

        self.pred_action = nn.Linear(
            tf_n_embd,
            2 * self.action_dim if self.learn_var else self.action_dim,
        )
        self.latent_residual_norm = nn.LayerNorm(tf_n_embd)
        self.latent_multiplicative_gate = nn.Linear(tf_n_embd, tf_n_embd)
        self.latent_gru_gate = nn.Linear(2 * tf_n_embd, tf_n_embd)
        self.latent_gru_candidate = nn.Linear(2 * tf_n_embd, tf_n_embd)

        self.compression_transformer = CompressionTransformer(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            n_compress_tokens=self.n_compress_tokens,
            dim_feedforward=tf_dim_feedforward,
            max_context_length=max(self.max_context_tokens + self.n_compress_tokens, self.max_seq_length),
        )
        self.reconstruction_decoder = ReconstructionDecoder(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
        )

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')

        nn.init.trunc_normal_(self.type_embedding, std=0.02)
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.null_latent_tokens, std=0.02)
        nn.init.zeros_(self.latent_multiplicative_gate.weight)
        nn.init.constant_(self.latent_multiplicative_gate.bias, config.get('latent_gate_init_bias', 4.0))
        nn.init.zeros_(self.latent_gru_gate.weight)
        nn.init.constant_(self.latent_gru_gate.bias, config.get('latent_gru_init_bias', -2.0))
        nn.init.zeros_(self.latent_gru_candidate.weight)
        nn.init.zeros_(self.latent_gru_candidate.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def _embed_state(self, states):
        return self.embed_state(states.float())

    def _embed_action(self, actions):
        return self.embed_action(actions.float())

    def _embed_reward(self, rewards):
        if rewards.dim() == 2:
            rewards = rearrange(rewards, 'b t -> b t 1')
        return self.embed_reward(rewards.float())

    def _build_token_sequence(self, states, actions, rewards, context_lengths=None):
        state_tokens = self._embed_state(states)
        action_tokens = self._embed_action(actions)
        reward_tokens = self._embed_reward(rewards)

        tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        tokens = tokens + self.type_embedding
        tokens = rearrange(tokens, 'b t m d -> b (t m) d')

        token_targets = actions.repeat_interleave(3, dim=1)

        bsz, timesteps = states.shape[:2]
        timestep_ids = torch.arange(timesteps, device=states.device).unsqueeze(0).expand(bsz, -1)
        if context_lengths is None:
            valid_timesteps = torch.ones_like(timestep_ids, dtype=torch.bool)
        else:
            valid_timesteps = timestep_ids < context_lengths.to(states.device).unsqueeze(1)

        valid_tokens = valid_timesteps.repeat_interleave(3, dim=1)
        token_positions = torch.arange(tokens.shape[1], device=states.device).unsqueeze(0).expand(bsz, -1)
        state_loss_mask = valid_tokens & (token_positions % 3 == 0)

        return tokens, state_loss_mask, token_targets

    def _action_loss(self, prediction, target):
        if not self.learn_var:
            return self.loss_fn(prediction, target)
        mean, log_var = prediction.split(self.action_dim, dim=-1)
        variance = torch.exp(log_var.clamp(min=-10.0, max=10.0))
        return self.loss_fn_gaussian(mean, target, variance)

    def _sample_action(self, prediction, sample):
        if self.learn_var:
            mean, log_var = prediction.split(self.action_dim, dim=-1)
            action = (
                mean + torch.exp(0.5 * log_var.clamp(-10.0, 10.0)) * torch.randn_like(mean)
                if sample else mean
            )
        elif sample:
            action = prediction + torch.randn_like(prediction)
        else:
            action = prediction

        if hasattr(self, 'action_low') and hasattr(self, 'action_high'):
            action = torch.maximum(torch.minimum(action, self.action_high), self.action_low)
        return action

    def _typed_state_token(self, states):
        return self._embed_state(states) + self.type_embedding[:, :, 0, :]

    def _typed_action_token(self, actions):
        return self._embed_action(actions) + self.type_embedding[:, :, 1, :]

    def _typed_reward_token(self, rewards):
        return self._embed_reward(rewards) + self.type_embedding[:, :, 2, :]

    def _get_attention_mask_for_latent(self, seq_len):
        """Generate attention mask for latent-prefix sequences; True=masked."""
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=self.device)
        recent_start = self.n_compress_tokens
        recent_len = seq_len - recent_start

        if recent_len > 0:
            # Latent queries must not read current recent tokens; otherwise later
            # recent outputs can receive future-token information through latents.
            mask[:recent_start, recent_start:] = True
            recent_mask = torch.triu(
                torch.ones((recent_len, recent_len), dtype=torch.bool, device=self.device),
                diagonal=1,
            )
            mask[recent_start:, recent_start:] = recent_mask

        return mask

    def _module_for_current_grad_mode(self, module):
        if torch.is_grad_enabled():
            return module
        return getattr(module, '_orig_mod', module)

    def _forward_ad_transformer(self, x, has_latent_prefix=False):
        ad_transformer = self._module_for_current_grad_mode(self.ad_transformer)
        if not has_latent_prefix:
            return ad_transformer(x, use_causal_mask=True)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        recent_len = seq_len - self.n_compress_tokens
        latent_type = self.latent_type_embedding.expand(batch_size, self.n_compress_tokens, -1)
        zero_type = torch.zeros(batch_size, recent_len, x.size(2), device=x.device)
        x = x + torch.cat([latent_type, zero_type], dim=1)

        attn_mask = self._get_attention_mask_for_latent(seq_len)
        return ad_transformer(x, attention_mask=attn_mask, use_causal_mask=False)

    def _update_latent_tokens(self, old_latent_tokens, candidate_latent_tokens):
        if old_latent_tokens is None or self.latent_update_mode == 'replace':
            return candidate_latent_tokens

        if old_latent_tokens.shape != candidate_latent_tokens.shape:
            raise ValueError(
                f'Latent update shape mismatch: old={old_latent_tokens.shape}, '
                f'candidate={candidate_latent_tokens.shape}'
            )

        if self.latent_update_mode == 'residual':
            return self.latent_residual_norm(old_latent_tokens + candidate_latent_tokens)

        if self.latent_update_mode == 'multiplicative_gate':
            gate = torch.sigmoid(self.latent_multiplicative_gate(old_latent_tokens))
            return gate * candidate_latent_tokens

        if self.latent_update_mode == 'gru_gate':
            update_input = torch.cat([old_latent_tokens, candidate_latent_tokens], dim=-1)
            update_gate = torch.sigmoid(self.latent_gru_gate(update_input))
            candidate_delta = torch.tanh(self.latent_gru_candidate(update_input))
            candidate_state = candidate_latent_tokens + candidate_delta
            return (1.0 - update_gate) * old_latent_tokens + update_gate * candidate_state

        raise RuntimeError(f'Unhandled latent_update_mode: {self.latent_update_mode}')

    def _compress_sequence(self, context_embed, allow_gradient, old_latent_tokens=None):
        grad_enabled = torch.is_grad_enabled() and allow_gradient
        with torch.set_grad_enabled(grad_enabled):
            compression_transformer = self._module_for_current_grad_mode(self.compression_transformer)
            latent_tokens = compression_transformer(context_embed)
            latent_tokens = self._update_latent_tokens(old_latent_tokens, latent_tokens)
        # Compiled CUDAGraph outputs can be overwritten by later compiled
        # invocations; latent memory is kept across compression rounds.
        latent_tokens = latent_tokens.clone()
        if not allow_gradient:
            latent_tokens = latent_tokens.detach()
        return latent_tokens

    def _compression_round_allows_gradient(self, compression_round, gradient_start_round):
        return compression_round >= gradient_start_round

    def _memory_sequence_len(self, latent_tokens, recent_context):
        latent_len = self.n_compress_tokens if self._uses_latent_prefix(latent_tokens) else 0
        recent_len = 0 if recent_context is None else recent_context.shape[1]
        return latent_len + recent_len

    def _uses_latent_prefix(self, latent_tokens):
        return latent_tokens is not None or self.always_use_latent_prefix

    def _null_latent_prefix(self, batch_size, device, dtype):
        return self.null_latent_tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _pack_memory_input(self, latent_tokens, recent_context):
        has_recent = recent_context is not None and recent_context.shape[1] > 0
        if latent_tokens is None and self.always_use_latent_prefix:
            if recent_context is None:
                return recent_context, False
            latent_tokens = self._null_latent_prefix(
                batch_size=recent_context.shape[0],
                device=recent_context.device,
                dtype=recent_context.dtype,
            )
        if latent_tokens is not None:
            if has_recent:
                return torch.cat([latent_tokens, recent_context], dim=1), True
            return latent_tokens, True
        return recent_context, False

    def _append_recent(self, recent_context, chunk):
        if recent_context is None or recent_context.shape[1] == 0:
            return chunk
        return torch.cat([recent_context, chunk], dim=1)

    def _uses_latent_prefix_from_state(self, has_latent_tokens):
        return has_latent_tokens or self.always_use_latent_prefix

    def _memory_sequence_len_from_state(self, has_latent_tokens, recent_len):
        latent_len = self.n_compress_tokens if self._uses_latent_prefix_from_state(has_latent_tokens) else 0
        return latent_len + recent_len

    def _count_compressions_until_fits(
        self,
        has_latent_tokens,
        recent_len,
        compression_round=0,
        respect_curriculum=True,
    ):
        num_compressions = 0
        while self._memory_sequence_len_from_state(has_latent_tokens, recent_len) > self.max_seq_length:
            if respect_curriculum and self.max_compressions is not None and compression_round >= self.max_compressions:
                keep_len = self.max_seq_length
                if self._uses_latent_prefix_from_state(has_latent_tokens):
                    keep_len -= self.n_compress_tokens
                recent_len = min(recent_len, max(0, keep_len))
                break

            keep_len = min(self.short_memory_keep_tokens, recent_len)
            prefix_len = recent_len - keep_len
            if prefix_len <= 0:
                keep_len = max(0, self.max_seq_length - self.n_compress_tokens)
                prefix_len = recent_len - keep_len
                if prefix_len <= 0:
                    break

            recent_len -= prefix_len
            has_latent_tokens = True
            compression_round += 1
            num_compressions += 1

        return has_latent_tokens, recent_len, num_compressions

    def _count_compressions_for_sequence(self, token_count, respect_curriculum=True):
        if token_count % 3 != 0:
            raise ValueError(f'RAD context must contain complete s/a/r triplets, got {token_count} tokens')

        has_latent_tokens = False
        recent_len = 0
        compression_round = 0
        total_compressions = 0
        cursor = 0

        while cursor < token_count:
            latent_len = self.n_compress_tokens if self._uses_latent_prefix_from_state(has_latent_tokens) else 0
            capacity = self.max_seq_length - latent_len
            remaining = token_count - cursor
            room = capacity - recent_len
            take_len = remaining if remaining <= room else min(room + 3, remaining)

            recent_len += take_len
            cursor += take_len

            has_latent_tokens, recent_len, num_compressions = self._count_compressions_until_fits(
                has_latent_tokens=has_latent_tokens,
                recent_len=recent_len,
                compression_round=compression_round,
                respect_curriculum=respect_curriculum,
            )
            compression_round += num_compressions
            total_compressions += num_compressions

        return total_compressions

    def _gradient_start_round(self, total_compressions):
        if self.max_gradient_rounds <= 0:
            return total_compressions
        return max(0, total_compressions - self.max_gradient_rounds)

    def _count_recurrent_compressions(self, timesteps, respect_curriculum=False):
        has_latent_tokens = False
        recent_len = 1
        compression_round = 0
        total_compressions = 0
        for _ in range(timesteps):
            recent_len += 3
            has_latent_tokens, recent_len, count = self._count_compressions_until_fits(
                has_latent_tokens,
                recent_len,
                compression_round=compression_round,
                respect_curriculum=respect_curriculum,
            )
            compression_round += count
            total_compressions += count
        return total_compressions

    def _append_recurrent_transition(
        self,
        latent_tokens,
        recent_tokens,
        transition_tokens,
        compression_round=0,
        gradient_start_round=0,
        respect_curriculum=True,
        collect_transitions=False,
    ):
        recent_tokens = self._append_recent(recent_tokens, transition_tokens)
        return self._compress_memory_until_fits(
            latent_tokens=latent_tokens,
            recent_context=recent_tokens,
            compression_round=compression_round,
            gradient_start_round=gradient_start_round,
            respect_curriculum=respect_curriculum,
            collect_transitions=collect_transitions,
        )

    def _compress_memory_until_fits(
        self,
        latent_tokens,
        recent_context,
        recent_state_mask=None,
        recent_targets=None,
        compression_round=0,
        gradient_start_round=0,
        respect_curriculum=True,
        collect_transitions=False,
    ):
        compression_info = {'num_compressions': 0, 'transitions': []}
        if recent_context is None:
            return latent_tokens, recent_context, recent_state_mask, recent_targets, compression_info

        while self._memory_sequence_len(latent_tokens, recent_context) > self.max_seq_length:
            if respect_curriculum and self.max_compressions is not None and compression_round >= self.max_compressions:
                keep_len = self.max_seq_length
                if self._uses_latent_prefix(latent_tokens):
                    keep_len -= self.n_compress_tokens
                keep_len = max(0, keep_len)
                recent_context = recent_context[:, -keep_len:] if keep_len > 0 else recent_context[:, :0]
                if recent_state_mask is not None:
                    recent_state_mask = recent_state_mask[:, -keep_len:] if keep_len > 0 else recent_state_mask[:, :0]
                if recent_targets is not None:
                    recent_targets = recent_targets[:, -keep_len:] if keep_len > 0 else recent_targets[:, :0]
                break

            keep_len = min(self.short_memory_keep_tokens, recent_context.shape[1])
            prefix_len = recent_context.shape[1] - keep_len
            if prefix_len <= 0:
                keep_len = max(0, self.max_seq_length - self.n_compress_tokens)
                prefix_len = recent_context.shape[1] - keep_len
                if prefix_len <= 0:
                    break

            prefix = recent_context[:, :prefix_len]
            previous_latent_tokens = latent_tokens
            if previous_latent_tokens is None and self.always_use_latent_prefix:
                previous_latent_tokens = self._null_latent_prefix(
                    prefix.shape[0],
                    prefix.device,
                    prefix.dtype,
                )
            compress_input = (
                torch.cat([previous_latent_tokens, prefix], dim=1)
                if previous_latent_tokens is not None
                else prefix
            )
            recent_context = recent_context[:, prefix_len:]

            if recent_state_mask is not None:
                recent_state_mask = recent_state_mask[:, prefix_len:]
            if recent_targets is not None:
                recent_targets = recent_targets[:, prefix_len:]

            latent_tokens = self._compress_sequence(
                compress_input,
                allow_gradient=self._compression_round_allows_gradient(
                    compression_round,
                    gradient_start_round,
                ),
                old_latent_tokens=previous_latent_tokens,
            )
            if collect_transitions:
                compression_info['transitions'].append({
                    'compression_input': compress_input,
                    'updated_latent_tokens': latent_tokens,
                })
            compression_round += 1
            compression_info['num_compressions'] += 1

        return latent_tokens, recent_context, recent_state_mask, recent_targets, compression_info

    def _roll_context_into_memory(
        self,
        context_embed,
        state_mask=None,
        token_targets=None,
        collect_transitions=False,
    ):
        if context_embed.shape[1] % 3 != 0:
            raise ValueError(
                f'RAD context must contain complete s/a/r triplets, got {context_embed.shape[1]} tokens'
            )

        latent_tokens = None
        recent_context = None
        recent_state_mask = None
        recent_targets = None
        compression_round = 0
        total_compressions = 0
        transitions = []
        planned_compressions = self._count_compressions_for_sequence(
            context_embed.shape[1],
            respect_curriculum=True,
        )
        gradient_start_round = self._gradient_start_round(planned_compressions)
        cursor = 0

        while cursor < context_embed.shape[1]:
            latent_len = self.n_compress_tokens if self._uses_latent_prefix(latent_tokens) else 0
            capacity = self.max_seq_length - latent_len
            recent_len = 0 if recent_context is None else recent_context.shape[1]
            remaining = context_embed.shape[1] - cursor
            room = capacity - recent_len
            take_len = remaining if remaining <= room else min(room + 3, remaining)

            chunk = context_embed[:, cursor:cursor + take_len]
            recent_context = self._append_recent(recent_context, chunk)

            if state_mask is not None:
                mask_chunk = state_mask[:, cursor:cursor + take_len]
                recent_state_mask = self._append_recent(recent_state_mask, mask_chunk)
            if token_targets is not None:
                target_chunk = token_targets[:, cursor:cursor + take_len]
                recent_targets = self._append_recent(recent_targets, target_chunk)

            cursor += take_len
            latent_tokens, recent_context, recent_state_mask, recent_targets, info = self._compress_memory_until_fits(
                latent_tokens=latent_tokens,
                recent_context=recent_context,
                recent_state_mask=recent_state_mask,
                recent_targets=recent_targets,
                compression_round=compression_round,
                gradient_start_round=gradient_start_round,
                respect_curriculum=True,
                collect_transitions=collect_transitions,
            )
            compression_round += info['num_compressions']
            total_compressions += info['num_compressions']
            transitions.extend(info['transitions'])

        return latent_tokens, recent_context, recent_state_mask, recent_targets, {
            'num_compressions': total_compressions,
            'transitions': transitions,
        }

    def forward(self, x, pretrain_compression=False):
        if pretrain_compression:
            return self.forward_pretrain_compression(x)

        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)
        context_lengths = x.get('context_lengths')
        if context_lengths is not None:
            context_lengths = context_lengths.to(self.device)

        tokens, state_mask, token_targets = self._build_token_sequence(
            states, actions, rewards, context_lengths=context_lengths
        )
        latent_tokens, recent_context, recent_state_mask, recent_targets, compression_info = self._roll_context_into_memory(
            tokens, state_mask=state_mask, token_targets=token_targets
        )

        transformer_input, has_latent = self._pack_memory_input(latent_tokens, recent_context)
        transformer_output = self._forward_ad_transformer(transformer_input, has_latent_prefix=has_latent)
        latent_len = self.n_compress_tokens if has_latent else 0
        recent_output = transformer_output[:, latent_len:]

        predicted_actions = self.pred_action(recent_output)
        selected_predictions = predicted_actions[recent_state_mask]
        selected_targets = recent_targets[recent_state_mask]

        loss_action = self._action_loss(selected_predictions, selected_targets)

        return {
            'loss_action': loss_action,
            'loss_total': loss_action,
            'acc_action': torch.zeros((), device=self.device),
            'num_compressions': compression_info['num_compressions'],
        }

    def forward_pretrain_compression(self, x):
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)
        next_states = x['next_states'].to(self.device)

        action_tokens = self._typed_action_token(actions)
        reward_tokens = self._typed_reward_token(rewards)
        next_state_tokens = self._typed_state_token(next_states)
        latent_tokens = None
        recent_tokens = self._typed_state_token(states[:, :1])
        compression_round = 0
        planned_compressions = self._count_recurrent_compressions(states.shape[1])
        gradient_start_round = self._gradient_start_round(planned_compressions)
        transitions = []
        for timestep in range(states.shape[1]):
            transition_tokens = torch.cat([
                action_tokens[:, timestep:timestep + 1],
                reward_tokens[:, timestep:timestep + 1],
                next_state_tokens[:, timestep:timestep + 1],
            ], dim=1)
            latent_tokens, recent_tokens, _, _, info = self._append_recurrent_transition(
                latent_tokens,
                recent_tokens,
                transition_tokens,
                compression_round=compression_round,
                gradient_start_round=gradient_start_round,
                respect_curriculum=False,
                collect_transitions=True,
            )
            compression_round += info['num_compressions']
            transitions.extend(info['transitions'])
        if not transitions:
            raise ValueError('Compression pretraining sample did not produce a recurrent transition')

        transition_losses = []
        for transition in transitions:
            compression_input = transition['compression_input']
            updated_latent_tokens = transition['updated_latent_tokens']
            reconstructed = self.reconstruction_decoder(
                updated_latent_tokens,
                compression_input.shape[1],
            )
            transition_losses.append(F.mse_loss(reconstructed, compression_input.detach()))
        recon_loss = torch.stack(transition_losses).mean()

        return {
            'loss_recon': recon_loss,
            'loss_total': recon_loss,
            'num_compressions': compression_round,
        }

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {'reward_episode': [], 'success': [], 'compression_events': []}
        reward_episode = np.zeros(vec_env.num_envs)
        success_episode = np.zeros(vec_env.num_envs, dtype=np.bool_)

        query_states = vec_env.reset()[..., :self.obs_dim]
        query_states = torch.as_tensor(query_states, device=self.device, dtype=torch.float)
        query_states = rearrange(query_states, 'e d -> e 1 d')

        latent_tokens = None
        recent_tokens = self._typed_state_token(query_states)
        compression_count = 0

        for step in range(eval_timesteps):
            transformer_input, has_latent = self._pack_memory_input(latent_tokens, recent_tokens)
            output = self._forward_ad_transformer(transformer_input, has_latent_prefix=has_latent)
            actions = self._sample_action(self.pred_action(output[:, -1]), sample=sample)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            action_tokens = rearrange(actions, 'e d -> e 1 d')

            reward_episode += rewards
            success_episode |= np.asarray(
                [bool(info.get('success', False)) for info in infos],
                dtype=np.bool_,
            )
            rewards_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float)
            rewards_tensor = rearrange(rewards_tensor, 'e -> e 1 1')

            query_states = torch.as_tensor(
                query_states[..., :self.obs_dim],
                device=self.device,
                dtype=torch.float,
            )
            query_states = rearrange(query_states, 'e d -> e 1 d')

            if dones[0]:
                outputs['reward_episode'].append(reward_episode.copy())
                outputs['success'].append(success_episode.copy())
                reward_episode = np.zeros(vec_env.num_envs)
                success_episode.fill(False)
                next_states = torch.as_tensor(
                    np.stack([
                        info['terminal_observation'][:self.obs_dim]
                        for info in infos
                    ]),
                    device=self.device,
                    dtype=torch.float,
                )
                next_states = rearrange(next_states, 'e d -> e 1 d')
            else:
                next_states = query_states

            new_tokens = torch.cat([
                self._typed_action_token(action_tokens),
                self._typed_reward_token(rewards_tensor),
                self._typed_state_token(next_states),
            ], dim=1)
            latent_tokens, recent_tokens, _, _, info = self._append_recurrent_transition(
                latent_tokens,
                recent_tokens,
                new_tokens,
                compression_round=compression_count,
                respect_curriculum=True,
            )
            if info['num_compressions'] > 0:
                compression_count += info['num_compressions']
                outputs['compression_events'].extend([step] * info['num_compressions'])

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.stack(outputs['success'], axis=1)
        outputs['total_compressions'] = compression_count
        return outputs

    def set_curriculum(self, max_compressions):
        self.max_compressions = max_compressions

    def set_obs_space(self, obs_space):
        self.obs_dim = min(self.config['dim_obs'], obs_space.shape[0])

    def set_action_space(self, action_space):
        low = torch.as_tensor(action_space.low, device=self.device, dtype=torch.float)
        high = torch.as_tensor(action_space.high, device=self.device, dtype=torch.float)
        if hasattr(self, 'action_low'):
            self.action_low.copy_(low)
            self.action_high.copy_(high)
        else:
            self.register_buffer('action_low', low)
            self.register_buffer('action_high', high)

    def load_pretrained_compression(self, pretrain_checkpoint_path):
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get('format') != 'metaworld-sar-v1':
            raise ValueError(
                f'{pretrain_checkpoint_path} uses the legacy packed-transition format; '
                're-run compression pretraining for the state/action/reward model'
            )
        if checkpoint.get('compression_pretrain_contract') != 'recurrent-transition-v1':
            raise ValueError(
                f'{pretrain_checkpoint_path} does not use recurrent-transition-v1; '
                're-run compression pretraining'
            )

        compression_state = {
            k.replace('compression_transformer.', ''): v
            for k, v in checkpoint['model'].items()
            if 'compression_transformer' in k
        }
        self.compression_transformer.load_state_dict(compression_state)

        decoder_state = {
            k.replace('reconstruction_decoder.', ''): v
            for k, v in checkpoint['model'].items()
            if 'reconstruction_decoder' in k
        }
        if 'position_queries' in decoder_state:
            pretrained = decoder_state['position_queries']
            current = self.reconstruction_decoder.position_queries
            if pretrained.shape != current.shape:
                merged = current.clone()
                copy_len = min(pretrained.shape[1], current.shape[1])
                merged[:, :copy_len, :] = pretrained[:, :copy_len, :]
                decoder_state['position_queries'] = merged
        self.reconstruction_decoder.load_state_dict(decoder_state)

        for module_name in ['embed_state', 'embed_action', 'embed_reward']:
            weight_key = f'{module_name}.weight'
            if weight_key in checkpoint['model']:
                state = {'weight': checkpoint['model'][weight_key]}
                bias_key = f'{module_name}.bias'
                if bias_key in checkpoint['model']:
                    state['bias'] = checkpoint['model'][bias_key]
                getattr(self, module_name).load_state_dict(state)

        if 'type_embedding' in checkpoint['model']:
            self.type_embedding.data.copy_(checkpoint['model']['type_embedding'])
        if 'latent_type_embedding' in checkpoint['model']:
            self.latent_type_embedding.data.copy_(checkpoint['model']['latent_type_embedding'])
        if 'null_latent_tokens' in checkpoint['model']:
            self.null_latent_tokens.data.copy_(checkpoint['model']['null_latent_tokens'])

        for module_name in [
            'latent_residual_norm',
            'latent_multiplicative_gate',
            'latent_gru_gate',
            'latent_gru_candidate',
        ]:
            prefix = f'{module_name}.'
            module_state = {
                key[len(prefix):]: value
                for key, value in checkpoint['model'].items()
                if key.startswith(prefix)
            }
            if module_state:
                getattr(self, module_name).load_state_dict(module_state)

        print(f"Loaded pre-trained compression from {pretrain_checkpoint_path}")

    def load_pretrained_ad(self, ad_checkpoint_path):
        checkpoint = torch.load(ad_checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'metaworld-sar-v1':
            raise ValueError(
                f'{ad_checkpoint_path} uses the legacy packed-transition format; '
                'train a compatible state/action/reward AD checkpoint'
            )
        ad_state = checkpoint['model']

        transformer_state = {
            k.replace('transformer.', ''): v
            for k, v in ad_state.items()
            if k.startswith('transformer.')
        }
        if transformer_state:
            self.ad_transformer.load_state_dict(transformer_state)
            print(f"  Loaded AD transformer ({len(transformer_state)} params)")

        for module_name in ['embed_state', 'embed_action', 'embed_reward', 'pred_action']:
            weight_key = f'{module_name}.weight'
            if weight_key in ad_state:
                state = {'weight': ad_state[weight_key]}
                bias_key = f'{module_name}.bias'
                if bias_key in ad_state:
                    state['bias'] = ad_state[bias_key]
                getattr(self, module_name).load_state_dict(state)
                print(f"  Loaded {module_name}")

        if 'type_embedding' in ad_state:
            self.type_embedding.data.copy_(ad_state['type_embedding'])
            print("  Loaded type_embedding")

        print(f"Loaded pre-trained AD from {ad_checkpoint_path}")
        print("  Note: Compression transformer is randomly initialized")
