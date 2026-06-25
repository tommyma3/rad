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

from env import map_dark_states
from .compression import CompressionTransformer, ReconstructionDecoder
from .gpt2 import GPT2Transformer


class RAD(nn.Module):
    """RAD over interleaved state/action/reward tokens."""

    def __init__(self, config):
        super(RAD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = 3 * self.n_transit
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        self.num_actions = config['num_actions']

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

        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        self.embed_action = nn.Linear(self.num_actions, tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, tf_n_embd))
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, tf_n_embd))
        self.null_latent_tokens = nn.Parameter(torch.zeros(1, self.n_compress_tokens, tf_n_embd))

        self.pred_action = nn.Linear(tf_n_embd, self.num_actions)

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

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        nn.init.trunc_normal_(self.type_embedding, std=0.02)
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.null_latent_tokens, std=0.02)

    def _state_ids(self, states):
        return map_dark_states(states, self.grid_size).to(torch.long)

    def _embed_state(self, states):
        return self.embed_state(self._state_ids(states))

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

        action_targets = actions.argmax(dim=-1)
        token_targets = action_targets.repeat_interleave(3, dim=1)

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

    def _forward_ad_transformer(self, x, has_latent_prefix=False):
        if not has_latent_prefix:
            return self.ad_transformer(x, use_causal_mask=True)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        recent_len = seq_len - self.n_compress_tokens
        latent_type = self.latent_type_embedding.expand(batch_size, self.n_compress_tokens, -1)
        zero_type = torch.zeros(batch_size, recent_len, x.size(2), device=x.device)
        x = x + torch.cat([latent_type, zero_type], dim=1)

        attn_mask = self._get_attention_mask_for_latent(seq_len)
        return self.ad_transformer(x, attention_mask=attn_mask, use_causal_mask=False)

    def _compress_sequence(self, context_embed, compression_round):
        latent_tokens = self.compression_transformer(context_embed)
        if compression_round >= self.max_gradient_rounds:
            latent_tokens = latent_tokens.detach()
        return latent_tokens

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

    def _compress_memory_until_fits(
        self,
        latent_tokens,
        recent_context,
        recent_state_mask=None,
        recent_targets=None,
        compression_round=0,
        respect_curriculum=True,
    ):
        compression_info = {'num_compressions': 0}
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
            compress_input = torch.cat([latent_tokens, prefix], dim=1) if latent_tokens is not None else prefix
            recent_context = recent_context[:, prefix_len:]

            if recent_state_mask is not None:
                recent_state_mask = recent_state_mask[:, prefix_len:]
            if recent_targets is not None:
                recent_targets = recent_targets[:, prefix_len:]

            latent_tokens = self._compress_sequence(compress_input, compression_round)
            compression_round += 1
            compression_info['num_compressions'] += 1

        return latent_tokens, recent_context, recent_state_mask, recent_targets, compression_info

    def _roll_context_into_memory(self, context_embed, state_mask=None, token_targets=None):
        latent_tokens = None
        recent_context = None
        recent_state_mask = None
        recent_targets = None
        compression_round = 0
        total_compressions = 0
        cursor = 0

        while cursor < context_embed.shape[1]:
            latent_len = self.n_compress_tokens if self._uses_latent_prefix(latent_tokens) else 0
            capacity = self.max_seq_length - latent_len
            recent_len = 0 if recent_context is None else recent_context.shape[1]
            remaining = context_embed.shape[1] - cursor
            room = capacity - recent_len
            take_len = remaining if remaining <= room else max(1, min(room + 1, remaining))

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
                respect_curriculum=True,
            )
            compression_round += info['num_compressions']
            total_compressions += info['num_compressions']

        return latent_tokens, recent_context, recent_state_mask, recent_targets, {
            'num_compressions': total_compressions,
        }

    def forward(self, x):
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

        logits_actions = self.pred_action(recent_output)
        selected_logits = logits_actions[recent_state_mask]
        selected_targets = recent_targets[recent_state_mask]

        loss_action = self.loss_fn(selected_logits, selected_targets)
        acc_action = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()

        return {
            'loss_action': loss_action,
            'loss_total': loss_action,
            'loss_recon': torch.tensor(0.0, device=self.device),
            'acc_action': acc_action,
            'num_compressions': compression_info['num_compressions'],
        }

    def forward_pretrain_compression(self, x):
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)

        tokens, _, _ = self._build_token_sequence(states, actions, rewards)
        latent_tokens = self.compression_transformer(tokens)
        reconstructed = self.reconstruction_decoder(latent_tokens, tokens.shape[1])
        recon_loss = F.mse_loss(reconstructed, tokens.detach())

        return {
            'loss_recon': recon_loss,
            'loss_total': recon_loss,
        }

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {'reward_episode': [], 'compression_events': []}
        reward_episode = np.zeros(vec_env.num_envs)

        query_states = vec_env.reset()
        query_states = torch.as_tensor(query_states, device=self.device, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')

        latent_tokens = None
        recent_tokens = self._typed_state_token(query_states)
        compression_count = 0

        for step in range(eval_timesteps):
            transformer_input, has_latent = self._pack_memory_input(latent_tokens, recent_tokens)
            output = self._forward_ad_transformer(transformer_input, has_latent_prefix=has_latent)
            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions_onehot = rearrange(actions, 'e -> e 1')
            actions_onehot = F.one_hot(actions_onehot, num_classes=self.num_actions).float()

            reward_episode += rewards
            rewards_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float)
            rewards_tensor = rearrange(rewards_tensor, 'e -> e 1 1')

            query_states = torch.as_tensor(query_states, device=self.device, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')

            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                next_states = torch.as_tensor(
                    np.stack([info['terminal_observation'] for info in infos]),
                    device=self.device,
                    dtype=torch.long,
                )
                next_states = rearrange(next_states, 'e d -> e 1 d')
            else:
                next_states = query_states

            new_tokens = torch.cat([
                self._typed_action_token(actions_onehot),
                self._typed_reward_token(rewards_tensor),
                self._typed_state_token(next_states),
            ], dim=1)
            recent_tokens = torch.cat([recent_tokens, new_tokens], dim=1)

            latent_tokens, recent_tokens, _, _, info = self._compress_memory_until_fits(
                latent_tokens=latent_tokens,
                recent_context=recent_tokens,
                compression_round=compression_count,
                respect_curriculum=True,
            )
            if info['num_compressions'] > 0:
                compression_count += info['num_compressions']
                outputs['compression_events'].extend([step] * info['num_compressions'])

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['total_compressions'] = compression_count
        return outputs

    def set_curriculum(self, max_compressions):
        self.max_compressions = max_compressions

    def load_pretrained_compression(self, pretrain_checkpoint_path):
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.device)

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

        print(f"Loaded pre-trained compression from {pretrain_checkpoint_path}")

    def load_pretrained_ad(self, ad_checkpoint_path):
        checkpoint = torch.load(ad_checkpoint_path, map_location='cpu')
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
