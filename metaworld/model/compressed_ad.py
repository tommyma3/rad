"""
Recurrent Algorithm Distillation (RAD) Model.

This model extends the original AD with a recurrence/compression mechanism:
1. When sequence length exceeds max_seq_length, compress older history into latent tokens
2. Continue AD with [latent_tokens, recent_transitions, query_state]
3. Repeat compression as needed for very long sequences

Adapted for Meta-world continuous action spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from .compression import CompressionTransformer, ReconstructionDecoder
from .gpt2 import GPT2Transformer


class RAD(nn.Module):
    """
    Recurrent Algorithm Distillation (RAD) with GPT-2 style decoder-only transformer.

    Adapted for Meta-world continuous action spaces.
    """
    def __init__(self, config):
        super(RAD, self).__init__()

        self.config = config
        self._device = config['device']  # Initial device, may change with accelerator
        self.n_transit = config['n_transit']  # Max sequence length for AD transformer
        self.max_seq_length = config['n_transit']
        # max_context_length: max sample length from dataset
        # used by reconstruction decoder for variable-length sequences
        self.max_context_length = config.get('max_context_length', 2000)
        self.mixed_precision = config['mixed_precision']
        
        # Observation dimension handling
        if 'task' in config and (config['task'] == "hammer-v2" or config['task'] == "stick-push-v2" or config['task'] == "stick-pull-v2"):
            assert config['dim_obs'] == 18
            self.obs_dim_idx = list(range(18))
        else:
            assert config['dim_obs'] == 11
            self.obs_dim_idx = list(range(11))

        # Compression config
        self.n_compress_tokens = config.get('n_compress_tokens', 32)
        short_capacity = max(0, self.max_seq_length - self.n_compress_tokens - 1)
        self.short_memory_keep = min(config.get('short_memory_keep', self.n_compress_tokens), short_capacity)
        self.compress_n_layers = config.get('compress_n_layers', 3)
        self.compress_n_heads = config.get('compress_n_heads', 4)
        self.max_gradient_rounds = config.get('max_gradient_rounds', 2)
        self.use_recon_reg = config.get('use_recon_reg', True)
        self.recon_reg_weight = config.get('recon_reg_weight', 0.1)

        # Reward-weighted reconstruction config
        self.use_reward_weighted_recon = config.get('use_reward_weighted_recon', True)
        self.reward_weight_multiplier = config.get('reward_weight_multiplier', 5.0)
        
        # Curriculum settings
        self.max_compressions = config.get('max_compressions', None)  # None = unlimited
        
        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        tf_dropout = config.get('tf_dropout', 0.1)

        # GPT-2 style AD Transformer (Pre-LayerNorm, causal)
        self.ad_transformer = GPT2Transformer(
            d_model=tf_n_embd,
            n_heads=tf_n_head,
            n_layers=tf_n_layer,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
        )

        # Embeddings
        self.embed_context = nn.Linear(config['dim_obs'] * 2 + config['dim_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Linear(config['dim_obs'], tf_n_embd)
        
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # Action prediction head (continuous)
        if config.get('learn_var', False):
            self.pred_actions = nn.Linear(tf_n_embd, 2 * config['dim_actions'])
            self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')
        else:
            self.pred_actions = nn.Linear(tf_n_embd, config['dim_actions'])
        
        # Dynamics prediction (optional)
        if config.get('dynamics', False):
            self.embed_query_action = nn.Linear(config['dim_actions'], tf_n_embd)
            self.pred_rewards = nn.Linear(tf_n_embd, 1)
            
            if config.get('learn_transition', False):
                self.pred_next_states = nn.Linear(tf_n_embd, config['dim_obs'])

        # Compression Transformer
        self.compression_transformer = CompressionTransformer(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            n_compress_tokens=self.n_compress_tokens,
            dim_feedforward=tf_dim_feedforward,
        )
        
        # Reconstruction Decoder (for pre-training and optional regularization)
        # Use max_context_length to support variable-length sequences during fine-tuning
        self.reconstruction_decoder = ReconstructionDecoder(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            max_seq_length=self.max_context_length,
            dim_feedforward=tf_dim_feedforward,
        )
        
        # Special embedding to mark latent tokens
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, tf_n_embd))

        # Initialize weights
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)

    @property
    def device(self):
        """Get the actual device of the model (works with accelerator)."""
        return next(self.parameters()).device

    def _get_attention_mask_for_latent(self, seq_len):
        """
        Generate attention mask for sequences with latent prefix.
        All tokens can attend to latent tokens (first n_compress_tokens),
        but recent tokens use causal masking among themselves.
        
        Returns a boolean mask where True means "mask out" (don't attend).
        """
        # Create mask: False = can attend, True = masked
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=self.device)
        recent_start = self.n_compress_tokens
        recent_len = seq_len - recent_start
        
        # Causal mask for recent tokens attending to each other
        if recent_len > 0:
            recent_mask = torch.triu(
                torch.ones((recent_len, recent_len), dtype=torch.bool, device=self.device), 
                diagonal=1
            )
            mask[recent_start:, recent_start:] = recent_mask
        
        return mask

    def _forward_ad_transformer(self, x, has_latent_prefix=False):
        """
        Forward pass through AD transformer with proper masking.
        
        Args:
            x: (batch, seq_len, d_model) - input embeddings
            has_latent_prefix: whether the input starts with latent tokens
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        if has_latent_prefix:
            # Add latent type embedding to latent tokens
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            recent_len = seq_len - self.n_compress_tokens
            
            latent_type = self.latent_type_embedding.expand(batch_size, self.n_compress_tokens, -1)
            zero_type = torch.zeros(batch_size, recent_len, x.size(2), device=x.device)
            type_emb = torch.cat([latent_type, zero_type], dim=1)
            x = x + type_emb
            
            # Get attention mask for latent prefix
            attn_mask = self._get_attention_mask_for_latent(seq_len)
            output = self.ad_transformer(x, attention_mask=attn_mask, use_causal_mask=False)
        else:
            # Standard causal masking
            output = self.ad_transformer(x, use_causal_mask=True)
        
        return output

    def _compress_sequence(self, context_embed, compression_round):
        """
        Compress a sequence using the compression transformer.
        
        Args:
            context_embed: (batch, seq_len, d_model) - embedded transitions
            compression_round: int - which compression round (for gradient truncation)
            
        Returns:
            latent_tokens: (batch, n_compress_tokens, d_model)
        """
        latent_tokens = self.compression_transformer(context_embed)
        
        # Gradient truncation after max_gradient_rounds
        if compression_round >= self.max_gradient_rounds:
            latent_tokens = latent_tokens.detach()
            
        return latent_tokens

    def _memory_sequence_len(self, latent_tokens, recent_context, query_len=1):
        latent_len = 0 if latent_tokens is None else latent_tokens.shape[1]
        recent_len = 0 if recent_context is None else recent_context.shape[1]
        return latent_len + recent_len + query_len

    def _pack_memory_input(self, latent_tokens, recent_context, query_states_embed):
        """Build the AD input as long-term memory + short-term memory + query."""
        has_recent = recent_context is not None and recent_context.shape[1] > 0

        if latent_tokens is not None and has_recent:
            transformer_input, _ = pack([latent_tokens, recent_context, query_states_embed], 'b * d')
            return transformer_input, True
        if latent_tokens is not None:
            transformer_input, _ = pack([latent_tokens, query_states_embed], 'b * d')
            return transformer_input, True
        if has_recent:
            transformer_input, _ = pack([recent_context, query_states_embed], 'b * d')
            return transformer_input, False
        return query_states_embed, False

    def _append_recent(self, recent_context, chunk):
        if recent_context is None or recent_context.shape[1] == 0:
            return chunk
        return torch.cat([recent_context, chunk], dim=1)

    def _compress_memory_until_fits(
        self,
        latent_tokens,
        recent_context,
        recent_rewards=None,
        compression_round=0,
        compute_recon_loss=False,
        respect_curriculum=True,
    ):
        """
        Recursively compress old context into latent memory and keep a recent suffix.

        The resulting representation is always:
        latent long-term memory + recent short-term transitions + query.
        """
        compression_info = {
            'num_compressions': 0,
            'recon_loss': torch.tensor(0.0, device=self.device),
        }

        if recent_context is None:
            return latent_tokens, recent_context, recent_rewards, compression_info

        total_recon_loss = torch.tensor(0.0, device=recent_context.device)
        query_len = 1

        while self._memory_sequence_len(latent_tokens, recent_context, query_len=query_len) > self.max_seq_length:
            if respect_curriculum and self.max_compressions is not None and compression_round >= self.max_compressions:
                keep_len = self.max_seq_length - query_len
                if latent_tokens is not None:
                    keep_len -= self.n_compress_tokens
                keep_len = max(0, keep_len)
                recent_context = recent_context[:, -keep_len:] if keep_len > 0 else recent_context[:, :0]
                if recent_rewards is not None:
                    recent_rewards = recent_rewards[:, -keep_len:] if keep_len > 0 else recent_rewards[:, :0]
                break

            keep_len = min(self.short_memory_keep, recent_context.shape[1])
            prefix_len = recent_context.shape[1] - keep_len

            if prefix_len <= 0:
                keep_len = max(0, self.max_seq_length - self.n_compress_tokens - query_len)
                prefix_len = recent_context.shape[1] - keep_len
                if prefix_len <= 0:
                    break

            transition_prefix = recent_context[:, :prefix_len]
            if latent_tokens is not None:
                compress_input = torch.cat([latent_tokens, transition_prefix], dim=1)
            else:
                compress_input = transition_prefix

            recent_context = recent_context[:, prefix_len:]

            prefix_rewards = None
            if recent_rewards is not None:
                prefix_rewards = recent_rewards[:, :prefix_len]
                recent_rewards = recent_rewards[:, prefix_len:]

            new_latent = self._compress_sequence(compress_input, compression_round)

            if compute_recon_loss and self.training and self.use_recon_reg:
                reconstructed = self.reconstruction_decoder(new_latent, compress_input.shape[1])
                position_mse = ((reconstructed - compress_input.detach()) ** 2).mean(dim=-1)

                if (self.use_reward_weighted_recon and prefix_rewards is not None and latent_tokens is None):
                    reward_weights = 1.0 + (self.reward_weight_multiplier - 1.0) * (prefix_rewards.squeeze(-1) > 0).float()
                    recon_loss = (position_mse * reward_weights).mean()
                else:
                    recon_loss = position_mse.mean()

                total_recon_loss = total_recon_loss + recon_loss

            latent_tokens = new_latent
            compression_round += 1
            compression_info['num_compressions'] += 1

        compression_info['recon_loss'] = total_recon_loss
        return latent_tokens, recent_context, recent_rewards, compression_info

    def _roll_context_into_memory(self, context_embed, rewards=None, compute_recon_loss=False):
        """Roll an offline context through the same chunked memory update used online."""
        latent_tokens = None
        recent_context = None
        recent_rewards = None
        compression_round = 0
        total_recon_loss = torch.tensor(0.0, device=context_embed.device)
        total_compressions = 0
        cursor = 0

        while cursor < context_embed.shape[1]:
            latent_len = 0 if latent_tokens is None else self.n_compress_tokens
            capacity = self.max_seq_length - latent_len - 1
            recent_len = 0 if recent_context is None else recent_context.shape[1]
            remaining = context_embed.shape[1] - cursor
            room = capacity - recent_len

            if remaining <= room:
                take_len = remaining
            else:
                take_len = max(1, room + 1)
                take_len = min(take_len, remaining)

            chunk = context_embed[:, cursor:cursor + take_len]
            recent_context = self._append_recent(recent_context, chunk)

            if rewards is not None:
                reward_chunk = rewards[:, cursor:cursor + take_len]
                recent_rewards = self._append_recent(recent_rewards, reward_chunk)

            cursor += take_len

            latent_tokens, recent_context, recent_rewards, compression_info = self._compress_memory_until_fits(
                latent_tokens=latent_tokens,
                recent_context=recent_context,
                recent_rewards=recent_rewards,
                compression_round=compression_round,
                compute_recon_loss=compute_recon_loss,
                respect_curriculum=True,
            )
            compression_round += compression_info['num_compressions']
            total_compressions += compression_info['num_compressions']
            total_recon_loss = total_recon_loss + compression_info['recon_loss']

        return latent_tokens, recent_context, {
            'num_compressions': total_compressions,
            'recon_loss': total_recon_loss,
        }

    def _forward_with_compression(self, context_embed, query_states_embed, rewards=None):
        """Forward pass using recursive long-term memory plus short-term context."""
        latent_tokens, recent_context, compression_info = self._roll_context_into_memory(
            context_embed,
            rewards=rewards,
            compute_recon_loss=True,
        )

        full_input, has_latent = self._pack_memory_input(latent_tokens, recent_context, query_states_embed)
        transformer_output = self._forward_ad_transformer(full_input, has_latent_prefix=has_latent)

        return transformer_output, compression_info

    def forward(self, x):
        """
        Training forward pass with automatic compression for long sequences.
        """
        query_states = x['query_states'].to(self.device)
        target_actions = x['target_actions'].to(self.device)
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        next_states = x['next_states'].to(self.device)
        rewards = x['rewards'].to(self.device)
        rewards = rearrange(rewards, 'b n -> b n 1')

        # Embed query state
        query_states_embed = self.embed_query_state(query_states)
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        # Embed context transitions
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)

        # Forward with compression if needed (pass rewards for weighted reconstruction)
        transformer_output, compression_info = self._forward_with_compression(
            context_embed, query_states_embed, rewards=rewards
        )

        result = {}

        # Predict action from last position
        if self.config.get('learn_var', False):
            dist = self.pred_actions(transformer_output[:, -1])
            mean = dist[:, :self.config['dim_actions']]
            var = torch.exp(dist[:, self.config['dim_actions']:])
            loss_action = self.loss_fn_gaussian(mean, target_actions, var)
        else:
            predicted_actions = self.pred_actions(transformer_output[:, -1])
            loss_action = self.loss_fn(predicted_actions, target_actions)

        result['loss_action'] = loss_action
        result['acc_action'] = torch.tensor(0.0, device=self.device)  # Not applicable for continuous
        result['num_compressions'] = compression_info['num_compressions']
        
        # Add reconstruction regularization if enabled
        if self.training and self.use_recon_reg and compression_info['recon_loss'] != 0.0:
            result['loss_recon'] = compression_info['recon_loss']
            result['loss_total'] = loss_action + self.recon_reg_weight * compression_info['recon_loss']
        else:
            result['loss_recon'] = torch.tensor(0.0, device=self.device)
            result['loss_total'] = loss_action

        return result

    def forward_pretrain_compression(self, x):
        """
        Pre-training forward pass for compression transformer only.
        Uses reconstruction loss to learn good compression.
        """
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        next_states = x['next_states'].to(self.device)
        rewards = x['rewards'].to(self.device)
        rewards = rearrange(rewards, 'b n -> b n 1')

        # Embed context transitions
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        
        # Compress
        latent_tokens = self.compression_transformer(context_embed)
        
        # Reconstruct
        reconstructed = self.reconstruction_decoder(latent_tokens, context_embed.shape[1])
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, context_embed.detach())
        
        result = {
            'loss_recon': recon_loss,
            'loss_total': recon_loss,
        }
        
        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, sample_size=1, beam_start=50, sample=True):
        """
        In-context evaluation with rolling compression.
        """
        outputs = {}
        outputs['reward_episode'] = []
        outputs['success'] = []
        outputs['compression_events'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        success = np.zeros(vec_env.num_envs)
        n_envs = vec_env.num_envs

        query_states = vec_env.reset()[:, self.obs_dim_idx]
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.float)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(query_states)
        
        # Initialize: no latent tokens, no transition history
        latent_tokens = None
        transition_buffer = None
        
        compression_count = 0

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            # Build input sequence: long-term memory + short-term memory + query.
            transformer_input, has_latent = self._pack_memory_input(
                latent_tokens, transition_buffer, query_states_embed
            )

            # Forward through AD transformer
            output = self._forward_ad_transformer(transformer_input, has_latent_prefix=has_latent)
            
            if self.config.get('learn_var', False):
                dist = self.pred_actions(output[:, -1])
                mean = dist[:, :self.config['dim_actions']]
                std = torch.exp(dist[:, self.config['dim_actions']:] / 2)
                actions = (std * torch.randn_like(mean) + mean) if sample else mean
            else:
                mean = self.pred_actions(output[:, -1])
                std = torch.ones_like(mean)
                actions = (std * torch.randn_like(mean) + mean) if sample else mean

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e d -> e 1 d')

            reward_episode += rewards
            rewards_tensor = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards_tensor = rearrange(rewards_tensor, 'e -> e 1 1')

            query_states = torch.tensor(query_states[:, self.obs_dim_idx], device=self.device, requires_grad=False, dtype=torch.float)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            success += np.array([info['success'] for info in infos])

            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                outputs['success'].append(success > 0.0)
                success = np.zeros(vec_env.num_envs)

                states_next = torch.tensor(np.stack([info['terminal_observation'][self.obs_dim_idx] for info in infos]),
                                           device=self.device, dtype=torch.float)
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach()

            query_states_embed = self.embed_query_state(query_states)

            # Embed new transition
            new_transition, _ = pack([query_states_prev, actions, rewards_tensor, states_next], 'e i *')
            new_transition_embed = self.embed_context(new_transition)

            # Add to buffer
            if transition_buffer is not None:
                transition_buffer = torch.cat([transition_buffer, new_transition_embed], dim=1)
            else:
                transition_buffer = new_transition_embed

            # Recursively compress old memory if needed, preserving a recent suffix.
            latent_tokens, transition_buffer, _, compression_info = self._compress_memory_until_fits(
                latent_tokens=latent_tokens,
                recent_context=transition_buffer,
                compression_round=compression_count,
                compute_recon_loss=False,
                respect_curriculum=True,
            )

            if compression_info['num_compressions'] > 0:
                compression_count += compression_info['num_compressions']
                outputs['compression_events'].extend([step] * compression_info['num_compressions'])

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.maximum.accumulate(np.stack(outputs['success'], axis=1), axis=-1)
        outputs['total_compressions'] = compression_count

        return outputs
    
    def set_curriculum(self, max_compressions):
        """Set curriculum limit on number of compressions."""
        self.max_compressions = max_compressions
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage at the cost of compute."""
        self.ad_transformer.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled for AD transformer")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.ad_transformer.disable_gradient_checkpointing()
    
    def set_obs_space(self, obs_space):
        # Use register_buffer so tensors move with the model
        self.register_buffer('obs_low', torch.tensor(obs_space.low[:self.config['dim_obs']], requires_grad=False, dtype=torch.float))
        self.register_buffer('obs_high', torch.tensor(obs_space.high[:self.config['dim_obs']], requires_grad=False, dtype=torch.float))
    
    def set_action_space(self, action_space):
        # Use register_buffer so tensors move with the model
        self.register_buffer('action_low', torch.tensor(action_space.low, requires_grad=False, dtype=torch.float))
        self.register_buffer('action_high', torch.tensor(action_space.high, requires_grad=False, dtype=torch.float))
        
    def load_pretrained_compression(self, pretrain_checkpoint_path):
        """Load pre-trained compression transformer weights."""
        checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu')  # Load to CPU first, let accelerator handle device
        
        # Load compression transformer
        compression_state = {k.replace('compression_transformer.', ''): v 
                           for k, v in checkpoint['model'].items() 
                           if 'compression_transformer' in k}
        self.compression_transformer.load_state_dict(compression_state)
        
        # Load reconstruction decoder
        decoder_state = {k.replace('reconstruction_decoder.', ''): v 
                        for k, v in checkpoint['model'].items() 
                        if 'reconstruction_decoder' in k}
        
        if 'position_queries' in decoder_state:
            pretrained_pos_queries = decoder_state['position_queries']
            current_pos_queries = self.reconstruction_decoder.position_queries
            
            if pretrained_pos_queries.shape != current_pos_queries.shape:
                pretrained_len = pretrained_pos_queries.shape[1]
                new_pos_queries = current_pos_queries.clone()
                new_pos_queries[:, :pretrained_len, :] = pretrained_pos_queries
                decoder_state['position_queries'] = new_pos_queries
        
        self.reconstruction_decoder.load_state_dict(decoder_state)
        
        # Load embedding layer (shared)
        if 'embed_context.weight' in checkpoint['model']:
            self.embed_context.load_state_dict({
                'weight': checkpoint['model']['embed_context.weight'],
                'bias': checkpoint['model']['embed_context.bias']
            })
        
        print(f"Loaded pre-trained compression from {pretrain_checkpoint_path}")

