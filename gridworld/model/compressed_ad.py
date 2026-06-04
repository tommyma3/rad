"""
Recurrent Algorithm Distillation (RAD) Model.

This model extends the original AD with a recurrence/compression mechanism:
1. When sequence length exceeds max_seq_length, compress older history into latent tokens
2. Continue AD with [latent_tokens, recent_transitions, query_state]
3. Repeat compression as needed for very long sequences

Uses GPT-2 style Pre-LayerNorm transformer architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse
from .compression import CompressionTransformer, ReconstructionDecoder
from .gpt2 import GPT2Transformer


class RAD(nn.Module):
    """
    Recurrent Algorithm Distillation (RAD) with GPT-2 style decoder-only transformer.

    Uses Pre-LayerNorm architecture as in the original GPT-2 paper,
    with an additional compression transformer for handling long sequences.
    """
    def __init__(self, config):
        super(RAD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']  # Max sequence length for AD transformer
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

        # Compression config
        self.n_compress_tokens = config.get('n_compress_tokens', 40)
        self.compress_n_layers = config.get('compress_n_layers', 2)
        self.compress_n_heads = config.get('compress_n_heads', 4)
        self.max_gradient_rounds = config.get('max_gradient_rounds', 2)
        self.use_recon_reg = config.get('use_recon_reg', True)
        self.recon_reg_weight = config.get('recon_reg_weight', 0.1)
        
        # Reward-weighted reconstruction config
        self.use_reward_weighted_recon = config.get('use_reward_weighted_recon', True)
        self.reward_weight_multiplier = config.get('reward_weight_multiplier', 5.0)  # Weight for positive reward transitions
        
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

        # Embeddings (shared)
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        
        # Action prediction head
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        # Compression Transformer
        self.compression_transformer = CompressionTransformer(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            n_compress_tokens=self.n_compress_tokens,
            dim_feedforward=tf_dim_feedforward,
        )
        
        # Reconstruction decoder (pretrain / regularization)
        self.reconstruction_decoder = ReconstructionDecoder(
            d_model=tf_n_embd,
            n_heads=self.compress_n_heads,
            n_layers=self.compress_n_layers,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
        )
        
        # Latent-type embedding
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, tf_n_embd))

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        # Initialize weights
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)

    def _get_attention_mask_for_latent(self, seq_len):
        """Generate attention mask for latent-prefix sequences; True=masked."""
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

    def _forward_with_compression(self, context_embed, query_states_embed, rewards=None):
        """
        Forward pass with rolling compression for long sequences.
        
        Args:
            context_embed: (batch, context_len, d_model) - all transition embeddings
            query_states_embed: (batch, 1, d_model) - query state embedding
            rewards: (batch, context_len, 1) - optional rewards for weighted reconstruction
            
        Returns:
            transformer_output: (batch, seq_len, d_model)
            compression_info: dict with compression statistics
        """
        batch_size = context_embed.shape[0]
        context_len = context_embed.shape[1]
        
        # Track reward positions for weighted reconstruction
        self._current_rewards = rewards
        self._compressed_start_idx = 0
        
        # Calculate available space: max_seq_length - 1 (for query state)
        available_for_context = self.max_seq_length - 1
        
        compression_info = {
            'num_compressions': 0,
            'recon_loss': 0.0,
        }
        
        # Check if compression is needed
        if context_len <= available_for_context:
            # No compression needed - standard forward
            full_input, _ = pack([context_embed, query_states_embed], 'b * d')
            transformer_output = self._forward_ad_transformer(full_input, has_latent_prefix=False)
            return transformer_output, compression_info
        
        # Compression is needed
        latent_tokens = None
        remaining_context = context_embed
        compression_round = 0
        total_recon_loss = 0.0
        
        while True:
            if latent_tokens is not None:
                # We have latent tokens - available space reduced
                available_for_new = self.max_seq_length - self.n_compress_tokens - 1  # -1 for query
            else:
                available_for_new = self.max_seq_length - 1  # -1 for query
            
            current_context_len = remaining_context.shape[1]
            
            # Check if we can fit everything now
            total_needed = current_context_len + (self.n_compress_tokens if latent_tokens is not None else 0) + 1
            
            if total_needed <= self.max_seq_length:
                # We can fit - build final input and exit loop
                break
            
            # Check curriculum limit
            if self.max_compressions is not None and compression_round >= self.max_compressions:
                # Truncate instead of compressing more
                keep_len = available_for_new
                remaining_context = remaining_context[:, -keep_len:]
                break
            
            # Need to compress
            # Determine how much to compress
            if latent_tokens is not None:
                # Include latent tokens in compression input
                # Keep some recent transitions outside compression
                compress_end = current_context_len - available_for_new + self.n_compress_tokens
                compress_input = torch.cat([latent_tokens, remaining_context[:, :compress_end - self.n_compress_tokens]], dim=1)
                remaining_context = remaining_context[:, compress_end - self.n_compress_tokens:]
            else:
                # First compression - compress older transitions
                compress_end = current_context_len - available_for_new + self.n_compress_tokens
                compress_input = remaining_context[:, :compress_end]
                remaining_context = remaining_context[:, compress_end:]
            
            # Perform compression
            new_latent = self._compress_sequence(compress_input, compression_round)
            
            # Optional: compute reward-weighted reconstruction loss for regularization
            # This gives higher weight to transitions with positive rewards (goal-finding)
            if self.training and self.use_recon_reg:
                reconstructed = self.reconstruction_decoder(new_latent, compress_input.shape[1])
                
                # Compute per-position MSE
                position_mse = ((reconstructed - compress_input.detach()) ** 2).mean(dim=-1)  # (batch, seq_len)
                
                # Apply reward weighting if enabled and rewards are available
                if (self.use_reward_weighted_recon and 
                    self._current_rewards is not None and 
                    compression_round == 0):
                    # Get rewards for the compressed positions
                    compress_rewards = self._current_rewards[:, self._compressed_start_idx:self._compressed_start_idx + compress_end]
                    compress_rewards = compress_rewards.squeeze(-1)  # (batch, seq_len)
                    
                    # Weight: 1.0 for zero-reward transitions, reward_weight_multiplier for positive-reward transitions
                    reward_weights = 1.0 + (self.reward_weight_multiplier - 1.0) * (compress_rewards > 0).float()
                    recon_loss = (position_mse * reward_weights).mean()
                    
                    # Update the start index for next compression
                    self._compressed_start_idx += compress_end
                else:
                    recon_loss = position_mse.mean()
                
                total_recon_loss = total_recon_loss + recon_loss
            
            latent_tokens = new_latent
            compression_round += 1
            compression_info['num_compressions'] = compression_round
        
        # Build final input for AD transformer
        if latent_tokens is not None:
            full_input, _ = pack([latent_tokens, remaining_context, query_states_embed], 'b * d')
            transformer_output = self._forward_ad_transformer(full_input, has_latent_prefix=True)
        else:
            full_input, _ = pack([remaining_context, query_states_embed], 'b * d')
            transformer_output = self._forward_ad_transformer(full_input, has_latent_prefix=False)
        
        compression_info['recon_loss'] = total_recon_loss
        
        return transformer_output, compression_info

    def forward(self, x):
        """
        Training forward pass with automatic compression for long sequences.
        
        All samples in batch have same context length (handled by collate_fn),
        enabling fully batched GPU processing.
        """
        query_states = x['query_states'].to(self.device)
        target_actions = x['target_actions'].to(self.device)
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        next_states = x['next_states'].to(self.device)
        rewards = x['rewards'].to(self.device)
        rewards = rearrange(rewards, 'b n -> b n 1')

        # Embed query state
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        # Embed context transitions
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)

        # Forward with compression (fully batched - all samples have same length)
        transformer_output, compression_info = self._forward_with_compression(
            context_embed, query_states_embed, rewards=rewards
        )

        result = {}

        # Predict action from last position
        logits_actions = self.pred_action(transformer_output[:, -1])

        loss_action = self.loss_fn(logits_actions, target_actions)
        acc_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_action
        result['acc_action'] = acc_action
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

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        """
        In-context evaluation with rolling compression.
        Maintains state across steps, compressing when needed.
        """
        outputs = {}
        outputs['reward_episode'] = []
        outputs['compression_events'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        n_envs = vec_env.num_envs

        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        
        # Initialize: no latent tokens, no transition history
        latent_tokens = None  # (n_envs, n_compress_tokens, d_model) when set
        transition_buffer = None  # (n_envs, buffer_len, d_model) - recent transitions
        
        compression_count = 0

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            # Build input sequence
            if latent_tokens is not None and transition_buffer is not None:
                transformer_input, _ = pack([latent_tokens, transition_buffer, query_states_embed], 'e * h')
                has_latent = True
            elif transition_buffer is not None:
                transformer_input, _ = pack([transition_buffer, query_states_embed], 'e * h')
                has_latent = False
            else:
                transformer_input = query_states_embed
                has_latent = False

            # Forward through AD transformer (GPT-2 style)
            output = self._forward_ad_transformer(transformer_input, has_latent_prefix=has_latent)
            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions_onehot = rearrange(actions, 'e -> e 1 1')
            actions_onehot = F.one_hot(actions_onehot, num_classes=self.config['num_actions']).float()

            reward_episode += rewards
            rewards_tensor = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards_tensor = rearrange(rewards_tensor, 'e -> e 1 1')

            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')

            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)

                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]),
                                           device=self.device, dtype=torch.float)
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)

            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))

            # Embed new transition
            new_transition, _ = pack([query_states_prev, actions_onehot, rewards_tensor, states_next], 'e i *')
            new_transition_embed = self.embed_context(new_transition)  # (e, 1, d_model)

            # Add to buffer
            if transition_buffer is not None:
                transition_buffer = torch.cat([transition_buffer, new_transition_embed], dim=1)
            else:
                transition_buffer = new_transition_embed

            # Check if compression is needed
            # Available space: max_seq_length - n_compress_tokens (if we have latent) - 1 (for query)
            if latent_tokens is not None:
                max_buffer_len = self.max_seq_length - self.n_compress_tokens - 1
            else:
                max_buffer_len = self.max_seq_length - 1

            if transition_buffer.shape[1] >= max_buffer_len:
                # Compress
                if latent_tokens is not None:
                    compress_input = torch.cat([latent_tokens, transition_buffer], dim=1)
                else:
                    compress_input = transition_buffer
                
                # Perform compression
                latent_tokens = self.compression_transformer(compress_input)
                transition_buffer = None  # Reset buffer
                
                compression_count += 1
                outputs['compression_events'].append(step)

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['total_compressions'] = compression_count

        return outputs
    
    def set_curriculum(self, max_compressions):
        """Set curriculum limit on number of compressions."""
        self.max_compressions = max_compressions
        
    def load_pretrained_compression(self, pretrain_checkpoint_path):
        """Load pre-trained compression transformer weights."""
        checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.device)
        
        # Load compression transformer
        compression_state = {k.replace('compression_transformer.', ''): v 
                           for k, v in checkpoint['model'].items() 
                           if 'compression_transformer' in k}
        self.compression_transformer.load_state_dict(compression_state)
        
        # Load reconstruction decoder (handle potential shape mismatch in position_queries)
        decoder_state = {k.replace('reconstruction_decoder.', ''): v 
                        for k, v in checkpoint['model'].items() 
                        if 'reconstruction_decoder' in k}
        
        # Handle position_queries shape mismatch (pretrain uses smaller max_seq_length)
        if 'position_queries' in decoder_state:
            pretrained_pos_queries = decoder_state['position_queries']
            current_pos_queries = self.reconstruction_decoder.position_queries
            
            if pretrained_pos_queries.shape != current_pos_queries.shape:
                # Copy pretrained weights into the beginning of the larger tensor
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

    def load_pretrained_ad(self, ad_checkpoint_path):
        """
        Load pre-trained AD model weights into the RAD model.
        
        This initializes the AD transformer, embeddings, and action prediction head
        from a trained AD model. The compression transformer is left randomly initialized
        and will be trained during finetuning.
        
        Args:
            ad_checkpoint_path: Path to the trained AD checkpoint
        """
        checkpoint = torch.load(ad_checkpoint_path, map_location='cpu')
        ad_state = checkpoint['model']
        
        # Map AD keys to RAD keys
        # AD uses 'transformer', RAD uses 'ad_transformer'
        key_mapping = {
            'transformer.': 'ad_transformer.',
            'embed_context.': 'embed_context.',
            'embed_query_state.': 'embed_query_state.',
            'pred_action.': 'pred_action.',
        }
        
        # Load AD transformer
        ad_transformer_state = {}
        for k, v in ad_state.items():
            if k.startswith('transformer.'):
                new_key = k.replace('transformer.', '')
                ad_transformer_state[new_key] = v
        
        if ad_transformer_state:
            self.ad_transformer.load_state_dict(ad_transformer_state)
            print(f"  Loaded AD transformer ({len(ad_transformer_state)} params)")
        
        # Load embeddings
        if 'embed_context.weight' in ad_state:
            self.embed_context.load_state_dict({
                'weight': ad_state['embed_context.weight'],
                'bias': ad_state['embed_context.bias']
            })
            print("  Loaded embed_context")
        
        if 'embed_query_state.weight' in ad_state:
            self.embed_query_state.load_state_dict({
                'weight': ad_state['embed_query_state.weight']
            })
            print("  Loaded embed_query_state")
        
        # Load action prediction head
        if 'pred_action.weight' in ad_state:
            self.pred_action.load_state_dict({
                'weight': ad_state['pred_action.weight'],
                'bias': ad_state['pred_action.bias']
            })
            print("  Loaded pred_action")
        
        print(f"Loaded pre-trained AD from {ad_checkpoint_path}")
        print("  Note: Compression transformer is randomly initialized")
