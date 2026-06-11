import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange

from .gpt2 import GPT2Transformer

class AD(torch.nn.Module):
    """
    Algorithm Distillation with GPT-2 style decoder-only transformer.
    
    Adapted for Meta-world continuous action spaces.
    Uses Pre-LayerNorm architecture as in the original GPT-2 paper.
    """
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self._device = config['device']  # Initial device, may change with accelerator
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        
        # Observation dimension handling
        if 'task' in config and (config['task'] == "hammer-v2" or config['task'] == "stick-push-v2" or config['task'] == "stick-pull-v2"):
            assert config['dim_obs'] == 18
            self.obs_dim_idx = list(range(18))
        else:
            assert config['dim_obs'] == 11
            self.obs_dim_idx = list(range(11))

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        tf_dropout = config.get('tf_dropout', 0.1)

        # GPT-2 style transformer (Pre-LayerNorm, causal)
        self.transformer = GPT2Transformer(
            d_model=tf_n_embd,
            n_heads=tf_n_head,
            n_layers=tf_n_layer,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
        )

        # Input embeddings
        self.embed_context = nn.Linear(config['dim_obs'] * 2 + config['dim_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Linear(config['dim_obs'], tf_n_embd)
        
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # Output head for continuous actions
        if config.get('learn_var', False):
            self.pred_actions = nn.Linear(tf_n_embd, 2 * config['dim_actions'])
            self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')
        else:
            self.pred_actions = nn.Linear(tf_n_embd, config['dim_actions'])

    @property
    def device(self):
        """Get the actual device of the model (works with accelerator)."""
        return next(self.parameters()).device

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_obs)
        target_actions = x['target_actions'].to(self.device)  # (batch_size, dim_actions)
        states = x['states'].to(self.device)  # (batch_size, n_transit-1, dim_obs)
        actions = x['actions'].to(self.device)  # (batch_size, n_transit-1, dim_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, n_transit-1, dim_obs)
        rewards = x['rewards'].to(self.device)  # (batch_size, n_transit-1)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(query_states)
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
        
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')

        # GPT-2 style transformer with causal masking
        transformer_output = self.transformer(context_embed, use_causal_mask=True)

        result = {}
        
        if self.config.get('learn_var', False):
            dist = self.pred_actions(transformer_output[:, self.n_transit-1])
            mean = dist[:, :self.config['dim_actions']]
            var = torch.exp(dist[:, self.config['dim_actions']:])
            result['loss_action'] = self.loss_fn_gaussian(mean, target_actions, var)
        else:
            predicted_actions = self.pred_actions(transformer_output[:, self.n_transit-1])
            result['loss_action'] = self.loss_fn(predicted_actions, target_actions)

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, sample_size=1, beam_start=50, sample=True):
        outputs = {}
        outputs['reward_episode'] = []
        outputs['success'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        success = np.zeros(vec_env.num_envs)
        
        # Get initial states embeddings
        query_states = vec_env.reset()[:, self.obs_dim_idx]  # (n_envs, obs_dim)
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.float)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(query_states)
        transformer_input = query_states_embed
        
        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            output = self.transformer(transformer_input, use_causal_mask=True)
            
            if self.config.get('learn_var', False):
                dist = self.pred_actions(output[:, -1])
                mean = dist[:, :self.config['dim_actions']]
                std = torch.exp(dist[:, self.config['dim_actions']:] / 2)
                actions = (std * torch.randn_like(mean) + mean)
            elif sample:
                mean = self.pred_actions(output[:, -1])
                std = torch.ones_like(mean)
                actions = (std * torch.randn_like(mean) + mean)
            else:
                actions = self.pred_actions(output[:, -1])
                        
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e d -> e 1 d')

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

            query_states = torch.tensor(query_states[:, self.obs_dim_idx], device=self.device, requires_grad=False, dtype=torch.float)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            success += np.array([info['success'] for info in infos])
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                outputs['success'].append(success > 0.0)
                success = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(np.stack([info['terminal_observation'][self.obs_dim_idx] for info in infos]), device=self.device, dtype=torch.float)
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach()
            
            query_states_embed = self.embed_query_state(query_states)

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)
            
            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]
                
            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')
            
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.maximum.accumulate(np.stack(outputs['success'], axis=1), axis=-1)

        return outputs
    
    def set_obs_space(self, obs_space):
        # Use register_buffer so tensors move with the model
        self.register_buffer('obs_low', torch.tensor(obs_space.low[:self.config['dim_obs']], requires_grad=False, dtype=torch.float))
        self.register_buffer('obs_high', torch.tensor(obs_space.high[:self.config['dim_obs']], requires_grad=False, dtype=torch.float))
    
    def set_action_space(self, action_space):
        # Use register_buffer so tensors move with the model
        self.register_buffer('action_low', torch.tensor(action_space.low, requires_grad=False, dtype=torch.float))
        self.register_buffer('action_high', torch.tensor(action_space.high, requires_grad=False, dtype=torch.float))
