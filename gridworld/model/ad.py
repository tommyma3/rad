import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse
from .gpt2 import GPT2Transformer


class AD(torch.nn.Module):
    """
    Algorithm Distillation with GPT-2 style decoder-only transformer.
    
    Uses Pre-LayerNorm architecture as in the original GPT-2 paper,
    which provides better training stability.
    """
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

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
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        
        # Output head
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')

        # GPT-2 style transformer with causal masking
        transformer_output = self.transformer(context_embed, use_causal_mask=True)

        result = {}

        # The last token in the sequence is the query state embedding
        # Predict action from this position
        logits_actions = self.pred_action(transformer_output[:, -1])  # (batch_size, num_actions)

        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)

        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            # GPT-2 style transformer with causal masking
            output = self.transformer(transformer_input, use_causal_mask=True)

            logits = self.pred_action(output[:, -1])

            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)

            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e -> e 1 1')
            actions = F.one_hot(actions, num_classes=self.config['num_actions'])

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

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

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)

            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]

            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        return outputs
