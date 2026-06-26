import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from env import map_dark_states
from .gpt2 import GPT2Transformer


class AD(torch.nn.Module):
    """
    Reward-aware SAD-style Algorithm Distillation.

    Each environment timestep is represented as three causal tokens:
        s_t, a_t, r_t

    The model predicts a_t from the output at the s_t token position. Because
    causal masking is used, the state token can attend to previous history and
    the current state, but not the current action or reward.
    """

    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = 3 * self.n_transit
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        self.num_actions = config['num_actions']

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        tf_dropout = config.get('tf_dropout', 0.1)

        self.transformer = GPT2Transformer(
            d_model=tf_n_embd,
            n_heads=tf_n_head,
            n_layers=tf_n_layer,
            max_seq_length=self.max_seq_length,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
        )
        if config.get('gradient_checkpointing', False):
            self.transformer.enable_gradient_checkpointing()

        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        self.embed_action = nn.Linear(self.num_actions, tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, tf_n_embd))

        self.pred_action = nn.Linear(tf_n_embd, self.num_actions)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        nn.init.trunc_normal_(self.type_embedding, std=0.02)

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

    def _build_token_sequence(self, states, actions, rewards):
        state_tokens = self._embed_state(states)
        action_tokens = self._embed_action(actions)
        reward_tokens = self._embed_reward(rewards)

        tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        tokens = tokens + self.type_embedding
        return rearrange(tokens, 'b t m d -> b (t m) d')

    def _module_for_current_grad_mode(self, module):
        if torch.is_grad_enabled():
            return module
        return getattr(module, '_orig_mod', module)

    def _typed_state_token(self, states):
        return self._embed_state(states) + self.type_embedding[:, :, 0, :]

    def _typed_action_token(self, actions):
        return self._embed_action(actions) + self.type_embedding[:, :, 1, :]

    def _typed_reward_token(self, rewards):
        return self._embed_reward(rewards) + self.type_embedding[:, :, 2, :]

    def forward(self, x):
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)

        tokens = self._build_token_sequence(states, actions, rewards)
        transformer = self._module_for_current_grad_mode(self.transformer)
        transformer_output = transformer(tokens, use_causal_mask=True)

        state_outputs = transformer_output[:, 0::3]
        logits_actions = self.pred_action(state_outputs)
        target_actions = actions.argmax(dim=-1)

        loss_action = self.loss_fn(
            rearrange(logits_actions, 'b t a -> (b t) a'),
            rearrange(target_actions, 'b t -> (b t)'),
        )
        acc_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        return {
            'loss_action': loss_action,
            'acc_action': acc_action,
        }

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {'reward_episode': []}
        reward_episode = np.zeros(vec_env.num_envs)

        query_states = vec_env.reset()
        query_states = torch.as_tensor(query_states, device=self.device, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        transformer_input = self._typed_state_token(query_states)

        for _ in range(eval_timesteps):
            transformer = self._module_for_current_grad_mode(self.transformer)
            output = transformer(transformer_input, use_causal_mask=True)
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

            new_tokens = [
                self._typed_action_token(actions_onehot),
                self._typed_reward_token(rewards_tensor),
                self._typed_state_token(next_states),
            ]
            transformer_input = torch.cat([transformer_input, *new_tokens], dim=1)
            max_eval_length = self.max_seq_length - 2
            transformer_input = transformer_input[:, -max_eval_length:]

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        return outputs
