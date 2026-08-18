import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .gpt2 import GPT2Transformer


class AD(torch.nn.Module):
    """Reward-aware causal Algorithm Distillation for continuous control.

    Each Meta-World timestep is represented with the same three-token contract
    used by gridworld_test: state, action, reward.  Action predictions are read
    from state-token outputs so the current action/reward cannot leak into the
    prediction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_transit = config['n_transit']
        self.max_seq_length = 3 * self.n_transit
        self.obs_dim = config['dim_obs']
        self.action_dim = config['dim_actions']
        self.learn_var = config.get('learn_var', False)

        tf_n_embd = config['tf_n_embd']
        self.transformer = GPT2Transformer(
            d_model=tf_n_embd,
            n_heads=config.get('tf_n_head', 4),
            n_layers=config.get('tf_n_layer', 4),
            max_seq_length=self.max_seq_length,
            dim_feedforward=config.get('tf_dim_feedforward', tf_n_embd * 4),
            dropout=config.get('tf_dropout', 0.1),
        )
        if config.get('gradient_checkpointing', False):
            self.transformer.enable_gradient_checkpointing()

        self.embed_state = nn.Linear(self.obs_dim, tf_n_embd)
        self.embed_action = nn.Linear(self.action_dim, tf_n_embd)
        self.embed_reward = nn.Linear(1, tf_n_embd)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, tf_n_embd))
        self.pred_action = nn.Linear(
            tf_n_embd,
            2 * self.action_dim if self.learn_var else self.action_dim,
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

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

    def _build_token_sequence(self, states, actions, rewards):
        tokens = torch.stack(
            [
                self._embed_state(states),
                self._embed_action(actions),
                self._embed_reward(rewards),
            ],
            dim=2,
        )
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

    def _action_loss(self, prediction, target):
        if not self.learn_var:
            return self.loss_fn(prediction, target)
        mean, log_var = prediction.split(self.action_dim, dim=-1)
        variance = torch.exp(log_var.clamp(min=-10.0, max=10.0))
        return self.loss_fn_gaussian(mean, target, variance)

    def _sample_action(self, prediction, sample):
        if self.learn_var:
            mean, log_var = prediction.split(self.action_dim, dim=-1)
            if sample:
                action = mean + torch.exp(0.5 * log_var.clamp(-10.0, 10.0)) * torch.randn_like(mean)
            else:
                action = mean
        elif sample:
            action = prediction + torch.randn_like(prediction)
        else:
            action = prediction

        if hasattr(self, 'action_low') and hasattr(self, 'action_high'):
            action = torch.maximum(torch.minimum(action, self.action_high), self.action_low)
        return action

    def forward(self, x):
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)

        tokens = self._build_token_sequence(states, actions, rewards)
        transformer = self._module_for_current_grad_mode(self.transformer)
        transformer_output = transformer(tokens, use_causal_mask=True)
        predicted_actions = self.pred_action(transformer_output[:, 0::3])
        loss_action = self._action_loss(predicted_actions, actions)

        return {
            'loss_action': loss_action,
            'acc_action': torch.zeros((), device=self.device),
        }

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {'reward_episode': [], 'success': []}
        reward_episode = np.zeros(vec_env.num_envs)
        success_episode = np.zeros(vec_env.num_envs, dtype=np.bool_)

        query_states = vec_env.reset()[..., :self.obs_dim]
        query_states = torch.as_tensor(query_states, device=self.device, dtype=torch.float)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        transformer_input = self._typed_state_token(query_states)

        for _ in range(eval_timesteps):
            transformer = self._module_for_current_grad_mode(self.transformer)
            output = transformer(transformer_input, use_causal_mask=True)
            actions = self._sample_action(self.pred_action(output[:, -1]), sample=sample)

            next_observations, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            action_tokens = rearrange(actions, 'e d -> e 1 d')
            reward_episode += rewards
            success_episode |= np.asarray(
                [bool(info.get('success', False)) for info in infos],
                dtype=np.bool_,
            )
            reward_tokens = torch.as_tensor(rewards, device=self.device, dtype=torch.float)
            reward_tokens = rearrange(reward_tokens, 'e -> e 1 1')

            reset_states = torch.as_tensor(
                next_observations[..., :self.obs_dim],
                device=self.device,
                dtype=torch.float,
            )
            reset_states = rearrange(reset_states, 'e d -> e 1 d')

            if dones[0]:
                outputs['reward_episode'].append(reward_episode.copy())
                outputs['success'].append(success_episode.copy())
                reward_episode.fill(0)
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
                next_states = reset_states

            transformer_input = torch.cat(
                [
                    transformer_input,
                    self._typed_action_token(action_tokens),
                    self._typed_reward_token(reward_tokens),
                    self._typed_state_token(next_states),
                ],
                dim=1,
            )
            transformer_input = transformer_input[:, -(self.max_seq_length - 2):]

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.stack(outputs['success'], axis=1)
        return outputs

    def set_obs_space(self, obs_space):
        # Observations are sliced to the task-dependent dimensions in the model.
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
