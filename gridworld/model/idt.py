"""Hierarchical IDT with GPT-2 backbones; protocol from dicp/gridworld.

Review: summed transition embeddings -> Gaussian decision parameters.
High level: (return-to-go, state, reviewed decision) -> next decision.
Low level: (sampled decision, state, action, reward) -> action at state.
"""

import numpy as np
import torch
from torch import nn

from env import map_dark_states
from .baseline_common import (
    choose_action, episode_array, make_transformer, run_transformer, terminal_states, validate_tokenization,
)


class ReviewingDecisions(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['tf_n_embd']
        self.transformer = make_transformer(config, config['low_per_high'])
        self.embed_state = nn.Embedding(config['grid_size'] ** 2, width)
        self.embed_next_state = nn.Embedding(config['grid_size'] ** 2, width)
        self.embed_action = nn.Embedding(config['num_actions'], width)
        self.embed_reward = nn.Embedding(2, width)
        self.predict_z = nn.Linear(width, 2 * config['dim_z'])

    def forward(self, states, actions, rewards, next_states):
        tokens = (self.embed_state(states) + self.embed_action(actions)
                  + self.embed_reward(rewards) + self.embed_next_state(next_states))
        batch, high, low, width = tokens.shape
        output = run_transformer(self.transformer, tokens.reshape(batch * high, low, width))
        return self.predict_z(output[:, -1]).reshape(batch, high, -1)


class HDecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['tf_n_embd']
        self.dim_z = config['dim_z']
        self.transformer = make_transformer(config, 3 * (config['n_transit'] // config['low_per_high']))
        self.embed_return = nn.Linear(1, width)
        self.embed_state = nn.Embedding(config['grid_size'] ** 2, width)
        self.embed_z = nn.Linear(2 * self.dim_z, width)
        self.pred_z_dist = nn.Linear(width, 2 * self.dim_z)

    def forward(self, return_to_go, states, z_dists=None):
        if z_dists is None:
            z_dists = torch.zeros(*states.shape, 2 * self.dim_z, device=states.device)
        elif z_dists.shape[1] == states.shape[1] - 1:
            z_dists = torch.nn.functional.pad(z_dists, (0, 0, 0, 1))
        tokens = torch.stack((self.embed_return(return_to_go.float().unsqueeze(-1)),
                              self.embed_state(states), self.embed_z(z_dists)), 2).flatten(1, 2)
        # The state output cannot see its own reviewed decision (the next token).
        return self.pred_z_dist(run_transformer(self.transformer, tokens)[:, 1::3])


class DecisionsToGo(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['tf_n_embd']
        self.transformer = make_transformer(config, 4 * config['low_per_high'])
        self.embed_z = nn.Linear(config['dim_z'], width)
        self.embed_state = nn.Embedding(config['grid_size'] ** 2, width)
        self.embed_action = nn.Embedding(config['num_actions'], width)
        self.embed_reward = nn.Embedding(2, width)
        self.pred_action = nn.Linear(width, config['num_actions'])
        self.dynamics = config.get('dynamics', False)
        if self.dynamics:
            self.pred_next_state = nn.Linear(width, config['grid_size'] ** 2)
            self.pred_reward = nn.Linear(width, 2)

    def forward(self, z, states, actions, rewards):
        tokens = torch.stack((self.embed_z(z), self.embed_state(states),
                              self.embed_action(actions), self.embed_reward(rewards)), -2)
        batch, high, low, _, width = tokens.shape
        output = run_transformer(self.transformer, tokens.reshape(batch * high, 4 * low, width))
        output = output.reshape(batch, high, low, 4, width)
        result = {'action': self.pred_action(output[:, :, :, 1])}
        if self.dynamics:
            result.update(next_state=self.pred_next_state(output[:, :, :, 2]),
                          reward=self.pred_reward(output[:, :, :, 2]))
        return result


class IDT(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_tokenization(config, 'IDT')
        self.config = config
        self.low_per_high = int(config['low_per_high'])
        self.n_transit = int(config['n_transit'])
        if self.low_per_high < 1 or self.n_transit < self.low_per_high or self.n_transit % self.low_per_high or config['horizon'] % self.low_per_high:
            raise ValueError('IDT requires positive low_per_high dividing n_transit and horizon')
        self.reviewing_decisions = ReviewingDecisions(config)
        self.h_decision_transformer = HDecisionTransformer(config)
        self.decisions_to_go = DecisionsToGo(config)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

    @property
    def device(self):
        return self.decisions_to_go.embed_state.weight.device

    def get_gaussian_sample(self, parameters, sample=True):
        mean, logvar = parameters.chunk(2, -1)
        mean = mean.unsqueeze(-2).expand(*mean.shape[:-1], self.low_per_high, mean.shape[-1])
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp().unsqueeze(-2)

    def forward(self, x):
        values = {k: v.to(self.device) for k, v in x.items()}
        for key in ('states', 'next_states'):
            values[key] = map_dark_states(values[key].long(), self.config['grid_size'])
        for key in ('actions', 'rewards'):
            values[key] = values[key].long()
        values = {k: v.reshape(v.shape[0], -1, self.low_per_high) for k, v in values.items()}
        states, actions, rewards, next_states = (values[k] for k in ('states', 'actions', 'rewards', 'next_states'))
        reviewed = self.reviewing_decisions(states, actions, rewards, next_states)
        predicted = self.h_decision_transformer(values['return_to_go'][:, :, 0], states[:, :, 0], reviewed)
        logits = self.decisions_to_go(self.get_gaussian_sample(predicted), states, actions, rewards)
        result = {}
        for key, target in (('action', actions), ('reward', rewards), ('next_state', next_states)):
            if key in logits:
                result[f'loss_{key}'] = self.loss_fn(logits[key].flatten(0, 2), target.flatten())
                result[f'acc_{key}'] = (logits[key].argmax(-1) == target).float().mean()
        return result

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        if beam_k:
            raise ValueError('IDT baseline evaluation supports policy sampling/argmax, not beam search')
        low = self.decisions_to_go
        max_high = self.n_transit // self.low_per_high
        observations = vec_env.reset()
        totals = np.zeros(vec_env.num_envs)
        episodes = [[] for _ in totals]
        rtg = torch.as_tensor(vec_env.env_method('get_max_return'), device=self.device, dtype=torch.float32)
        high_states, high_returns, reviews = [], [], []
        history = []
        for step in range(eval_timesteps):
            states = map_dark_states(torch.as_tensor(observations, device=self.device).long(), self.config['grid_size'])
            offset = step % self.low_per_high
            if offset == 0:
                high_states.append(states)
                high_returns.append(rtg.clone())
                high_states = high_states[-max_high:]
                high_returns = high_returns[-max_high:]
                reviews = reviews[-(max_high - 1):] if max_high > 1 else []
                predicted = self.h_decision_transformer(
                    torch.stack(high_returns, 1), torch.stack(high_states, 1),
                    torch.stack(reviews, 1) if reviews else None,
                )[:, -1]
                z = self.get_gaussian_sample(predicted, sample=sample)
                tokens = torch.empty(len(totals), 0, self.config['tf_n_embd'], device=self.device)
                history = []
            tokens = torch.cat((tokens, low.embed_z(z[:, offset]).unsqueeze(1), low.embed_state(states).unsqueeze(1)), 1)
            actions = choose_action(low.pred_action(run_transformer(low.transformer, tokens)[:, -1]), sample)
            observations, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            terminal = map_dark_states(torch.as_tensor(terminal_states(observations, dones, infos), device=self.device).long(), self.config['grid_size'])
            reward_ids = torch.as_tensor(rewards, device=self.device).long()
            tokens = torch.cat((tokens, low.embed_action(actions).unsqueeze(1), low.embed_reward(reward_ids).unsqueeze(1)), 1)
            history.append((states, actions, reward_ids, terminal))
            if offset == self.low_per_high - 1:
                review_inputs = [torch.stack([item[i] for item in history], 1).unsqueeze(1) for i in range(4)]
                reviews.append(self.reviewing_decisions(*review_inputs)[:, 0])
            rtg -= torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
            totals += rewards
            for i, done in enumerate(dones):
                if done:
                    episodes[i].append(totals[i])
                    totals[i] = 0
                    rtg[i] = float(vec_env.env_method('get_max_return', indices=i)[0])
        return {'reward_episode': episode_array(episodes)}
