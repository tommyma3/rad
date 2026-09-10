"""Decision Pretrained Transformer, following dicp/gridworld's DPT protocol.

One padded query-state token precedes packed (s, a, r, next_s) tokens.
Every nonempty causal context prefix predicts the same optimal query action.
The backbone is this repository's GPT-2, not the reference's TinyLlama.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env import map_dark_states
from .baseline_common import (
    choose_action, episode_array, make_transformer, run_transformer, terminal_states, validate_tokenization,
)


class DPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_tokenization(config, 'DPT')
        self.config = config
        self.n_transit = int(config['n_transit'])
        if self.n_transit < 2:
            raise ValueError('DPT n_transit must include a query and at least one transition')
        self.dynamics = config.get('dynamics', False)
        width = config['tf_n_embd']
        self.transformer = make_transformer(config, self.n_transit + int(self.dynamics))
        self.embed_context = nn.Linear(2 * config['dim_states'] + config['num_actions'] + 1, width)
        self.pred_actions = nn.Linear(width, config['num_actions'])
        if self.dynamics:
            self.embed_query_action = nn.Embedding(config['num_actions'], width)
            self.pred_rewards = nn.Linear(width, 2)
            self.pred_next_states = nn.Linear(width, config['grid_size'] ** 2)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

    @property
    def device(self):
        return self.embed_context.weight.device

    def build_tokens(self, query_states, states, actions, rewards, next_states):
        if actions.ndim == states.ndim - 1:
            actions = F.one_hot(actions.long(), self.config['num_actions'])
        packed = torch.cat((states.float(), actions.float(), rewards.float().unsqueeze(-1), next_states.float()), -1)
        query = F.pad(query_states.float(), (0, packed.shape[-1] - query_states.shape[-1]))
        return self.embed_context(torch.cat((query.unsqueeze(1), packed), 1))

    def forward(self, x):
        x = {key: value.to(self.device) for key, value in x.items()}
        tokens = self.build_tokens(*(x[key] for key in ('query_states', 'states', 'actions', 'rewards', 'next_states')))
        if tokens.shape[1] != self.n_transit:
            raise ValueError('DPT training expects n_transit - 1 context transitions')
        if self.dynamics:
            tokens = torch.cat((tokens, self.embed_query_action(x['query_actions'].long()).unsqueeze(1)), 1)
        output = run_transformer(self.transformer, tokens)
        logits = self.pred_actions(output[:, 1:self.n_transit])
        targets = x['target_actions'].long()[:, None].expand(logits.shape[:2])
        result = {
            'loss_action': self.loss_fn(logits.flatten(0, 1), targets.flatten()),
            'acc_action': (logits.argmax(-1) == targets).float().mean(),
        }
        if self.dynamics:
            for name, head, target in (
                ('reward', self.pred_rewards, x['target_rewards'].long()),
                ('next_state', self.pred_next_states, map_dark_states(x['target_next_states'].long(), self.config['grid_size'])),
            ):
                prediction = head(output[:, -1])
                result[f'loss_{name}'] = self.loss_fn(prediction, target)
                result[f'acc_{name}'] = (prediction.argmax(-1) == target).float().mean()
        return result

    @torch.inference_mode()
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        if beam_k:
            raise ValueError('DPT baseline evaluation supports policy sampling/argmax, not beam search')
        observations = vec_env.reset()
        context = torch.empty(vec_env.num_envs, 0, self.config['tf_n_embd'], device=self.device)
        totals = np.zeros(vec_env.num_envs)
        episodes = [[] for _ in totals]
        for _ in range(eval_timesteps):
            states = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
            query = F.pad(states, (0, self.embed_context.in_features - states.shape[-1]))
            tokens = torch.cat((self.embed_context(query).unsqueeze(1), context), 1)
            actions = choose_action(self.pred_actions(run_transformer(self.transformer, tokens)[:, -1]), sample)
            observations, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            next_states = torch.as_tensor(terminal_states(observations, dones, infos), device=self.device, dtype=torch.float32)
            reward_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
            transition = torch.cat((states, F.one_hot(actions, self.config['num_actions']).float(), reward_tensor[:, None], next_states), -1)
            context = torch.cat((context, self.embed_context(transition).unsqueeze(1)), 1)[:, -(self.n_transit - 1):]
            totals += rewards
            for i, done in enumerate(dones):
                if done:
                    episodes[i].append(totals[i])
                    totals[i] = 0
        return {'reward_episode': episode_array(episodes)}
