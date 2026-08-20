"""PyTorch DreamerV2 source learner with sequential TBTT replay.

The implementation follows the Memory Maze paper's defining TBTT contract:
each batch slot replays one episode sequentially, carries its RSSM posterior to
the next chunk, detaches gradients at chunk boundaries, and resets state only
when that episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(self.deterministic.detach(), self.stochastic.detach())

    @property
    def features(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self.stochastic.flatten(-2)], dim=-1)


class ConvEncoder(nn.Module):
    def __init__(self, depth: int = 32, output_dim: int = 1024) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, depth, 4, 2), nn.ELU(),
            nn.Conv2d(depth, 2 * depth, 4, 2), nn.ELU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, 2), nn.ELU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, 2), nn.ELU(),
        )
        self.projection = nn.Linear(8 * depth * 2 * 2, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] == 3:
            images = images.movedim(-1, -3)
        leading = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:]).float().div(255.0).sub(0.5)
        output = self.projection(self.network(flat).flatten(1))
        return output.reshape(*leading, output.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, depth: int = 32) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, 8 * depth * 4 * 4)
        self.depth = depth
        self.network = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(2 * depth, depth, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(depth, 3, 4, 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        leading = features.shape[:-1]
        hidden = self.project(features.reshape(-1, features.shape[-1]))
        hidden = hidden.reshape(-1, 8 * self.depth, 4, 4)
        images = torch.sigmoid(self.network(hidden))
        return images.reshape(*leading, *images.shape[-3:])


class DiscreteRSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int,
        stochastic_dim: int,
        classes: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stochastic_dim = stochastic_dim
        self.classes = classes
        stochastic_flat = stochastic_dim * classes
        self.input = nn.Sequential(
            nn.Linear(stochastic_flat + action_dim, hidden_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stochastic_flat),
        )
        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stochastic_flat),
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deterministic = torch.zeros(batch_size, self.deter_dim, device=device)
        stochastic = torch.zeros(
            batch_size, self.stochastic_dim, self.classes, device=device
        )
        stochastic[..., 0] = 1.0
        return RSSMState(deterministic, stochastic)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        distribution = torch.distributions.OneHotCategorical(logits=logits)
        sample = distribution.sample()
        probabilities = distribution.probs
        return sample + probabilities - probabilities.detach()

    def imagine_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
    ) -> tuple[RSSMState, torch.Tensor]:
        hidden = self.input(torch.cat([previous.stochastic.flatten(-2), action], dim=-1))
        deterministic = self.gru(hidden, previous.deterministic)
        prior_logits = self.prior(deterministic).reshape(
            -1, self.stochastic_dim, self.classes
        )
        return RSSMState(deterministic, self._sample(prior_logits)), prior_logits

    def observe_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        prior_state, prior_logits = self.imagine_step(previous, action)
        posterior_logits = self.posterior(
            torch.cat([prior_state.deterministic, embedding], dim=-1)
        ).reshape(-1, self.stochastic_dim, self.classes)
        posterior = RSSMState(prior_state.deterministic, self._sample(posterior_logits))
        return posterior, prior_logits, posterior_logits


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int = 4) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(current, hidden_dim), nn.ELU()])
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


class DreamerModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        action_dim = int(config.get("num_actions", 6))
        embed_dim = int(config.get("encoder_dim", 1024))
        deter_dim = int(config.get("rssm_deter", 2048))
        stochastic_dim = int(config.get("rssm_stoch", 32))
        classes = int(config.get("rssm_classes", 32))
        hidden_dim = int(config.get("rssm_hidden", 1000))
        feature_dim = deter_dim + stochastic_dim * classes
        self.action_dim = action_dim
        self.encoder = ConvEncoder(int(config.get("cnn_depth", 32)), embed_dim)
        self.rssm = DiscreteRSSM(
            action_dim, embed_dim, deter_dim, stochastic_dim, classes, hidden_dim
        )
        self.decoder = ConvDecoder(feature_dim, int(config.get("cnn_depth", 32)))
        self.reward_head = _mlp(feature_dim, 1, 400)
        self.continue_head = _mlp(feature_dim, 1, 400)
        self.actor = _mlp(feature_dim, action_dim, 400)
        self.critic = _mlp(feature_dim, 1, 400)
        self.slow_critic = _mlp(feature_dim, 1, 400)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        self.slow_critic.requires_grad_(False)

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        return self.rssm.initial(batch_size, device)


@dataclass
class ReplayEpisode:
    images: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass
class _TBTTSlot:
    episode_index: int
    cursor: int
    state: RSSMState | None


class SequentialTBTTReplay:
    """Sequential replay with distinct episodes assigned to batch slots."""

    def __init__(self, capacity_steps: int, sequence_length: int, batch_size: int) -> None:
        self.capacity_steps = int(capacity_steps)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.episodes: list[ReplayEpisode] = []
        self.total_steps = 0
        self.slots: list[_TBTTSlot] = []

    def add(self, episode: ReplayEpisode) -> None:
        self.episodes.append(episode)
        self.total_steps += episode.length
        while self.total_steps > self.capacity_steps and len(self.episodes) > self.batch_size:
            removed = self.episodes.pop(0)
            self.total_steps -= removed.length
            self.slots.clear()

    def ready(self) -> bool:
        return len(self.episodes) >= self.batch_size

    def _initialize_slots(self) -> None:
        indices = random.sample(range(len(self.episodes)), self.batch_size)
        first_length = self.episodes[indices[0]].length
        first_offset = random.randrange(max(1, first_length - self.sequence_length + 1))
        self.slots = [
            _TBTTSlot(index, first_offset if slot == 0 else 0, None)
            for slot, index in enumerate(indices)
        ]

    def next_batch(self) -> tuple[dict[str, torch.Tensor], list[RSSMState | None]]:
        if not self.ready():
            raise RuntimeError("Replay does not contain enough distinct episodes")
        if len(self.slots) != self.batch_size:
            self._initialize_slots()
        selected: list[ReplayEpisode] = []
        starts: list[int] = []
        initial_states: list[RSSMState | None] = []
        occupied = {slot.episode_index for slot in self.slots}
        for slot_index, slot in enumerate(self.slots):
            episode = self.episodes[slot.episode_index]
            if slot.cursor + self.sequence_length > episode.length:
                choices = [i for i in range(len(self.episodes)) if i not in occupied]
                if not choices:
                    choices = list(range(len(self.episodes)))
                occupied.discard(slot.episode_index)
                slot.episode_index = random.choice(choices)
                occupied.add(slot.episode_index)
                slot.cursor = 0
                slot.state = None
                episode = self.episodes[slot.episode_index]
            selected.append(episode)
            starts.append(slot.cursor)
            initial_states.append(slot.state)
        length = min(
            self.sequence_length,
            min(ep.length - start for ep, start in zip(selected, starts)),
        )
        images = np.stack([ep.images[start : start + length] for ep, start in zip(selected, starts)])
        actions = np.stack([ep.actions[start : start + length] for ep, start in zip(selected, starts)])
        rewards = np.stack([ep.rewards[start : start + length] for ep, start in zip(selected, starts)])
        dones = np.stack([ep.dones[start : start + length] for ep, start in zip(selected, starts)])
        for slot in self.slots:
            slot.cursor += length
        batch = {
            "images": torch.from_numpy(images),
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).float(),
        }
        return batch, initial_states

    def update_states(self, states: RSSMState) -> None:
        for index, slot in enumerate(self.slots):
            slot.state = RSSMState(
                states.deterministic[index : index + 1].detach(),
                states.stochastic[index : index + 1].detach(),
            )


class DreamerTBTT:
    def __init__(self, config: dict, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = DreamerModel(config).to(self.device)
        model_parameters = (
            list(self.model.encoder.parameters())
            + list(self.model.rssm.parameters())
            + list(self.model.decoder.parameters())
            + list(self.model.reward_head.parameters())
            + list(self.model.continue_head.parameters())
        )
        self.model_optimizer = torch.optim.AdamW(
            model_parameters,
            lr=float(config.get("world_model_lr", 3e-4)),
            eps=float(config.get("adam_epsilon", 1e-5)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=float(config.get("actor_lr", 1e-4)),
            eps=float(config.get("adam_epsilon", 1e-5)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=float(config.get("critic_lr", 1e-4)),
            eps=float(config.get("adam_epsilon", 1e-5)),
            weight_decay=float(config.get("weight_decay", 1e-2)),
        )
        self.replay = SequentialTBTTReplay(
            int(config.get("replay_capacity", 10_000_000)),
            int(config.get("sequence_length", 48)),
            int(config.get("batch_size", 32)),
        )
        self.updates = 0

    def _stack_initial_states(
        self,
        states: list[RSSMState | None],
        batch_size: int,
    ) -> RSSMState:
        fallback = self.model.initial(1, self.device)
        deterministic = torch.cat(
            [(state or fallback).deterministic.to(self.device) for state in states], dim=0
        )
        stochastic = torch.cat(
            [(state or fallback).stochastic.to(self.device) for state in states], dim=0
        )
        if deterministic.shape[0] != batch_size:
            raise RuntimeError("TBTT state batch does not match replay batch")
        return RSSMState(deterministic, stochastic)

    def _kl_loss(self, posterior: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        balance = float(self.config.get("kl_balance", 0.8))
        posterior_dist = torch.distributions.Categorical(logits=posterior)
        prior_dist = torch.distributions.Categorical(logits=prior)
        left = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=posterior.detach()), prior_dist
        )
        right = torch.distributions.kl_divergence(
            posterior_dist, torch.distributions.Categorical(logits=prior.detach())
        )
        return (balance * left + (1.0 - balance) * right).mean()

    def _observe(
        self,
        batch: dict[str, torch.Tensor],
        initial: RSSMState,
    ) -> tuple[list[RSSMState], torch.Tensor, torch.Tensor]:
        images = batch["images"].to(self.device)
        actions = F.one_hot(
            batch["actions"].to(self.device), self.model.action_dim
        ).float()
        embeddings = self.model.encoder(images)
        state = initial
        states: list[RSSMState] = []
        priors: list[torch.Tensor] = []
        posteriors: list[torch.Tensor] = []
        for step in range(images.shape[1]):
            state, prior, posterior = self.model.rssm.observe_step(
                state, actions[:, step], embeddings[:, step]
            )
            states.append(state)
            priors.append(prior)
            posteriors.append(posterior)
        return states, torch.stack(priors, 1), torch.stack(posteriors, 1)

    def _world_model_loss(
        self,
        batch: dict[str, torch.Tensor],
        states: list[RSSMState],
        priors: torch.Tensor,
        posteriors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = torch.stack([state.features for state in states], dim=1)
        reconstructed = self.model.decoder(features)
        targets = batch["images"].to(self.device)
        if targets.shape[-1] == 3:
            targets = targets.movedim(-1, -3)
        targets = targets.float().div(255.0)
        image_loss = F.mse_loss(reconstructed, targets)
        reward_loss = F.mse_loss(
            self.model.reward_head(features).squeeze(-1),
            batch["rewards"].to(self.device),
        )
        continuation_target = 1.0 - batch["dones"].to(self.device)
        continue_loss = F.binary_cross_entropy_with_logits(
            self.model.continue_head(features).squeeze(-1), continuation_target
        )
        kl_loss = self._kl_loss(posteriors, priors)
        total = image_loss + reward_loss + continue_loss + float(
            self.config.get("kl_scale", 1.0)
        ) * kl_loss
        return total, {
            "loss_model": total.detach(),
            "loss_image": image_loss.detach(),
            "loss_reward": reward_loss.detach(),
            "loss_continue": continue_loss.detach(),
            "loss_kl": kl_loss.detach(),
        }

    def _lambda_returns(
        self,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        values: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> torch.Tensor:
        lambda_ = float(self.config.get("lambda", 0.95))
        next_values = torch.cat([values[:, 1:], bootstrap[:, None]], dim=1)
        inputs = rewards + discounts * (1.0 - lambda_) * next_values
        returns = []
        accumulator = bootstrap
        for step in reversed(range(rewards.shape[1])):
            accumulator = inputs[:, step] + discounts[:, step] * lambda_ * accumulator
            returns.append(accumulator)
        return torch.stack(list(reversed(returns)), dim=1)

    def _behavior_loss(self, start: RSSMState) -> tuple[torch.Tensor, torch.Tensor, dict]:
        horizon = int(self.config.get("imagination_horizon", 15))
        state = start.detach()
        log_probs, entropies, rewards, discounts, features = [], [], [], [], []
        for _ in range(horizon):
            feature = state.features.detach()
            distribution = torch.distributions.Categorical(logits=self.model.actor(feature))
            action_index = distribution.sample()
            action = F.one_hot(action_index, self.model.action_dim).float()
            state, _ = self.model.rssm.imagine_step(state, action)
            imagined_feature = state.features
            features.append(imagined_feature)
            log_probs.append(distribution.log_prob(action_index))
            entropies.append(distribution.entropy())
            rewards.append(self.model.reward_head(imagined_feature).squeeze(-1))
            continuation = torch.sigmoid(self.model.continue_head(imagined_feature).squeeze(-1))
            discounts.append(float(self.config.get("discount", 0.995)) * continuation)
        features_tensor = torch.stack(features, 1)
        rewards_tensor = torch.stack(rewards, 1)
        discounts_tensor = torch.stack(discounts, 1)
        values = self.model.slow_critic(features_tensor).squeeze(-1)
        bootstrap = values[:, -1]
        returns = self._lambda_returns(rewards_tensor, discounts_tensor, values, bootstrap)
        log_probs_tensor = torch.stack(log_probs, 1)
        entropy_tensor = torch.stack(entropies, 1)
        actor_loss = -(
            log_probs_tensor * returns.detach()
            + float(self.config.get("actor_entropy", 0.001)) * entropy_tensor
        ).mean()
        critic_prediction = self.model.critic(features_tensor.detach()).squeeze(-1)
        critic_loss = F.mse_loss(critic_prediction, returns.detach())
        return actor_loss, critic_loss, {
            "loss_actor": actor_loss.detach(),
            "loss_critic": critic_loss.detach(),
            "imagined_return": returns.mean().detach(),
        }

    def train_step(self) -> dict[str, float]:
        batch, cached_states = self.replay.next_batch()
        batch_size = batch["images"].shape[0]
        initial = self._stack_initial_states(cached_states, batch_size)
        states, priors, posteriors = self._observe(batch, initial)
        model_loss, metrics = self._world_model_loss(batch, states, priors, posteriors)
        self.model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip", 200)))
        self.model_optimizer.step()
        self.replay.update_states(states[-1])

        start = RSSMState(
            torch.cat([state.deterministic for state in states], dim=0),
            torch.cat([state.stochastic for state in states], dim=0),
        )
        actor_loss, critic_loss, behavior_metrics = self._behavior_loss(start)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), float(self.config.get("grad_clip", 200)))
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), float(self.config.get("grad_clip", 200)))
        self.critic_optimizer.step()
        self.updates += 1
        if self.updates % int(self.config.get("slow_critic_interval", 100)) == 0:
            self.model.slow_critic.load_state_dict(self.model.critic.state_dict())
        metrics.update(behavior_metrics)
        return {name: float(value.cpu()) for name, value in metrics.items()}

    @torch.inference_mode()
    def policy(
        self,
        observation: np.ndarray,
        previous_action: int,
        state: RSSMState | None,
        *,
        sample: bool = True,
    ) -> tuple[int, RSSMState]:
        image = torch.from_numpy(np.asarray(observation)[None, None]).to(self.device)
        embedding = self.model.encoder(image)[:, 0]
        if state is None:
            state = self.model.initial(1, self.device)
        action = F.one_hot(
            torch.tensor([previous_action], device=self.device), self.model.action_dim
        ).float()
        state, _, _ = self.model.rssm.observe_step(state, action, embedding)
        distribution = torch.distributions.Categorical(logits=self.model.actor(state.features))
        selected = distribution.sample() if sample else distribution.probs.argmax(-1)
        return int(selected.item()), state

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
            "config": self.config,
        }
