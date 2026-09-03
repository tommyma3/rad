"""Causal Algorithm Distillation baseline for MiniGrid Memory."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .common import GPT2Transformer, MiniGridStateEncoder


class AD(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.n_transit = int(config["n_transit"])
        self.max_sequence_length = 3 * self.n_transit
        self.num_actions = int(config.get("num_actions", 7))
        embedding_dim = int(config["tf_n_embd"])
        self.state_encoder = MiniGridStateEncoder(
            embedding_dim,
            int(config.get("tile_embedding_dim", 16)),
        )
        self.embed_action = nn.Embedding(self.num_actions, embedding_dim)
        self.embed_reward = nn.Linear(3, embedding_dim)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, embedding_dim))
        self.transformer = GPT2Transformer(
            embedding_dim=embedding_dim,
            n_heads=int(config.get("tf_n_head", 4)),
            n_layers=int(config.get("tf_n_layer", 4)),
            max_sequence_length=self.max_sequence_length,
            feedforward_dim=int(config.get("tf_dim_feedforward", 4 * embedding_dim)),
            dropout=float(config.get("tf_dropout", 0.1)),
        )
        self.pred_action = nn.Linear(embedding_dim, self.num_actions)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _state_tokens(self, images: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(images, directions) + self.type_embedding[:, :, 0]

    def _action_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        return self.embed_action(actions.long()) + self.type_embedding[:, :, 1]

    def _reward_tokens(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([rewards.float(), terminated.float(), truncated.float()], dim=-1)
        return self.embed_reward(features) + self.type_embedding[:, :, 2]

    def build_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        states = self._state_tokens(
            batch["images"].to(self.device),
            batch["directions"].to(self.device),
        )
        actions = self._action_tokens(batch["actions"].to(self.device))
        rewards = self._reward_tokens(
            batch["rewards"].to(self.device),
            batch["terminated"].to(self.device),
            batch["truncated"].to(self.device),
        )
        return torch.stack([states, actions, rewards], dim=2).flatten(1, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        valid = batch.get(
            "valid_mask",
            torch.ones_like(batch["actions"], dtype=torch.bool),
        ).to(self.device)
        tokens = self.build_tokens(batch)
        token_valid = valid.unsqueeze(-1).expand(-1, -1, 3).flatten(1, 2)
        output = self.transformer(tokens, token_valid)
        logits = self.pred_action(output[:, 0::3])
        targets = batch["actions"].to(self.device).long()
        final_indices = valid.sum(dim=1).long() - 1
        rows = torch.arange(targets.shape[0], device=self.device)
        final_logits = logits[rows, final_indices]
        final_targets = targets[rows, final_indices]
        loss = self.loss_fn(final_logits, final_targets).mean()
        accuracy = (final_logits.argmax(-1) == final_targets).float().mean()
        decision = batch.get("decision_mask")
        decision_accuracy = torch.zeros((), device=self.device)
        if decision is not None:
            final_decision = decision.to(self.device)[rows, final_indices]
            if final_decision.any():
                decision_accuracy = (
                    (final_logits.argmax(-1) == final_targets) & final_decision
                ).sum() / final_decision.sum()
        return {
            "loss_action": loss,
            "loss_total": loss,
            "acc_action": accuracy,
            "acc_decision": decision_accuracy,
            "logits": logits,
            "final_logits": final_logits,
        }

    @torch.inference_mode()
    def start_context(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = torch.as_tensor(np.asarray(observation["image"]), device=self.device)[None, None]
        direction = torch.as_tensor([[observation["direction"]]], device=self.device)
        return {"tokens": self._state_tokens(image, direction)}

    @torch.inference_mode()
    def action_logits(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.transformer(context["tokens"])
        return self.pred_action(output[:, -1])

    @torch.inference_mode()
    def act(self, context: dict[str, torch.Tensor], *, sample: bool = False) -> int:
        logits = self.action_logits(context)
        action = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
        return int(action.item())

    @torch.inference_mode()
    def observe(
        self,
        context: dict[str, torch.Tensor],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_observation: dict[str, Any],
    ) -> None:
        additions = self._transition_tokens(
            action,
            reward,
            terminated,
            truncated,
            next_observation,
        )
        context["tokens"] = torch.cat([context["tokens"], additions], dim=1)
        maximum = self.max_sequence_length - 2
        context["tokens"] = context["tokens"][:, -maximum:]

    def _transition_tokens(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_observation: dict[str, Any],
    ) -> torch.Tensor:
        action_tensor = torch.tensor([[action]], device=self.device)
        reward_tensor = torch.tensor([[reward]], device=self.device)
        terminated_tensor = torch.tensor([[terminated]], device=self.device)
        truncated_tensor = torch.tensor([[truncated]], device=self.device)
        image = torch.as_tensor(np.asarray(next_observation["image"]), device=self.device)[None, None]
        direction = torch.as_tensor([[next_observation["direction"]]], device=self.device)
        return torch.cat(
            [
                self._action_tokens(action_tensor),
                self._reward_tokens(reward_tensor, terminated_tensor, truncated_tensor),
                self._state_tokens(image, direction),
            ],
            dim=1,
        )
