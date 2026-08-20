"""Visual Algorithm Distillation with causal state/action/reward tokens."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import GPT2Transformer, ImageEncoder


class AD(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.n_transit = int(config["n_transit"])
        self.num_actions = int(config.get("num_actions", 6))
        embedding_dim = int(config["tf_n_embd"])
        self.image_encoder = ImageEncoder(embedding_dim, int(config.get("cnn_depth", 32)))
        self.embed_action = nn.Embedding(self.num_actions, embedding_dim)
        # Reward and episode-boundary flag share the reward token.
        self.embed_reward = nn.Linear(2, embedding_dim)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, embedding_dim))
        self.transformer = GPT2Transformer(
            embedding_dim=embedding_dim,
            n_heads=int(config.get("tf_n_head", 8)),
            n_layers=int(config.get("tf_n_layer", 4)),
            max_sequence_length=3 * self.n_transit,
            feedforward_dim=int(config.get("tf_dim_feedforward", 4 * embedding_dim)),
            dropout=float(config.get("tf_dropout", 0.1)),
        )
        self.pred_action = nn.Linear(embedding_dim, self.num_actions)
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            reduction="none",
        )
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _embed_reward(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        return self.embed_reward(torch.stack([rewards.float(), dones.float()], dim=-1))

    def build_tokens(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self.image_encoder(states)
        action_tokens = self.embed_action(actions.long())
        reward_tokens = self._embed_reward(rewards, dones)
        tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        tokens = tokens + self.type_embedding
        return tokens.flatten(1, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        valid = batch.get("valid_mask", torch.ones_like(actions, dtype=torch.bool)).to(self.device)
        tokens = self.build_tokens(states, actions, rewards, dones)
        token_mask = valid.unsqueeze(-1).expand(-1, -1, 3).flatten(1, 2)
        output = self.transformer(tokens, attention_mask=token_mask)
        logits = self.pred_action(output[:, 0::3])
        losses = self.loss_fn(logits.flatten(0, 1), actions.flatten()).reshape_as(actions)
        denominator = valid.sum().clamp_min(1)
        loss = (losses * valid).sum() / denominator
        accuracy = ((logits.argmax(-1) == actions) & valid).sum() / denominator
        return {"loss_action": loss, "loss_total": loss, "acc_action": accuracy}

    @torch.inference_mode()
    def predict_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        *,
        sample: bool = True,
    ) -> torch.Tensor:
        """Predict the next action from completed history plus the final query state.

        ``states`` has one more timestep than actions/rewards/dones.
        """
        completed_states = states[:, :-1]
        tokens = self.build_tokens(completed_states, actions, rewards, dones)
        query = self.image_encoder(states[:, -1:]) + self.type_embedding[:, :, 0]
        tokens = torch.cat([tokens, query], dim=1)
        max_inference_length = self.transformer.max_sequence_length - 2
        tokens = tokens[:, -max_inference_length:]
        logits = self.pred_action(self.transformer(tokens)[:, -1])
        if sample:
            return torch.distributions.Categorical(logits=logits).sample()
        return logits.argmax(-1)

    @torch.inference_mode()
    def start_context(self, observation: np.ndarray | torch.Tensor) -> dict:
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0).unsqueeze(0)
        token = self.image_encoder(state) + self.type_embedding[:, :, 0]
        return {"tokens": token}

    @torch.inference_mode()
    def act(self, context: dict, *, sample: bool = True) -> int:
        logits = self.pred_action(self.transformer(context["tokens"])[:, -1])
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(-1)
        return int(action.item())

    @torch.inference_mode()
    def observe(
        self,
        context: dict,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray | torch.Tensor,
    ) -> None:
        action_token = self.embed_action(
            torch.tensor([[action]], device=self.device)
        ) + self.type_embedding[:, :, 1]
        reward_token = self.embed_reward(
            torch.tensor([[[reward, float(done)]]], device=self.device)
        ) + self.type_embedding[:, :, 2]
        state = torch.as_tensor(next_observation, device=self.device).unsqueeze(0).unsqueeze(0)
        state_token = self.image_encoder(state) + self.type_embedding[:, :, 0]
        context["tokens"] = torch.cat(
            [context["tokens"], action_token, reward_token, state_token], dim=1
        )
        # Inference context is (s, a, r, ..., s_query), hence 1 mod 3.
        max_inference_length = self.transformer.max_sequence_length - 2
        context["tokens"] = context["tokens"][:, -max_inference_length:]
