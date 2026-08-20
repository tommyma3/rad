"""Visual Recurrent Algorithm Distillation (RAD)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    CompressionTransformer,
    GPT2Transformer,
    ImageEncoder,
    ReconstructionDecoder,
)


class RAD(nn.Module):
    LATENT_UPDATE_MODES = {
        "replace",
        "residual",
        "multiplicative_gate",
        "gru_gate",
    }

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.n_transit = int(config["n_transit"])
        self.max_sequence_length = 3 * self.n_transit
        self.num_actions = int(config.get("num_actions", 6))
        self.n_compress_tokens = int(config["n_compress_tokens"])
        if self.n_compress_tokens % 3:
            raise ValueError("n_compress_tokens must be divisible by the S/A/R token group size 3")
        self.short_memory_keep = int(config["short_memory_keep"])
        self.short_memory_tokens = 3 * self.short_memory_keep
        self.max_gradient_rounds = int(config.get("max_gradient_rounds", 2))
        self.max_compressions = config.get("max_compressions")
        self.always_use_latent_prefix = bool(config.get("always_use_latent_prefix", True))
        self.latent_update_mode = config.get("latent_update_mode", "replace")
        if self.latent_update_mode not in self.LATENT_UPDATE_MODES:
            raise ValueError(
                f"Unknown latent_update_mode {self.latent_update_mode!r}; "
                f"expected one of {sorted(self.LATENT_UPDATE_MODES)}"
            )

        embedding_dim = int(config["tf_n_embd"])
        n_heads = int(config.get("tf_n_head", 8))
        dropout = float(config.get("tf_dropout", 0.1))
        self.image_encoder = ImageEncoder(embedding_dim, int(config.get("cnn_depth", 32)))
        self.embed_action = nn.Embedding(self.num_actions, embedding_dim)
        self.embed_reward = nn.Linear(2, embedding_dim)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, embedding_dim))
        self.ad_transformer = GPT2Transformer(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=int(config.get("tf_n_layer", 4)),
            max_sequence_length=self.max_sequence_length,
            feedforward_dim=int(config.get("tf_dim_feedforward", 4 * embedding_dim)),
            dropout=dropout,
        )
        self.compression_transformer = CompressionTransformer(
            embedding_dim=embedding_dim,
            n_tokens=self.n_compress_tokens,
            n_heads=int(config.get("compress_n_heads", 4)),
            n_layers=int(config.get("compress_n_layers", 4)),
            dropout=dropout,
        )
        self.reconstruction_decoder = ReconstructionDecoder(
            embedding_dim=embedding_dim,
            max_output_length=self.max_sequence_length,
            n_heads=int(config.get("compress_n_heads", 4)),
            n_layers=int(config.get("compress_n_layers", 4)),
            dropout=dropout,
        )
        self.pred_action = nn.Linear(embedding_dim, self.num_actions)
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            reduction="none",
        )

        if self.latent_update_mode == "multiplicative_gate":
            self.latent_gate = nn.Linear(2 * embedding_dim, embedding_dim)
        elif self.latent_update_mode == "gru_gate":
            self.latent_gru = nn.GRUCell(embedding_dim, embedding_dim)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_tokens(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        state_tokens = self.image_encoder(states)
        action_tokens = self.embed_action(actions.long())
        reward_tokens = self.embed_reward(torch.stack([rewards.float(), dones.float()], dim=-1))
        tokens = torch.stack([state_tokens, action_tokens, reward_tokens], dim=2)
        return (tokens + self.type_embedding).flatten(1, 2)

    def _update_latent(
        self,
        previous: torch.Tensor | None,
        proposed: torch.Tensor,
    ) -> torch.Tensor:
        if previous is None or self.latent_update_mode == "replace":
            return proposed
        if self.latent_update_mode == "residual":
            return previous + proposed
        if self.latent_update_mode == "multiplicative_gate":
            gate = torch.sigmoid(self.latent_gate(torch.cat([previous, proposed], dim=-1)))
            return gate * previous + (1.0 - gate) * proposed
        if self.latent_update_mode == "gru_gate":
            shape = proposed.shape
            return self.latent_gru(proposed.flatten(0, 1), previous.flatten(0, 1)).reshape(shape)
        raise RuntimeError("unreachable latent update mode")

    def _initial_latent(self, batch_size: int) -> torch.Tensor:
        return self.compression_transformer.query_tokens.expand(batch_size, -1, -1)

    def _compression_plan(self, token_length: int) -> int:
        reserved_prefix = self.n_compress_tokens if self.always_use_latent_prefix else 0
        if token_length + reserved_prefix <= self.max_sequence_length:
            return 0
        consumed_per_round = self.max_sequence_length - self.short_memory_tokens
        if consumed_per_round <= self.n_compress_tokens:
            raise ValueError("n_transit/short_memory_keep leave no room for compression progress")
        remaining = token_length
        rounds = 0
        has_latent = self.always_use_latent_prefix
        while remaining + (self.n_compress_tokens if has_latent else 0) > self.max_sequence_length:
            capacity = self.max_sequence_length - (self.n_compress_tokens if has_latent else 0)
            consume = max(1, capacity - self.short_memory_tokens)
            remaining -= consume
            has_latent = True
            rounds += 1
            if self.max_compressions is not None and rounds >= int(self.max_compressions):
                break
        return rounds

    def _roll_context(
        self,
        tokens: torch.Tensor,
        *,
        respect_curriculum: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, int]:
        latent = self._initial_latent(tokens.shape[0]) if self.always_use_latent_prefix else None
        remaining = tokens
        planned = self._compression_plan(tokens.shape[1])
        gradient_start = max(0, planned - self.max_gradient_rounds)
        rounds = 0
        while remaining.shape[1] + (
            self.n_compress_tokens
            if latent is not None or self.always_use_latent_prefix
            else 0
        ) > self.max_sequence_length:
            if respect_curriculum and self.max_compressions is not None:
                if rounds >= int(self.max_compressions):
                    # Curriculum-limited samples retain the newest representable context.
                    available = self.max_sequence_length - (
                        self.n_compress_tokens if latent is not None else 0
                    )
                    remaining = remaining[:, -available:]
                    break
            prefix_size = self.n_compress_tokens if latent is not None else 0
            capacity = self.max_sequence_length - prefix_size
            consume = max(1, capacity - self.short_memory_tokens)
            old = remaining[:, :consume]
            remaining = remaining[:, consume:]
            compression_input = torch.cat([latent, old], dim=1) if latent is not None else old
            if rounds < gradient_start:
                with torch.no_grad():
                    proposed = self.compression_transformer(compression_input)
                proposed = proposed.detach()
            else:
                proposed = self.compression_transformer(compression_input)
            latent = self._update_latent(latent, proposed)
            rounds += 1
        return latent, remaining, rounds

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        valid = batch.get("valid_mask", torch.ones_like(actions, dtype=torch.bool)).to(self.device)
        raw_tokens = self._build_tokens(states, actions, rewards, dones)
        latent, recent, rounds = self._roll_context(raw_tokens, respect_curriculum=True)
        transformer_input = torch.cat([latent, recent], dim=1) if latent is not None else recent
        output = self.ad_transformer(transformer_input)

        recent_steps = recent.shape[1] // 3
        state_offset = transformer_input.shape[1] - recent.shape[1]
        state_output = output[:, state_offset::3][:, :recent_steps]
        logits = self.pred_action(state_output)
        targets = actions[:, -recent_steps:]
        target_valid = valid[:, -recent_steps:]
        losses = self.loss_fn(logits.flatten(0, 1), targets.flatten()).reshape_as(targets)
        denominator = target_valid.sum().clamp_min(1)
        loss = (losses * target_valid).sum() / denominator
        accuracy = ((logits.argmax(-1) == targets) & target_valid).sum() / denominator
        return {
            "loss_action": loss,
            "loss_total": loss,
            "acc_action": accuracy,
            "num_compressions": torch.tensor(rounds, device=self.device),
        }

    def forward_pretrain_compression(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tokens = self._build_tokens(
            batch["states"].to(self.device),
            batch["actions"].to(self.device),
            batch["rewards"].to(self.device),
            batch["dones"].to(self.device),
        )
        if tokens.shape[1] > self.max_sequence_length:
            start = torch.randint(
                0,
                tokens.shape[1] - self.max_sequence_length + 1,
                (),
                device=tokens.device,
            ).item()
            tokens = tokens[:, start : start + self.max_sequence_length]
        latent = self.compression_transformer(tokens)
        reconstructed = self.reconstruction_decoder(latent, tokens.shape[1])
        loss = F.mse_loss(reconstructed, tokens.detach())
        return {"loss_recon": loss, "loss_total": loss}

    def set_curriculum(self, max_compressions: int | None) -> None:
        self.max_compressions = max_compressions

    def load_pretrained_compression(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint["model"]
        own_state = self.state_dict()
        prefixes = (
            "image_encoder.",
            "embed_action.",
            "embed_reward.",
            "type_embedding",
            "compression_transformer.",
            "reconstruction_decoder.",
        )
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        missing, unexpected = self.load_state_dict(selected, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected compression checkpoint keys: {unexpected}")
        required_missing = [key for key in own_state if key.startswith(prefixes) and key in missing]
        if required_missing:
            raise ValueError(f"Missing compression checkpoint keys: {required_missing}")

    @torch.inference_mode()
    def start_context(self, observation: np.ndarray | torch.Tensor) -> dict:
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0).unsqueeze(0)
        token = self.image_encoder(state) + self.type_embedding[:, :, 0]
        latent = self._initial_latent(1) if self.always_use_latent_prefix else None
        return {"latent": latent, "recent": token, "num_compressions": 0}

    @torch.inference_mode()
    def _compress_streaming_context(self, context: dict) -> None:
        while context["recent"].shape[1] + (
            self.n_compress_tokens if context["latent"] is not None else 0
        ) > self.max_sequence_length:
            prefix_size = self.n_compress_tokens if context["latent"] is not None else 0
            capacity = self.max_sequence_length - prefix_size
            consume = max(1, capacity - self.short_memory_tokens)
            old = context["recent"][:, :consume]
            context["recent"] = context["recent"][:, consume:]
            compression_input = (
                torch.cat([context["latent"], old], dim=1)
                if context["latent"] is not None
                else old
            )
            proposed = self.compression_transformer(compression_input)
            context["latent"] = self._update_latent(context["latent"], proposed)
            context["num_compressions"] += 1

    @torch.inference_mode()
    def act(self, context: dict, *, sample: bool = True) -> int:
        self._compress_streaming_context(context)
        transformer_input = (
            torch.cat([context["latent"], context["recent"]], dim=1)
            if context["latent"] is not None
            else context["recent"]
        )
        logits = self.pred_action(self.ad_transformer(transformer_input)[:, -1])
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
        context["recent"] = torch.cat(
            [context["recent"], action_token, reward_token, state_token], dim=1
        )
        self._compress_streaming_context(context)
