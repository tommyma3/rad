"""Recurrent Algorithm Distillation with bounded active MiniGrid context."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ad import AD
from .common import CompressionTransformer, ReconstructionDecoder


class RAD(AD):
    LATENT_UPDATE_MODES = {"replace", "residual", "multiplicative_gate", "gru_gate"}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        embedding_dim = int(config["tf_n_embd"])
        self.n_compress_tokens = int(config["n_compress_tokens"])
        if self.n_compress_tokens % 3:
            raise ValueError("n_compress_tokens must be divisible by the S/A/R group size")
        if self.n_compress_tokens >= self.max_sequence_length:
            raise ValueError("Compression tokens must fit beside at least one raw transition")
        self.short_memory_keep = int(config["short_memory_keep"])
        self.short_memory_tokens = 3 * self.short_memory_keep
        self.max_gradient_rounds = int(config.get("max_gradient_rounds", 2))
        self.max_compressions = config.get("max_compressions")
        self.always_use_latent_prefix = bool(config.get("always_use_latent_prefix", True))
        self.latent_update_mode = config.get("latent_update_mode", "gru_gate")
        if self.latent_update_mode not in self.LATENT_UPDATE_MODES:
            raise ValueError(f"Unknown latent update mode: {self.latent_update_mode}")
        self.null_latent_tokens = nn.Parameter(
            torch.empty(1, self.n_compress_tokens, embedding_dim)
        )
        nn.init.trunc_normal_(self.null_latent_tokens, std=0.02)
        self.compression_transformer = CompressionTransformer(
            embedding_dim,
            self.n_compress_tokens,
            int(config.get("compress_n_heads", 4)),
            int(config.get("compress_n_layers", 3)),
            float(config.get("tf_dropout", 0.1)),
        )
        self.reconstruction_decoder = ReconstructionDecoder(
            embedding_dim,
            self.max_sequence_length + self.n_compress_tokens,
            int(config.get("compress_n_heads", 4)),
            int(config.get("compress_n_layers", 3)),
            float(config.get("tf_dropout", 0.1)),
        )
        self.latent_residual_norm = nn.LayerNorm(embedding_dim)
        self.latent_multiplicative_gate = nn.Linear(2 * embedding_dim, embedding_dim)
        self.latent_gru_gate = nn.Linear(2 * embedding_dim, embedding_dim)
        self.latent_gru_candidate = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.zeros_(self.latent_gru_gate.weight)
        nn.init.constant_(self.latent_gru_gate.bias, -2.0)

    def _initial_latent(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.always_use_latent_prefix:
            return None
        return self.null_latent_tokens.to(dtype=dtype).expand(batch_size, -1, -1)

    def _update_latent(self, old: torch.Tensor | None, candidate: torch.Tensor) -> torch.Tensor:
        if old is None or self.latent_update_mode == "replace":
            return candidate
        joined = torch.cat([old, candidate], dim=-1)
        if self.latent_update_mode == "residual":
            return self.latent_residual_norm(old + candidate)
        if self.latent_update_mode == "multiplicative_gate":
            gate = torch.sigmoid(self.latent_multiplicative_gate(joined))
            return gate * old + (1.0 - gate) * candidate
        if self.latent_update_mode == "gru_gate":
            gate = torch.sigmoid(self.latent_gru_gate(joined))
            proposal = torch.tanh(self.latent_gru_candidate(joined))
            return gate * old + (1.0 - gate) * proposal
        raise RuntimeError("unreachable")

    def _compression_plan(self, token_length: int) -> int:
        latent_length = self.n_compress_tokens if self.always_use_latent_prefix else 0
        recent = int(token_length)
        rounds = 0
        while latent_length + recent > self.max_sequence_length:
            capacity = self.max_sequence_length - latent_length
            consume = max(3, capacity - self.short_memory_tokens)
            consume -= consume % 3
            if consume <= 0:
                raise ValueError("RAD configuration leaves no compression progress")
            recent -= min(consume, recent - self.short_memory_tokens)
            latent_length = self.n_compress_tokens
            rounds += 1
        return rounds

    def _roll_context(
        self,
        tokens: torch.Tensor,
        *,
        respect_curriculum: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor, int]:
        latent = self._initial_latent(tokens.shape[0], tokens.dtype)
        recent = tokens
        planned = self._compression_plan(tokens.shape[1])
        gradient_start = max(0, planned - self.max_gradient_rounds)
        rounds = 0
        while recent.shape[1] + (0 if latent is None else latent.shape[1]) > self.max_sequence_length:
            if respect_curriculum and self.max_compressions is not None and rounds >= int(self.max_compressions):
                available = self.max_sequence_length - (0 if latent is None else latent.shape[1])
                available -= available % 3
                recent = recent[:, -available:]
                break
            latent_length = 0 if latent is None else latent.shape[1]
            capacity = self.max_sequence_length - latent_length
            consume = max(3, capacity - self.short_memory_tokens)
            consume -= consume % 3
            consume = min(consume, recent.shape[1] - self.short_memory_tokens)
            consume -= consume % 3
            if consume <= 0:
                raise ValueError("RAD configuration cannot reduce its recent context")
            prefix = recent[:, :consume]
            recent = recent[:, consume:]
            compression_input = torch.cat([latent, prefix], dim=1) if latent is not None else prefix
            if rounds < gradient_start:
                with torch.no_grad():
                    candidate = self.compression_transformer(compression_input)
                candidate = candidate.detach()
            else:
                candidate = self.compression_transformer(compression_input)
            latent = self._update_latent(latent, candidate)
            rounds += 1
        return latent, recent, rounds

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        pretrain_compression: bool = False,
    ) -> dict[str, torch.Tensor]:
        if pretrain_compression:
            return self.forward_pretrain_compression(batch)
        valid = batch.get(
            "valid_mask",
            torch.ones_like(batch["actions"], dtype=torch.bool),
        ).to(self.device)
        raw_tokens = self.build_tokens(batch)
        targets = batch["actions"].to(self.device).long()
        decisions = batch.get("decision_mask")
        if decisions is not None:
            decisions = decisions.to(self.device)
        losses = []
        correct = []
        decision_correct = []
        decision_count = 0
        logits_by_row = []
        total_compressions = 0
        for row in range(raw_tokens.shape[0]):
            length = int(valid[row].sum().item())
            # Training queries match streaming inference exactly: completed
            # transitions followed by the final state, with no future a/r pair.
            tokens = raw_tokens[row : row + 1, : 3 * length - 2]
            latent, recent, rounds = self._roll_context(tokens, respect_curriculum=True)
            total_compressions += rounds
            packed = torch.cat([latent, recent], dim=1) if latent is not None else recent
            output = self.transformer(packed)
            offset = packed.shape[1] - recent.shape[1]
            logits = self.pred_action(output[:, offset::3])
            final_logits = logits[:, -1]
            row_targets = targets[row : row + 1, length - 1]
            row_losses = self.loss_fn(final_logits, row_targets)
            losses.append(row_losses)
            predictions = final_logits.argmax(-1)
            correct.append((predictions == row_targets).flatten())
            logits_by_row.append(final_logits)
            if decisions is not None:
                row_decision = decisions[row : row + 1, length - 1]
                if row_decision.any():
                    decision_correct.append(((predictions == row_targets) & row_decision).sum())
                    decision_count += int(row_decision.sum().item())
        all_losses = torch.cat(losses)
        all_correct = torch.cat(correct)
        decision_accuracy = torch.zeros((), device=self.device)
        if decision_count:
            decision_accuracy = torch.stack(decision_correct).sum() / decision_count
        return {
            "loss_action": all_losses.mean(),
            "loss_total": all_losses.mean(),
            "acc_action": all_correct.float().mean(),
            "acc_decision": decision_accuracy,
            "num_compressions": torch.tensor(total_compressions, device=self.device),
            "logits_by_row": logits_by_row,
        }

    def forward_pretrain_compression(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        valid = batch.get(
            "valid_mask",
            torch.ones_like(batch["actions"], dtype=torch.bool),
        ).to(self.device)
        raw = self.build_tokens(batch)
        losses = []
        for row in range(raw.shape[0]):
            length = int(valid[row].sum().item())
            tokens = raw[row : row + 1, : min(3 * length, self.max_sequence_length)]
            null = self.null_latent_tokens.to(tokens.dtype)
            target = torch.cat([null, tokens], dim=1) if self.always_use_latent_prefix else tokens
            latent = self.compression_transformer(target)
            reconstructed = self.reconstruction_decoder(latent, target.shape[1])
            losses.append(F.mse_loss(reconstructed, target.detach()))
        loss = torch.stack(losses).mean()
        return {"loss_recon": loss, "loss_total": loss}

    @torch.inference_mode()
    def start_context(self, observation: dict[str, Any]) -> dict[str, Any]:
        base = super().start_context(observation)
        recent = base["tokens"]
        return {
            "latent": self._initial_latent(1, recent.dtype),
            "recent": recent,
            "num_compressions": 0,
        }

    @torch.inference_mode()
    def _compress_streaming(self, context: dict[str, Any]) -> None:
        while context["recent"].shape[1] + (
            0 if context["latent"] is None else context["latent"].shape[1]
        ) > self.max_sequence_length:
            latent_length = 0 if context["latent"] is None else context["latent"].shape[1]
            capacity = self.max_sequence_length - latent_length
            consume = max(3, capacity - self.short_memory_tokens)
            consume -= consume % 3
            consume = min(consume, context["recent"].shape[1] - self.short_memory_tokens)
            consume -= consume % 3
            prefix = context["recent"][:, :consume]
            context["recent"] = context["recent"][:, consume:]
            compressor_input = (
                torch.cat([context["latent"], prefix], dim=1)
                if context["latent"] is not None
                else prefix
            )
            candidate = self.compression_transformer(compressor_input)
            context["latent"] = self._update_latent(context["latent"], candidate)
            context["num_compressions"] += 1

    @torch.inference_mode()
    def action_logits(self, context: dict[str, Any]) -> torch.Tensor:
        self._compress_streaming(context)
        packed = (
            torch.cat([context["latent"], context["recent"]], dim=1)
            if context["latent"] is not None
            else context["recent"]
        )
        return self.pred_action(self.transformer(packed)[:, -1])

    @torch.inference_mode()
    def act(self, context: dict[str, Any], *, sample: bool = False) -> int:
        logits = self.action_logits(context)
        action = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
        return int(action.item())

    @torch.inference_mode()
    def observe(
        self,
        context: dict[str, Any],
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
        context["recent"] = torch.cat([context["recent"], additions], dim=1)
        self._compress_streaming(context)

    def set_curriculum(self, max_compressions: int | None) -> None:
        self.max_compressions = max_compressions
