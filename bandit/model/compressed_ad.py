"""Recurrent AD with separate raw-history and latent budgets.

Compression happens on complete transitions after appending transition K+1.
The oldest transitions are compressed and the latest p transitions are retained.
Query observations never enter the compressor until their outcomes are observed.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ad import AD
from .compression import CompressionTransformer, ReconstructionDecoder


class RAD(AD):
    def __init__(self, config):
        super().__init__(config)
        self.n_compress_tokens = int(config.get("n_compress_tokens", 15))
        self.short_memory_keep = int(config.get("short_memory_keep", 5))
        self.max_gradient_rounds = config.get("max_gradient_rounds")
        if self.n_compress_tokens <= 0 or self.n_compress_tokens % 3:
            raise ValueError("n_compress_tokens must be a positive multiple of three")
        if not 0 <= self.short_memory_keep < self.context_steps:
            raise ValueError("short_memory_keep must be in [0, context_steps)")
        if self.max_gradient_rounds is not None and self.max_gradient_rounds < 1:
            raise ValueError("max_gradient_rounds must be null or positive")
        self.always_use_latent_prefix = config.get("always_use_latent_prefix", False)
        self.latent_update_mode = config.get("latent_update_mode", "gru_gate")
        if self.latent_update_mode not in ("replace", "residual", "multiplicative_gate", "gru_gate"):
            raise ValueError("Unknown latent update mode")
        width = config["tf_n_embd"]
        self.latent_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.null_latent_tokens = nn.Parameter(torch.zeros(1, self.n_compress_tokens, width))
        nn.init.trunc_normal_(self.latent_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.null_latent_tokens, std=0.02)
        if self.latent_update_mode == "residual":
            self.latent_residual_norm = nn.LayerNorm(width)
        elif self.latent_update_mode == "multiplicative_gate":
            self.latent_multiplicative_gate = nn.Linear(width, width)
            nn.init.zeros_(self.latent_multiplicative_gate.weight)
            nn.init.constant_(self.latent_multiplicative_gate.bias, 4.0)
        elif self.latent_update_mode == "gru_gate":
            self.latent_gru_gate = nn.Linear(2 * width, width)
            self.latent_gru_candidate = nn.Linear(2 * width, width)
            nn.init.zeros_(self.latent_gru_gate.weight)
            nn.init.constant_(self.latent_gru_gate.bias, -2.0)
            nn.init.zeros_(self.latent_gru_candidate.weight)
            nn.init.zeros_(self.latent_gru_candidate.bias)
        self.compression_transformer = CompressionTransformer(
            width, config["compress_n_heads"], config["compress_n_layers"],
            self.n_compress_tokens, config["tf_dim_feedforward"], config["tf_dropout"],
            max_context_length=3 * (self.context_steps + 1) + self.n_compress_tokens)
        self.reconstruction_decoder = ReconstructionDecoder(
            width, config["compress_n_heads"], config["compress_n_layers"],
            3 * (self.context_steps + 1), config["tf_dim_feedforward"], config["tf_dropout"])

    def _update_latent_tokens(self, old, candidate):
        if old is None or self.latent_update_mode == "replace":
            return candidate
        if self.latent_update_mode == "residual":
            return self.latent_residual_norm(old + candidate)
        if self.latent_update_mode == "multiplicative_gate":
            return torch.sigmoid(self.latent_multiplicative_gate(old)) * candidate
        combined = torch.cat((old, candidate), dim=-1)
        gate = torch.sigmoid(self.latent_gru_gate(combined))
        candidate = candidate + torch.tanh(self.latent_gru_candidate(combined))
        return (1 - gate) * old + gate * candidate

    def compression_count_for_length(self, length):
        return max(0, math.ceil((length - self.context_steps) /
                                (self.context_steps + 1 - self.short_memory_keep)))

    def ingest(self, state, tokens, total_compressions=None):
        if tokens.shape[1] % 3:
            raise ValueError("Ingest complete observation/action/reward triples")
        cursor = 0
        while cursor < tokens.shape[1]:
            recent = state["recent"]
            recent_length = 0 if recent is None else recent.shape[1]
            # Fill to the next compression event, matching one-transition inference.
            take = min(tokens.shape[1] - cursor, 3 * (self.context_steps + 1) - recent_length)
            chunk = tokens[:, cursor:cursor + take]
            recent = chunk if recent is None else torch.cat((recent, chunk), dim=1)
            cursor += take
            if recent.shape[1] > 3 * self.context_steps:
                keep = 3 * self.short_memory_keep
                old_tokens = recent[:, :recent.shape[1] - keep]
                old_latent = state["latent"]
                pieces = [old_tokens] if old_latent is None else [old_latent, old_tokens]
                allow_gradient = (total_compressions is None or self.max_gradient_rounds is None or
                                  state["compression_count"] >= total_compressions - self.max_gradient_rounds)
                with torch.set_grad_enabled(torch.is_grad_enabled() and allow_gradient):
                    candidate = self.compression_transformer(torch.cat(pieces, dim=1))
                    state["latent"] = self._update_latent_tokens(old_latent, candidate)
                recent = recent[:, -keep:] if keep else recent[:, :0]
                state["compression_count"] += 1
            state["recent"] = recent
        return state

    def prefix_state(self, batch):
        tokens = self.embed_transitions(batch["states"], batch["actions"], batch["rewards"])
        return self.ingest(self.new_state(), tokens,
                           total_compressions=self.compression_count_for_length(batch["states"].shape[1]))

    def query_logits(self, state, observations):
        query = self.embed_state(observations.long()).unsqueeze(1) + self.type_embedding[:, :, 0]
        pieces = []
        latent = state["latent"]
        if latent is None and self.always_use_latent_prefix:
            latent = self.null_latent_tokens.expand(query.shape[0], -1, -1)
        if latent is not None:
            pieces.append(latent + self.latent_type_embedding)
        if state["recent"] is not None:
            pieces.append(state["recent"])
        pieces.append(query)
        output = self.ad_transformer(torch.cat(pieces, dim=1), use_causal_mask=True)
        return self.pred_action(output[:, -1])

    def forward(self, batch, pretrain=False):
        if not pretrain:
            return super().forward(batch)
        tokens = self.embed_transitions(batch["states"], batch["actions"], batch["rewards"]).detach()
        if tokens.shape[1] == 0 or tokens.shape[1] > 3 * (self.context_steps + 1):
            raise ValueError("Pretraining requires a nonempty chunk within the raw memory budget")
        latent = self.compression_transformer(tokens)
        reconstruction = self.reconstruction_decoder(latent, tokens.shape[1])
        loss = F.mse_loss(reconstruction, tokens)
        return {"loss": loss, "accuracy": loss.new_zeros(()), "num_compressions": 1}
