"""Shared visual, causal-transformer, and compression modules."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Dreamer-style strided CNN that maps 64x64 RGB frames to one token."""

    def __init__(self, embedding_dim: int, depth: int = 32) -> None:
        super().__init__()
        channels = [3, depth, 2 * depth, 4 * depth, 8 * depth]
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2),
                    nn.ELU(),
                ]
            )
        self.convolution = nn.Sequential(*layers)
        # Four 4x4/stride-2 convolutions map 64x64 inputs to 2x2 features.
        self.projection = nn.Linear(8 * depth * 2 * 2, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] == 3:
            images = images.movedim(-1, -3)
        leading_shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:]).float().div(255.0).sub(0.5)
        encoded = self.convolution(flat).flatten(1)
        encoded = self.normalization(self.projection(encoded))
        return encoded.reshape(*leading_shape, encoded.shape[-1])


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, embedding_dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        additive_mask = None
        if attention_mask is not None:
            additive_mask = torch.zeros(
                batch, 1, 1, length, device=x.device, dtype=x.dtype
            ).masked_fill(~attention_mask[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, embedding_dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attention_mask)
        return x + self.mlp(self.norm2(x))


class GPT2Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        max_sequence_length: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_sequence_length = int(max_sequence_length)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_sequence_length, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, n_heads, feedforward_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds {self.max_sequence_length}"
            )
        x = self.dropout(x + self.position_embedding[:, : x.shape[1]])
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)


class CompressionTransformer(nn.Module):
    """Cross-attention bottleneck from arbitrary context to fixed latent tokens."""

    def __init__(
        self,
        embedding_dim: int,
        n_tokens: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(1, n_tokens, embedding_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(embedding_dim),
                        "context_norm": nn.LayerNorm(embedding_dim),
                        "attention": nn.MultiheadAttention(
                            embedding_dim,
                            n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "mlp_norm": nn.LayerNorm(embedding_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(embedding_dim, 4 * embedding_dim),
                            nn.GELU(),
                            nn.Linear(4 * embedding_dim, embedding_dim),
                        ),
                    }
                )
            )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        query = self.query_tokens.expand(context.shape[0], -1, -1)
        for layer in self.layers:
            update, _ = layer["attention"](
                layer["query_norm"](query),
                layer["context_norm"](context),
                layer["context_norm"](context),
                need_weights=False,
            )
            query = query + update
            query = query + layer["mlp"](layer["mlp_norm"](query))
        return query


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_output_length: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.position_queries = nn.Parameter(
            torch.empty(1, max_output_length, embedding_dim)
        )
        nn.init.trunc_normal_(self.position_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, latent: torch.Tensor, output_length: int) -> torch.Tensor:
        if output_length > self.position_queries.shape[1]:
            raise ValueError("Requested reconstruction exceeds configured maximum")
        queries = self.position_queries[:, :output_length].expand(latent.shape[0], -1, -1)
        return self.output(self.decoder(queries, latent))
