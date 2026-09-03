"""Categorical MiniGrid encoding and shared transformer components."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MiniGridStateEncoder(nn.Module):
    """Maps symbolic object/color/state tiles and direction to one state token."""

    def __init__(self, embedding_dim: int, tile_dim: int = 16) -> None:
        super().__init__()
        self.object_embedding = nn.Embedding(16, tile_dim)
        self.color_embedding = nn.Embedding(8, tile_dim)
        self.state_embedding = nn.Embedding(4, tile_dim)
        self.direction_embedding = nn.Embedding(4, tile_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(3 * tile_dim, 4 * tile_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(4 * tile_dim, 4 * tile_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(5 * tile_dim, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, images: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        leading = images.shape[:-3]
        height, width, channels = images.shape[-3:]
        if channels != 3:
            raise ValueError(f"Expected MiniGrid image channels last, got {images.shape}")
        flat = images.reshape(-1, height, width, channels).long()
        objects = self.object_embedding(flat[..., 0].clamp(0, 15))
        colors = self.color_embedding(flat[..., 1].clamp(0, 7))
        states = self.state_embedding(flat[..., 2].clamp(0, 3))
        tiles = torch.cat([objects, colors, states], dim=-1).movedim(-1, 1)
        spatial = self.spatial(tiles).flatten(1)
        direction = self.direction_embedding(directions.reshape(-1).long().clamp(0, 3))
        encoded = self.normalization(self.projection(torch.cat([spatial, direction], dim=-1)))
        return encoded.reshape(*leading, encoded.shape[-1])


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float, max_length: int):
        super().__init__()
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = float(dropout)
        self.register_buffer(
            "causal_block",
            torch.triu(torch.ones(max_length, max_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, embedding_dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        blocked = self.causal_block[:length, :length][None, None].expand(batch, 1, -1, -1)
        if valid_mask is not None:
            invalid_keys = ~valid_mask[:, None, None, :]
            blocked = blocked | invalid_keys
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=~blocked,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, embedding_dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, feedforward_dim: int, dropout: float, max_length: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, n_heads, dropout, max_length)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_mask)
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
                TransformerBlock(
                    embedding_dim,
                    n_heads,
                    feedforward_dim,
                    dropout,
                    self.max_sequence_length,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds {self.max_sequence_length}"
            )
        x = self.dropout(x + self.position_embedding[:, : x.shape[1]])
        for block in self.blocks:
            x = block(x, valid_mask)
        return self.final_norm(x)


class CompressionTransformer(nn.Module):
    """Cross-attention bottleneck from context into fixed latent queries."""

    def __init__(self, embedding_dim: int, n_tokens: int, n_heads: int, n_layers: int, dropout: float):
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
            normalized_context = layer["context_norm"](context)
            update, _ = layer["attention"](
                layer["query_norm"](query),
                normalized_context,
                normalized_context,
                need_weights=False,
            )
            query = query + update
            query = query + layer["mlp"](layer["mlp_norm"](query))
        return query


class ReconstructionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, max_output_length: int, n_heads: int, n_layers: int, dropout: float):
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
