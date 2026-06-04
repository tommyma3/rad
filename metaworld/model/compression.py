"""
Compression Transformer Module for Recurrent Algorithm Distillation (RAD).

This module implements:
1. CompressionTransformer: Compresses sequences into fixed-size latent tokens
2. ReconstructionDecoder: Reconstructs sequences from latent tokens (for pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer where queries attend to key-value pairs from context.
    Used in the compression transformer to compress context into query tokens.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, queries, context):
        """
        Args:
            queries: (batch, n_queries, d_model) - the learnable query tokens
            context: (batch, context_len, d_model) - the sequence to compress
            
        Returns:
            output: (batch, n_queries, d_model)
        """
        batch_size, n_queries, _ = queries.shape
        context_len = context.shape[1]
        
        # Project queries, keys, values
        q = self.q_proj(queries)  # (batch, n_queries, d_model)
        k = self.k_proj(context)  # (batch, context_len, d_model)
        v = self.v_proj(context)  # (batch, context_len, d_model)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, n_queries, context_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, n_queries, head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        return out


class CompressionTransformerLayer(nn.Module):
    """
    Single layer of the compression transformer.
    Consists of: Self-Attention -> Cross-Attention -> FFN
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Self-attention among query tokens
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention: queries attend to context
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, context):
        """
        Args:
            queries: (batch, n_queries, d_model)
            context: (batch, context_len, d_model)
            
        Returns:
            queries: (batch, n_queries, d_model)
        """
        # Self-attention (queries attend to each other)
        q_norm = self.self_attn_norm(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_attn_out
        
        # Cross-attention (queries attend to context)
        q_norm = self.cross_attn_norm(queries)
        cross_attn_out = self.cross_attn(q_norm, context)
        queries = queries + cross_attn_out
        
        # FFN
        q_norm = self.ffn_norm(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out
        
        return queries


class CompressionTransformer(nn.Module):
    """
    Compression Transformer that compresses a sequence into fixed-size latent tokens.
    
    Architecture (Q-Former style):
    - Learnable query tokens that attend to the input sequence
    - Multiple layers of [Self-Attention -> Cross-Attention -> FFN]
    - Output: latent tokens that represent compressed information
    """
    def __init__(self, d_model, n_heads, n_layers, n_compress_tokens, dim_feedforward=None, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_compress_tokens = n_compress_tokens
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        # Learnable compression query tokens
        self.compress_queries = nn.Parameter(torch.randn(1, n_compress_tokens, d_model) * 0.02)
        
        # Positional embedding for context (will be added to input)
        self.max_context_length = 2048  # Maximum context length
        self.context_pos_embedding = nn.Parameter(torch.randn(1, self.max_context_length, d_model) * 0.02)
        
        # Compression layers
        self.layers = nn.ModuleList([
            CompressionTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, context):
        """
        Compress a sequence into latent tokens.
        
        Args:
            context: (batch, context_len, d_model) - the sequence to compress
            
        Returns:
            latent_tokens: (batch, n_compress_tokens, d_model)
        """
        batch_size = context.shape[0]
        context_len = context.shape[1]
        
        # Add positional embeddings to context
        context = context + self.context_pos_embedding[:, :context_len, :]
        
        # Expand query tokens for batch
        queries = self.compress_queries.expand(batch_size, -1, -1)
        
        # Pass through compression layers
        for layer in self.layers:
            queries = layer(queries, context)
            
        # Final normalization
        latent_tokens = self.final_norm(queries)
        
        return latent_tokens


class ReconstructionDecoder(nn.Module):
    """
    Decoder that reconstructs the original sequence from latent tokens.
    Used for pre-training the compression transformer.
    
    Architecture: Transformer decoder with cross-attention to latent tokens.
    """
    def __init__(self, d_model, n_heads, n_layers, max_seq_length, dim_feedforward=None, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        # max_seq_length for variable-length reconstruction
        self.max_seq_length = max_seq_length
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        # Learnable position queries for reconstruction
        self.position_queries = nn.Parameter(torch.randn(1, self.max_seq_length, d_model) * 0.02)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            ReconstructionDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, latent_tokens, target_length):
        """
        Reconstruct sequence from latent tokens.
        
        Args:
            latent_tokens: (batch, n_compress_tokens, d_model)
            target_length: int, the length of sequence to reconstruct
            
        Returns:
            reconstructed: (batch, target_length, d_model)
        """
        batch_size = latent_tokens.shape[0]
        
        # Validate target_length doesn't exceed max_seq_length
        if target_length > self.max_seq_length:
            raise ValueError(f"target_length ({target_length}) exceeds max_seq_length ({self.max_seq_length})")
        
        # Use position queries up to target length
        queries = self.position_queries[:, :target_length, :].expand(batch_size, -1, -1)
        
        # Pass through decoder layers
        for layer in self.layers:
            queries = layer(queries, latent_tokens)
            
        # Final normalization
        reconstructed = self.final_norm(queries)
        
        return reconstructed


class ReconstructionDecoderLayer(nn.Module):
    """
    Single layer of the reconstruction decoder.
    Similar to CompressionTransformerLayer but queries attend to latent tokens.
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Self-attention among reconstruction queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention: queries attend to latent tokens
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, latent_tokens):
        """
        Args:
            queries: (batch, seq_len, d_model) - position queries for reconstruction
            latent_tokens: (batch, n_compress_tokens, d_model)
            
        Returns:
            queries: (batch, seq_len, d_model)
        """
        # Self-attention
        q_norm = self.self_attn_norm(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_attn_out
        
        # Cross-attention to latent tokens
        q_norm = self.cross_attn_norm(queries)
        cross_attn_out = self.cross_attn(q_norm, latent_tokens)
        queries = queries + cross_attn_out
        
        # FFN
        q_norm = self.ffn_norm(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out
        
        return queries
