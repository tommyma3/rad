"""
GPT-2 Style Transformer Block for Algorithm Distillation.

GPT-2 uses Pre-LayerNorm (LayerNorm before attention/FFN), which differs from
the original Transformer's Post-LayerNorm design.

Pre-LN structure:
    x = x + Attn(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

This provides better gradient flow and training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional masking.
    GPT-2 style implementation.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_length=1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool(),
            persistent=False
        )
        
    def forward(self, x, attention_mask=None, use_causal_mask=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional custom attention mask (batch, seq_len, seq_len) or (seq_len, seq_len)
            use_causal_mask: Whether to apply causal masking
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch, n_heads, seq_len, head_dim)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        if use_causal_mask:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        # Apply custom attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(attention_mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, head_dim)
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


class MLP(nn.Module):
    """
    GPT-2 style MLP (Feed-Forward Network).
    Uses GELU activation.
    """
    def __init__(self, d_model, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """
    GPT-2 style transformer block with Pre-LayerNorm.
    
    Structure:
        x = x + Attn(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1, max_seq_length=1024):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dim_feedforward, dropout)
        
    def forward(self, x, attention_mask=None, use_causal_mask=True):
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            use_causal_mask: Whether to apply causal masking
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-LN Self-Attention
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask, use_causal_mask=use_causal_mask)
        # Pre-LN MLP
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Transformer(nn.Module):
    """
    GPT-2 style decoder-only transformer.
    
    Features:
    - Pre-LayerNorm (LayerNorm before attention/MLP)
    - Learned positional embeddings
    - Causal masking for autoregressive generation
    - Final LayerNorm before output
    """
    def __init__(self, d_model, n_heads, n_layers, max_seq_length=1024, dim_feedforward=None, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_heads, dim_feedforward, dropout, max_seq_length)
            for _ in range(n_layers)
        ])
        
        # Final layer norm (GPT-2 style)
        self.ln_f = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
    def _init_weights(self, module):
        """Initialize weights like GPT-2."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(self, x, attention_mask=None, use_causal_mask=True):
        """
        Args:
            x: (batch, seq_len, d_model) - input embeddings
            attention_mask: Optional attention mask
            use_causal_mask: Whether to apply causal masking
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        return x
    
    def forward_with_custom_positions(self, x, positions, attention_mask=None, use_causal_mask=True):
        """
        Forward pass with custom positional indices.
        Useful when latent tokens need special positional treatment.
        
        Args:
            x: (batch, seq_len, d_model) - input embeddings
            positions: (batch, seq_len) - position indices for each token
            attention_mask: Optional attention mask
            use_causal_mask: Whether to apply causal masking
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Gather positional embeddings based on position indices
        pos_emb = self.pos_embedding.squeeze(0)[positions]  # (batch, seq_len, d_model)
        
        x = x + pos_emb
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
        
        x = self.ln_f(x)
        
        return x
