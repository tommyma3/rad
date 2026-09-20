"""FLOP counting for AD/RAD inference via forward hooks.

Conventions:
- 1 MAC = 2 FLOPs; matmuls counted as 2*m*n*k (bias adds excluded, negligible).
- Attention counted with the full T^2 convention (causal-triangular saving ignored).
- Counted module types: nn.Linear (catches fused QKV/out-proj/MLP/gru gates),
  nn.Conv2d, CausalSelfAttention (SDPA matmuls only; projections are child
  Linears), nn.MultiheadAttention (whole block analytically - packed in/out
  projections + attention matmuls, since fused fast paths bypass child hooks),
  CrossAttentionLayer (SDPA matmuls only; projections are child Linears).
- Not counted (non-matmul / negligible): embeddings, LayerNorm, GELU, softmax,
  sampling, one-hot, token concatenation.
"""

import torch
import torch.nn as nn


def _attn_flops(batch, heads, q_len, kv_len, head_dim):
    """QK^T and AV matmul FLOPs: 2 * B*H*Tq*Tk*dh each."""
    return 4 * batch * heads * q_len * kv_len * head_dim


class FlopCounter:
    """Sums FLOPs of every hooked module under each attached root, tagged per root."""

    def __init__(self):
        self.by_tag = {}
        self.ad_seq_lens = []        # T of every CausalSelfAttention call (all layers)
        self.compress_context_lens = []  # S of every CrossAttentionLayer call (all layers)
        self._handles = []

    @property
    def total(self):
        return sum(self.by_tag.values())

    def add(self, tag, flops):
        self.by_tag[tag] = self.by_tag.get(tag, 0) + int(flops)

    def _linear_hook(self, tag):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            tokens = out.numel() // out.shape[-1]
            self.add(tag, 2 * tokens * module.in_features * module.out_features)
        return hook

    def _conv_hook(self, tag):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            batch = out.shape[0]
            out_hw = out.shape[2] * out.shape[3]
            kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            self.add(tag, 2 * batch * out.shape[1] * out_hw * kernel_ops)
        return hook

    def _causal_self_attn_hook(self, tag):
        def hook(module, inputs, output):
            x = inputs[0]
            batch, seq_len, _ = x.shape
            self.add(tag, _attn_flops(batch, module.n_heads, seq_len, seq_len, module.head_dim))
            self.ad_seq_lens.append(seq_len)
        return hook

    def _multihead_attn_hook(self, tag):
        def hook(module, inputs, output):
            query, key = inputs[0], inputs[1]
            batch, q_len, dim = query.shape
            kv_len = key.shape[1]
            head_dim = getattr(module, 'head_dim', None) or dim // module.num_heads
            # Count the whole attention block analytically: recent torch versions
            # take a fused fast path where in_proj/out_proj bypass module hooks.
            in_proj = 2 * (q_len + 2 * kv_len) * dim * dim
            out_proj = 2 * q_len * dim * dim
            attn = _attn_flops(batch, module.num_heads, q_len, kv_len, head_dim)
            self.add(tag, in_proj + out_proj + attn)
        return hook

    def _cross_attn_hook(self, tag):
        def hook(module, inputs, output):
            queries, context = inputs[0], inputs[1]
            batch, n_queries, _ = queries.shape
            ctx_len = context.shape[1]
            self.add(tag, _attn_flops(batch, module.n_heads, n_queries, ctx_len, module.head_dim))
            self.compress_context_lens.append(ctx_len)
        return hook

    def attach(self, root, tag, exclude=()):
        """Register counting hooks on every matmul-bearing module under root.

        Modules in `exclude` (and their descendants) are skipped; attach them
        separately under their own tag to get a per-part decomposition.
        """
        if root is None:
            return
        skipped = set()
        for module in exclude:
            if module is not None:
                skipped.update(module.modules())
        # MultiheadAttention projections are counted analytically by its hook
        # (the fused fast path bypasses child module hooks); do not also hook
        # its descendants (in_proj/out_proj) as plain Linears.
        mha_modules = [m for m in root.modules()
                       if isinstance(m, nn.MultiheadAttention) and m not in skipped]
        for module in mha_modules:
            handle = module.register_forward_hook(self._multihead_attn_hook(tag))
            self._handles.append(handle)
            skipped.update(module.modules())
        for module in root.modules():
            if module in skipped:
                continue
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(self._linear_hook(tag))
            elif isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(self._conv_hook(tag))
            elif type(module).__name__ == 'CausalSelfAttention':
                handle = module.register_forward_hook(self._causal_self_attn_hook(tag))
            elif type(module).__name__ == 'CrossAttentionLayer':
                handle = module.register_forward_hook(self._cross_attn_hook(tag))
            else:
                continue
            self._handles.append(handle)

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
