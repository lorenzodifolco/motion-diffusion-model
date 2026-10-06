"""LoRA layers for MDM's nn.TransformerEncoder (torch 1.7.1).

nn.MultiheadAttention packs q/k/v into `in_proj_weight` (3d x d) and its forward passes
`in_proj_weight` and `out_proj.weight` straight to F.multi_head_attention_forward, so wrapping
`out_proj` as a module has no effect and `peft` cannot target it; torch 1.7.1 also has no
torch.nn.utils.parametrize. LoRAMultiheadAttention therefore keeps the frozen base module and calls
F.multi_head_attention_forward itself with merged weights W0 + (alpha/r) * B @ A.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """Low-rank update delta W = scale * B @ A, A: (r, in), B: (out, r). B = 0 at init -> delta W = 0."""

    def __init__(self, in_features, out_features, r, alpha):
        super().__init__()
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init as the reference LoRA impl.

    def delta(self):
        return (self.lora_B @ self.lora_A) * self.scale


class LoRALinear(nn.Module):
    """y = base(x) + scale * B A dropout(x); base is frozen."""

    def __init__(self, base: nn.Linear, r, alpha, dropout=0.0):
        super().__init__()
        self.base = base
        self.adapter = LoRAAdapter(base.in_features, base.out_features, r, alpha).to(base.weight.device, base.weight.dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        a = self.adapter
        return self.base(x) + F.linear(F.linear(self.dropout(x), a.lora_A), a.lora_B) * a.scale


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for a (self-attention) nn.MultiheadAttention with LoRA on q, k, v and out_proj.

    targets: subset of {'q', 'k', 'v', 'o'}; each projection gets its own rank-r adapter.
    """

    def __init__(self, base: nn.MultiheadAttention, r, alpha, targets=('q', 'k', 'v', 'o')):
        super().__init__()
        assert base._qkv_same_embed_dim and base.bias_k is None and not base.add_zero_attn
        self.base = base
        d = base.embed_dim
        self.targets = tuple(targets)
        self.adapters = nn.ModuleDict({t: LoRAAdapter(d, d, r, alpha) for t in self.targets}).to(base.in_proj_weight.device,
                                                                                                  base.in_proj_weight.dtype)

    def merged_weights(self):
        w_in = self.base.in_proj_weight
        if any(t in self.adapters for t in 'qkv'):
            d = self.base.embed_dim
            zero = w_in.new_zeros(d, d)
            delta = torch.cat([self.adapters[t].delta() if t in self.adapters else zero for t in 'qkv'], 0)
            w_in = w_in + delta
        w_out = self.base.out_proj.weight
        if 'o' in self.adapters:
            w_out = w_out + self.adapters['o'].delta()
        return w_in, w_out

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        b = self.base
        w_in, w_out = self.merged_weights()
        return F.multi_head_attention_forward(
            query, key, value, b.embed_dim, b.num_heads,
            w_in, b.in_proj_bias,
            b.bias_k, b.bias_v, b.add_zero_attn,
            b.dropout, w_out, b.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
