"""Model architectures.

One `HybridLM` skeleton — atomic-vocab embedding -> N pre-norm blocks
(RMSNorm -> token-mixer -> residual -> RMSNorm -> SwiGLU -> residual) -> final norm -> tied LM head
— parameterised by which token-mixer each layer uses, so the five architectures share everything
except the mixer:

  transformer : every layer RoPE causal attention        (softmax baseline)
  mamba2      : every layer fla.layers.Mamba2             (diagonal-SSM baseline)
  gdp_hybrid  : [GDP, GDP, attn, GDP]                     (the "both" hypothesis; n_h=4, neg-eig)
  gdn_hybrid  : [GDN, GDN, attn, GDN]                     (recall-only contrast)
  gru         : every layer GRU (pure torch)              (S_k extrapolation reference)

Recurrent mixers (Mamba-2/GDN/GDP) come from flash-linear-attention (Triton, bf16). Attention and
GRU are pure PyTorch. The hybrid attention sits at position 2 (attention_ratio=0.25, every_n).
`use_forget_gate` is a build_model flag (an ablation knob).

This module imports torch + fla, so it is NOT imported by `factworld/__init__.py` — the data/eval
layer stays usable without a GPU. Import it explicitly: `from factworld.models import build_model`.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers import GatedDeltaNet, GatedDeltaProduct, Mamba2

ARCHS = ("transformer", "mamba2", "gdp_hybrid", "gdn_hybrid", "gru", "fprm")
_HYBRID_RECURRENT = {"gdp_hybrid": "gdp", "gdn_hybrid": "gdn"}


def _filtered(cls, **want):
    """Construct an fla layer passing only the kwargs its __init__ actually accepts."""
    params = set(inspect.signature(cls.__init__).parameters)
    return cls(**{k: v for k, v in want.items() if k in params})


class FlaMixer(nn.Module):
    """Adapt an fla layer to forward(x) -> tensor (fla layers return a tuple)."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


class RoPECausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x):  # x: (B, h, T, hd)
        T = x.shape[-2]
        pos = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None]
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rot.float() * sin).to(x.dtype)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = self._rope(q), self._rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class GRUMixer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, mixer: nn.Module):
        super().__init__()
        self.n1 = nn.RMSNorm(d_model)
        self.mix = mixer
        self.n2 = nn.RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ff(self.n2(x))


def layer_plan(arch: str, n_layers: int, attention_ratio: float = 0.25) -> list[str]:
    """Per-layer mixer kind. Hybrids place attention by the every_n rule (idx 2 for 4 layers)."""
    if arch == "transformer":
        return ["attn"] * n_layers
    if arch == "mamba2":
        return ["mamba2"] * n_layers
    if arch == "gru":
        return ["gru"] * n_layers
    if arch == "gdp_pure":                  # attention-FREE GatedDeltaProduct (attention-confound control)
        return ["gdp"] * n_layers
    if arch == "gdn_pure":
        return ["gdn"] * n_layers
    if arch in _HYBRID_RECURRENT:
        plan = [_HYBRID_RECURRENT[arch]] * n_layers
        n_attn = max(1, round(n_layers * attention_ratio))
        for j in range(n_attn):
            plan[int((j + 0.5) * n_layers / n_attn)] = "attn"
        return plan
    raise ValueError(f"unknown arch {arch!r}")


def _make_mixer(kind, d_model, n_heads, head_dim, use_forget_gate, num_householder=4, allow_neg_eigval=True, use_short_conv=False):
    if kind == "attn":
        return RoPECausalAttention(d_model, n_heads)
    if kind == "gru":
        return GRUMixer(d_model)
    if kind == "mamba2":
        # Mamba-2 sizes its heads off expand*hidden_size (not the attention convention) —
        # let it self-configure from hidden_size rather than forcing num_heads*head_dim.
        return FlaMixer(_filtered(Mamba2, hidden_size=d_model))
    if kind == "gdn":
        return FlaMixer(_filtered(GatedDeltaNet, hidden_size=d_model, num_heads=n_heads,
                                  head_dim=head_dim, expand_v=1.0, use_short_conv=use_short_conv))
    if kind == "gdp":
        return FlaMixer(_filtered(GatedDeltaProduct, hidden_size=d_model, num_heads=n_heads,
                                  head_dim=head_dim, num_householder=num_householder,
                                  allow_neg_eigval=allow_neg_eigval,
                                  use_forget_gate=use_forget_gate, use_output_gate=True,
                                  use_short_conv=use_short_conv, expand_v=1.0))
    raise ValueError(kind)


class HybridLM(nn.Module):
    def __init__(self, arch: str, vocab_size: int, d_model: int = 320, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 1280, use_forget_gate: bool = True, tie_head: bool = True,
                 num_householder: int = 4, allow_neg_eigval: bool = True,
                 use_short_conv: bool = False, resid_init: bool = False):
        super().__init__()
        self.arch = arch
        head_dim = d_model // n_heads
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)  # sane tied-head logit scale at init
        self.layers_plan = layer_plan(arch, n_layers)
        self.blocks = nn.ModuleList(
            Block(d_model, d_ff, _make_mixer(k, d_model, n_heads, head_dim, use_forget_gate,
                                             num_householder, allow_neg_eigval, use_short_conv))
            for k in self.layers_plan
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_head:
            self.head.weight = self.embed.weight
        if resid_init:  # GPT-2 residual-scaled init on the residual-path output projections
            import math
            s = 1.0 / math.sqrt(2 * n_layers)
            for b in self.blocks:
                b.ff.w3.weight.data.mul_(s)
                if isinstance(b.mix, RoPECausalAttention):
                    b.mix.proj.weight.data.mul_(s)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))  # (B, T, vocab)

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:  # tied head/embed counted once
                seen.add(id(p))
                total += p.numel()
        return total


class CausalConv1d(nn.Module):
    """Depthwise causal 1-D convolution: pad on the left only, crop to input length."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=False
        )

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)                          # (B, C, T)
        x = F.pad(x, (self.conv.kernel_size[0] - 1, 0))  # causal left pad
        x = self.conv(x)
        x = x[..., :T]                                 # crop to input length
        return x.transpose(1, 2)                       # (B, T, C)


class FPRMBlock(nn.Module):
    """Single FPRM layer: pre-norm -> depthwise causal conv -> RoPE attention -> residual,
    then pre-norm -> SwiGLU -> residual.  This block is weight-tied across loops in FPRM."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.conv = CausalConv1d(d_model, kernel_size=3)
        self.attn = RoPECausalAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.conv(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class FPRM(nn.Module):
    """Fast Parallel Recurrent Model: embedding -> weight-tied FPRMBlock looped n_loops times
    (default n_loops = n_layers) -> final RMSNorm -> tied LM head."""

    def __init__(self, vocab_size: int, d_model: int = 320, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 1280, tie_head: bool = True,
                 n_loops: int | None = None):
        super().__init__()
        self.arch = "fprm"
        self.n_loops = n_loops if n_loops is not None else n_layers
        self.layers_plan = ["fprm"] * self.n_loops
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.block = FPRMBlock(d_model, n_heads, d_ff)
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_head:
            self.head.weight = self.embed.weight

    def forward(self, idx):
        x = self.embed(idx)
        for _ in range(self.n_loops):
            x = self.block(x)
        return self.head(self.norm(x))  # (B, T, vocab)

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:  # tied head/embed counted once
                seen.add(id(p))
                total += p.numel()
        return total


def build_model(arch: str, vocab_size: int, **kw):
    if arch == "fprm":
        fprm_kw = {k: v for k, v in kw.items() if k in {
            "d_model", "n_layers", "n_heads", "d_ff", "tie_head", "n_loops"
        }}
        return FPRM(vocab_size, **fprm_kw)
    return HybridLM(arch, vocab_size, **kw)
