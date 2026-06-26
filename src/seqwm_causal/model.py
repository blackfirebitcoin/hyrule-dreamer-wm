"""Factorized space-time transformer (DiT-style) for the sequence-level latent
world model -- CAUSAL variant (src/seqwm_causal).

Identical to src/seqwm_spread/model.py except the TEMPORAL self-attention is
optionally causal (frame t attends only to frames <= t). Spatial attention stays
full (all S tokens within a frame see each other). Causality is what makes the
model samplable online (rolling/receding-horizon), at the cost of the non-causal
"see-the-future-anchor" trick that spread relied on. cfg.causal toggles it so the
same code can reproduce the non-causal baseline.

Shapes flowing through forward (B=batch, T=frames, S=H*W=256, d=d_model):
  x:      [B, T, Cz, H, W]
  tokens: [B, T, S, d]
  out:    [B, T, Cz, H, W]

Action conditioning (cfg.action_conditioning):
  "additive" (default, baseline): act_emb(act) is ADDED to all S spatial tokens.
      One vector competes with content+pos across 1024 tokens; AdaLN carries only
      noise level. This is the weak pathway the model can (and does) ignore.
  "token": prepend ONE action token per frame -> [B,T,1+S,d]. Spatial attention
      sees it within each frame; temporal attention sees the action-token row
      across frames. Output drops it (only the S patch tokens are decoded). This
      lets attention ROUTE the action to the spatial cells that need it.
  Default is "additive" so existing checkpoints load/behave identically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The flash SDPA kernel raises CUDA "invalid argument" for is_causal=True at
# head_dim 32 on this GB10/Blackwell GPU. Route the (small, L=T) causal temporal
# attention through the mem-efficient/math backend; spatial (non-causal, large L)
# keeps the default fast path. Backend selection only -- not an architecture change.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _CAUSAL_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    _HAVE_SDPA_KERNEL = True
except Exception:  # pragma: no cover - older torch
    _HAVE_SDPA_KERNEL = False


@dataclass
class SeqWMConfig:
    cz: int = 16
    h: int = 16
    w: int = 16
    t: int = 16
    d_model: int = 384
    n_blocks: int = 8
    n_heads: int = 6
    mlp_ratio: int = 4
    num_actions: int = 9
    fourier_dim: int = 128       # dim of fourier features for c_noise embedding
    causal: bool = True          # causal TEMPORAL attention (online-samplable)
    action_conditioning: str = "additive"   # "additive" | "token"
    action_adaln: bool = False   # if True: add zero-init per-action emb to the
                                 # AdaLN cond (action modulates every block). Exp2.
    patch_size: int = 1          # PxP latent pixels per spatial token.
                                 # 1 preserves the original parameter contract;
                                 # 2 maps a 32x32 latent grid to S=256.


# ----------------------------------------------------------------------------
# Conditioning: fourier features of c_noise -> MLP -> conditioning vector.
# ----------------------------------------------------------------------------
class FourierFeatures(nn.Module):
    """Random-frequency fourier features (fixed), as used by EDM/DiT for scalar
    noise conditioning. Input [...], output [..., dim] (dim even)."""

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(dim // 2) * scale, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., None] * self.freqs * 2 * math.pi
        return torch.cat([x.sin(), x.cos()], dim=-1)


class NoiseEmbedding(nn.Module):
    """c_noise [B,T] -> conditioning vector [B,T,d_model] for AdaLN."""

    def __init__(self, fourier_dim: int, d_model: int):
        super().__init__()
        self.ff = FourierFeatures(fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ff(c_noise))            # [B,T,d_model]


# ----------------------------------------------------------------------------
# Multi-head self-attention (batched). Operates on [N, L, d]: attends over L.
# is_causal applies a lower-triangular mask over L (used for the temporal axis).
# ----------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        N, L, d = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # [3, N, heads, L, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # scaled_dot_product_attention handles fp16/flash + the causal mask.
        if is_causal and _HAVE_SDPA_KERNEL:
            with sdpa_kernel(_CAUSAL_SDPA_BACKENDS):   # avoid the broken flash causal kernel
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)  # [N, heads, L, hd]
        o = o.transpose(1, 2).reshape(N, L, d)
        return self.proj(o)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation. x [B,T,S,d]; shift/scale [B,T,1,d] (per-frame)."""
    return x * (1 + scale) + shift


# ----------------------------------------------------------------------------
# Factorized block: spatial attn, temporal attn, MLP — each AdaLN-Zero gated.
# Temporal attn is causal iff cfg.causal.
# ----------------------------------------------------------------------------
class FactorizedBlock(nn.Module):
    def __init__(self, cfg: SeqWMConfig):
        super().__init__()
        d = cfg.d_model
        self.causal = cfg.causal
        self.norm_s = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn_s = SelfAttention(d, cfg.n_heads)
        self.norm_t = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn_t = SelfAttention(d, cfg.n_heads)
        self.norm_m = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(d, cfg.mlp_ratio)
        # AdaLN-Zero: produce shift/scale/gate for each of the 3 sub-layers
        # from the per-frame conditioning vector. 9 = 3 sub-layers * 3 params.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        # zero-init the final layer so blocks start as identity (AdaLN-Zero).
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, B: int, T: int, S: int) -> torch.Tensor:
        # x: [B,T,S,d]; cond: [B,T,d] (per-frame conditioning)
        d = x.shape[-1]
        params = self.ada(cond)                      # [B,T,9d]
        (sh_s, sc_s, g_s,
         sh_t, sc_t, g_t,
         sh_m, sc_m, g_m) = params.chunk(9, dim=-1)
        # broadcast per-frame params over the S spatial tokens: [B,T,1,d]
        u = lambda p: p[:, :, None, :]

        # (1) spatial self-attention: attend over S tokens within each frame (full).
        h = modulate(self.norm_s(x), u(sh_s), u(sc_s))
        h = h.reshape(B * T, S, d)
        h = self.attn_s(h).reshape(B, T, S, d)
        x = x + u(g_s) * h

        # (2) temporal self-attention: attend over T frames, batched over B*S.
        #     causal iff self.causal -> frame t only attends to frames <= t.
        h = modulate(self.norm_t(x), u(sh_t), u(sc_t))
        h = h.permute(0, 2, 1, 3).reshape(B * S, T, d)   # [B*S, T, d]
        h = self.attn_t(h, is_causal=self.causal).reshape(B, S, T, d).permute(0, 2, 1, 3)
        x = x + u(g_t) * h

        # (3) MLP
        h = modulate(self.norm_m(x), u(sh_m), u(sc_m))
        h = self.mlp(h)
        x = x + u(g_m) * h
        return x


class SeqWM(nn.Module):
    def __init__(self, cfg: SeqWMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.patch_size = getattr(cfg, "patch_size", 1)
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >=1, got {self.patch_size}")
        if cfg.h % self.patch_size or cfg.w % self.patch_size:
            raise ValueError(
                f"h/w ({cfg.h},{cfg.w}) not divisible by patch_size "
                f"{self.patch_size}"
            )
        self.S = (cfg.h // self.patch_size) * (cfg.w // self.patch_size)
        self.action_conditioning = getattr(cfg, "action_conditioning", "additive")

        patch_dim = cfg.cz * self.patch_size * self.patch_size
        self.in_proj = nn.Linear(patch_dim, d)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.S, d))
        self.temporal_pos = nn.Parameter(torch.zeros(1, cfg.t, 1, d))
        self.act_emb = nn.Embedding(cfg.num_actions, d)
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)

        # action-token variant: one learned slot-position for the prepended action
        # token. Created ONLY in token mode so additive-mode state_dict is unchanged
        # (old checkpoints load identically).
        if self.action_conditioning == "token":
            self.act_token_pos = nn.Parameter(torch.zeros(1, 1, 1, d))
            nn.init.normal_(self.act_token_pos, std=0.02)
        elif self.action_conditioning != "additive":
            raise ValueError(f"unknown action_conditioning={self.action_conditioning!r}")

        self.noise_emb = NoiseEmbedding(cfg.fourier_dim, d)
        # Exp2: optional action-conditioned AdaLN. Zero-init -> identity at init
        # (cond == noise_emb only), so existing checkpoints load/behave identically.
        self.action_adaln = getattr(cfg, "action_adaln", False)
        if self.action_adaln:
            self.act_cond_emb = nn.Embedding(cfg.num_actions, d)
            nn.init.zeros_(self.act_cond_emb.weight)

        self.blocks = nn.ModuleList([FactorizedBlock(cfg) for _ in range(cfg.n_blocks)])

        # Output head with final AdaLN-Zero (standard DiT final layer).
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        self.out_proj = nn.Linear(d, patch_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """x [B,T,Cz,H,W], c_noise [B,T], act [B,T] -> F_pred [B,T,Cz,H,W]."""
        cfg = self.cfg
        B, T, Cz, H, W = x.shape
        P = self.patch_size
        if (Cz, H, W) != (cfg.cz, cfg.h, cfg.w):
            raise ValueError(
                f"latent shape mismatch: got {(Cz, H, W)}, "
                f"expected {(cfg.cz, cfg.h, cfg.w)}"
            )
        Hp, Wp = H // P, W // P
        S = Hp * Wp
        # Variable-length causal inference (Page 2026-06-13): allow T<=cfg.t.
        # Causal temporal attention makes positions [0,T) independent of the
        # dropped future slots, and temporal_pos[:, :T] reuses the first T
        # checkpoint embeddings (no weight resize). T>cfg.t is rejected.
        if T > cfg.t:
            raise ValueError(f"T={T} exceeds cfg.t={cfg.t}; sampler must request T<=cfg.t")

        # P=1 is algebraically identical to the original per-pixel tokenization.
        tok = x.reshape(B, T, Cz, Hp, P, Wp, P)
        tok = tok.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            B, T, S, Cz * P * P
        )
        tok = self.in_proj(tok)

        # add spatial + temporal embeddings to the patch tokens
        tok = tok + self.spatial_pos + self.temporal_pos[:, :T]

        if self.action_conditioning == "token":
            # prepend ONE action token per frame: [B,T,1+S,d]. Spatial attn sees it
            # within each frame; temporal attn sees the action-token row across frames
            # (it carries temporal_pos too). Output drops it -> only S patches decoded.
            act_tok = self.act_emb(act)[:, :, None, :]               # [B,T,1,d]
            act_tok = act_tok + self.act_token_pos + self.temporal_pos[:, :T]
            tok = torch.cat([act_tok, tok], dim=2)                   # [B,T,1+S,d]
            S_eff = S + 1
        else:  # additive (default): broadcast action embedding over all S tokens
            tok = tok + self.act_emb(act)[:, :, None, :]
            S_eff = S

        cond = self.noise_emb(c_noise)                 # [B,T,d]
        if self.action_adaln:
            cond = cond + self.act_cond_emb(act)       # zero-init: 0 at start

        for blk in self.blocks:
            tok = blk(tok, cond, B, T, S_eff)

        if self.action_conditioning == "token":
            tok = tok[:, :, 1:, :]                     # drop action token -> [B,T,S,d]

        # final AdaLN-Zero head
        sh, sc = self.ada_out(cond).chunk(2, dim=-1)
        tok = modulate(self.norm_out(tok), sh[:, :, None, :], sc[:, :, None, :])
        out = self.out_proj(tok)                       # [B,T,S,Cz*P*P]
        out = out.reshape(B, T, Hp, Wp, Cz, P, P)
        out = out.permute(0, 1, 4, 2, 5, 3, 6)
        out = out.reshape(B, T, Cz, H, W).contiguous()
        return out
