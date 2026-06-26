"""f4 KL-VAE tokenizer for Zelda frames (64x64x3 -> [Cz, 32, 32] continuous latents).

Identical block design to kl_vae.py (the proven f8 tokenizer) but with ONE
stride-2 stage instead of two, so the latent stays at 32x32 (f4) rather than 16x16
(f8). Motivation: the resolution probe (resolution_probe.py) showed the f8 VAE's
normal-play Link presence ceiling (0.354) is fully explained by the 16x16 spatial
size -- a plain 64->16->64 bilinear round-trip scores the same (0.346). A 32x32
spatial round-trip scores 0.576 (+0.22). So f4 is the lever for the ~3px sprite.

Kept as a SEPARATE module so the f8 KLVAE (load-bearing for load_kl_vae, the WM
dataset, and every existing eval) is untouched. Same encode/decode/reparam/forward
/kl_loss interface as KLVAE.

NOTE: the f4 latent is [Cz,32,32] = 4x the floats of f8 [Cz,16,16]. That matters
only for a future WM retrain (gated behind the ceiling proof), not for this
tokenizer's reconstruction ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KLVAEF4Config:
    in_channels: int = 3
    z_channels: int = 16            # Cz; 16 floats per spatial cell
    hidden: Tuple[int, int] = (64, 128)
    spatial: int = 32               # 64 -> 32 (single stride-2 stage)
    beta: float = 1e-4              # KL weight


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class EncoderF4(nn.Module):
    def __init__(self, cfg: KLVAEF4Config):
        super().__init__()
        h1, h2 = cfg.hidden
        self.stem = nn.Conv2d(cfg.in_channels, h1, 3, padding=1)
        self.down1 = nn.Conv2d(h1, h1, 4, stride=2, padding=1)   # 64 -> 32
        self.res1a = ResBlock(h1)
        self.res1b = ResBlock(h1)
        self.proj1 = nn.Conv2d(h1, h2, 1)
        self.res2a = ResBlock(h2)                                # extra capacity at 32x32
        self.res2b = ResBlock(h2)
        self.norm_out = nn.GroupNorm(8, h2)
        self.head = nn.Conv2d(h2, 2 * cfg.z_channels, 1)         # mean+logvar

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.res1a(h); h = self.res1b(h)
        h = self.proj1(h)
        h = self.res2a(h); h = self.res2b(h)
        h = F.silu(self.norm_out(h))
        return self.head(h)


class DecoderF4(nn.Module):
    def __init__(self, cfg: KLVAEF4Config):
        super().__init__()
        h1, h2 = cfg.hidden
        self.proj = nn.Conv2d(cfg.z_channels, h2, 1)
        self.res2a = ResBlock(h2)                                # 32x32
        self.res2b = ResBlock(h2)
        self.up1 = nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1)   # 32 -> 64
        self.res1a = ResBlock(h1)
        self.res1b = ResBlock(h1)
        self.norm_out = nn.GroupNorm(8, h1)
        self.head = nn.Conv2d(h1, cfg.in_channels, 3, padding=1)

    def forward(self, z):
        h = self.proj(z)
        h = self.res2a(h); h = self.res2b(h)
        h = self.up1(h)
        h = self.res1a(h); h = self.res1b(h)
        h = F.silu(self.norm_out(h))
        h = self.head(h)
        return torch.tanh(h)   # outputs in [-1, 1]


class KLVAEF4(nn.Module):
    def __init__(self, cfg: KLVAEF4Config | None = None):
        super().__init__()
        self.cfg = cfg or KLVAEF4Config()
        self.encoder = EncoderF4(self.cfg)
        self.decoder = DecoderF4(self.cfg)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        return mean, logvar

    @staticmethod
    def reparam(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample: bool = True):
        mean, logvar = self.encode(x)
        z = self.reparam(mean, logvar) if sample else mean
        recon = self.decode(z)
        return recon, mean, logvar, z

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.flatten(1).sum(dim=1).mean()


def latent_floats_per_frame(cfg: KLVAEF4Config | None = None) -> int:
    cfg = cfg or KLVAEF4Config()
    return cfg.z_channels * cfg.spatial * cfg.spatial
