"""KL-VAE tokenizer for Zelda frames (64x64x3 -> [Cz, 16, 16] continuous latents).

Phase-2 latent-dynamics path. Design rationale:
  - 2 stride-2 conv stages so 64x64 -> 16x16 (preserve sprite detail; Link ~12px).
  - Continuous Gaussian latent with tiny KL beta (1e-4) — DIAMOND-friendly,
    avoids posterior collapse without over-regularizing.
  - Mean-only at inference; reparam sampling during training.
Outputs in [-1, 1] (data is normalized from uint8 to [-1,1] in dataloader).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KLVAEConfig:
    in_channels: int = 3
    z_channels: int = 16            # Cz; 16 floats per spatial cell
    hidden: Tuple[int, int] = (64, 128)
    spatial: int = 16               # 64 -> 32 -> 16 (two stride-2 stages)
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


class Encoder(nn.Module):
    def __init__(self, cfg: KLVAEConfig):
        super().__init__()
        h1, h2 = cfg.hidden
        self.stem = nn.Conv2d(cfg.in_channels, h1, 3, padding=1)
        self.down1 = nn.Conv2d(h1, h1, 4, stride=2, padding=1)   # 64 -> 32
        self.res1a = ResBlock(h1)
        self.res1b = ResBlock(h1)
        self.proj1 = nn.Conv2d(h1, h2, 1)
        self.down2 = nn.Conv2d(h2, h2, 4, stride=2, padding=1)   # 32 -> 16
        self.res2a = ResBlock(h2)
        self.res2b = ResBlock(h2)
        self.norm_out = nn.GroupNorm(8, h2)
        self.head = nn.Conv2d(h2, 2 * cfg.z_channels, 1)         # mean+logvar

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.res1a(h); h = self.res1b(h)
        h = self.proj1(h)
        h = self.down2(h)
        h = self.res2a(h); h = self.res2b(h)
        h = F.silu(self.norm_out(h))
        return self.head(h)


class Decoder(nn.Module):
    def __init__(self, cfg: KLVAEConfig):
        super().__init__()
        h1, h2 = cfg.hidden
        self.proj = nn.Conv2d(cfg.z_channels, h2, 1)
        self.res2a = ResBlock(h2)
        self.res2b = ResBlock(h2)
        self.up2 = nn.ConvTranspose2d(h2, h1, 4, stride=2, padding=1)   # 16 -> 32
        self.res1a = ResBlock(h1)
        self.res1b = ResBlock(h1)
        self.up1 = nn.ConvTranspose2d(h1, h1, 4, stride=2, padding=1)   # 32 -> 64
        self.norm_out = nn.GroupNorm(8, h1)
        self.head = nn.Conv2d(h1, cfg.in_channels, 3, padding=1)

    def forward(self, z):
        h = self.proj(z)
        h = self.res2a(h); h = self.res2b(h)
        h = self.up2(h)
        h = self.res1a(h); h = self.res1b(h)
        h = self.up1(h)
        h = F.silu(self.norm_out(h))
        h = self.head(h)
        # outputs in [-1, 1] via tanh
        return torch.tanh(h)


class KLVAE(nn.Module):
    def __init__(self, cfg: KLVAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or KLVAEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

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
        # KL(N(mean, var) || N(0, I)) summed over latent dims, mean over batch
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.flatten(1).sum(dim=1).mean()


def latent_floats_per_frame(cfg: KLVAEConfig | None = None) -> int:
    cfg = cfg or KLVAEConfig()
    return cfg.z_channels * cfg.spatial * cfg.spatial
