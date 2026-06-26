"""EDM (Karras et al. 2022) preconditioning + denoiser loss for the sequence-level
latent world model.

All sigma tensors are PER-FRAME: shape [B, T]. Preconditioning coefficients are
computed per-frame and broadcast over the latent (Cz, H, W) dims.

We use the numerically-stable reparameterized denoiser loss:
  - Model predicts F(c_in * x_noisy ; c_noise, cond).
  - Denoised estimate   D = c_skip * x_noisy + c_out * F.
  - EDM weighted loss    = lambda(sigma) * ||D - clean||^2  with lambda = 1/c_out^2.
  - Substituting D and lambda collapses to a plain MSE on F against the target
        F_target = (clean - c_skip * x_noisy) / c_out,
    which is exactly the stable form we optimize (no division-by-tiny-c_out blowup
    because the (1/c_out^2) weight and the c_out in D cancel analytically).

sigma_data is the measured std of the KL-VAE latents (0.1364).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EDMConfig:
    sigma_data: float = 0.1364
    # Lognormal sampling of target-frame sigmas (EDM training distribution).
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 2e-3
    sigma_max: float = 20.0
    # World-model context conditioning.
    sigma_ctx: float = 0.05    # fixed low noise on the n_ctx context frames
    # Exposure-bias training: instead of a single tiny fixed context sigma,
    # optionally sample each context frame's sigma from a lognormal so the model
    # learns to denoise targets from IMPERFECT context (matching the distribution
    # of its own self-generated context at rollout). "fixed" == original v1.
    ctx_noise_mode: str = "fixed"   # "fixed" | "lognormal"
    ctx_p_mean: float = -2.3        # exp(-2.3) ~= 0.10 mean context sigma
    ctx_p_std: float = 0.9
    ctx_sigma_max: float = 1.0      # clamp ceiling for context noise
    # Anchor placement: "front" = first n_ctx frames (v1/v2/v3); "spread" = n_ctx
    # anchors spaced evenly across the clip so no target is far from a clean
    # anchor (attacks intra-sequence far-from-anchor decay).
    context_mode: str = "front"     # "front" | "spread"


def context_indices(T: int, n_ctx: int, mode: str = "front"):
    """Frame indices that act as CONTEXT (anchors) for a T-frame clip.

    front:  [0, 1, ..., n_ctx-1]
    spread: n_ctx anchors spaced evenly across [0, T-1] inclusive of both ends.
    """
    n_ctx = max(0, min(n_ctx, T))
    if n_ctx == 0:
        return []
    if mode == "spread":
        if n_ctx == 1:
            return [0]
        return sorted({int(round(i * (T - 1) / (n_ctx - 1))) for i in range(n_ctx)})
    return list(range(n_ctx))


def edm_coeffs(sigma: torch.Tensor, sigma_data: float):
    """EDM preconditioning coefficients for a per-frame sigma tensor [B, T].

    Returns (c_in, c_skip, c_out, c_noise), each [B, T].
    """
    sd2 = sigma_data * sigma_data
    s2 = sigma * sigma
    denom = s2 + sd2
    c_in = 1.0 / torch.sqrt(denom)
    c_skip = sd2 / denom
    c_out = sigma * sigma_data / torch.sqrt(denom)
    c_noise = torch.log(sigma) / 4.0
    return c_in, c_skip, c_out, c_noise


def sample_sigma_lognormal(shape, cfg: EDMConfig, device, generator=None) -> torch.Tensor:
    """Sample target-frame sigmas from the EDM lognormal, clipped to [sigma_min, sigma_max]."""
    n = torch.randn(shape, device=device, generator=generator)
    sigma = torch.exp(cfg.p_mean + cfg.p_std * n)
    return sigma.clamp(cfg.sigma_min, cfg.sigma_max)


def build_clip_sigma(B: int, T: int, n_ctx: int, cfg: EDMConfig, device, generator=None):
    """Per-frame sigma [B, T] + context mask [T] (bool, True = anchor/context).

    Anchors placed per cfg.context_mode (front or spread) get context sigma
    (fixed sigma_ctx or lognormal exposure-bias noise); every other frame gets a
    lognormal target sigma. Returns (sigma, ctx_mask).
    """
    ctx_idx = context_indices(T, n_ctx, getattr(cfg, "context_mode", "front"))
    ctx_mask = torch.zeros(T, dtype=torch.bool, device=device)
    if ctx_idx:
        ctx_mask[torch.tensor(ctx_idx, device=device)] = True

    sigma = torch.empty(B, T, device=device)
    n_c = int(ctx_mask.sum().item())
    n_t = T - n_c
    if n_c > 0:
        if cfg.ctx_noise_mode == "lognormal":
            n = torch.randn((B, n_c), device=device, generator=generator)
            s = torch.exp(cfg.ctx_p_mean + cfg.ctx_p_std * n)
            sigma[:, ctx_mask] = s.clamp(cfg.sigma_min, cfg.ctx_sigma_max)
        else:
            sigma[:, ctx_mask] = cfg.sigma_ctx
    if n_t > 0:
        sigma[:, ~ctx_mask] = sample_sigma_lognormal((B, n_t), cfg, device, generator)
    return sigma, ctx_mask


def add_noise(clean: torch.Tensor, sigma: torch.Tensor, generator=None) -> torch.Tensor:
    """clean [B,T,Cz,H,W] + N(0, sigma^2) with per-frame sigma [B,T]."""
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    return clean + sigma[:, :, None, None, None] * noise


def shift_actions(act: torch.Tensor) -> torch.Tensor:
    """One-step action realignment: act_shifted[..., k] = act[..., k-1], with a
    no-op (0) at k=0.

    The stored 20fps act[k] is the OUTGOING action (it describes the edge k->k+1),
    but the model adds act_emb on z[k]'s own token and denoises z[k]; the action
    that PRODUCED z[k] is act[k-1]. Feeding act_shifted puts the producing action on
    the frame it produced (verified against ground-truth RAM displacement,
    diag_wm_actshift_reval_20260602: alignment flips fwd->bwd, margin 0.53).
    Operates on the last dim T of an [..., T] long tensor. Data/feeding-only; no
    architecture change. Apply UNIFORMLY in train, eval, sampler, and rollout.
    """
    shifted = torch.zeros_like(act)
    shifted[..., 1:] = act[..., :-1]
    return shifted


def denoise(model, x_noisy: torch.Tensor, sigma: torch.Tensor, act: torch.Tensor,
            sigma_data: float) -> torch.Tensor:
    """Run the model with EDM preconditioning; return denoised estimate D [B,T,Cz,H,W]."""
    c_in, c_skip, c_out, c_noise = edm_coeffs(sigma, sigma_data)
    b = lambda c: c[:, :, None, None, None]  # broadcast [B,T] -> [B,T,1,1,1]
    F_pred = model(b(c_in) * x_noisy, c_noise, act)
    return b(c_skip) * x_noisy + b(c_out) * F_pred


def edm_loss_per_frame(model, clean: torch.Tensor, sigma: torch.Tensor, act: torch.Tensor,
                       sigma_data: float, return_denoised: bool = False):
    """Reparameterized EDM denoiser loss, returned PER FRAME (no reduction over T).

    Returns:
      loss_tf : [B, T]  -- mean-squared error per frame (already the EDM-weighted
                            objective; equals lambda*||D-clean||^2 averaged over the
                            latent dims of that frame).
      x_noisy : [B,T,Cz,H,W] the noised input (handy for debugging).
    """
    c_in, c_skip, c_out, c_noise = edm_coeffs(sigma, sigma_data)
    b = lambda c: c[:, :, None, None, None]

    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype)
    x_noisy = clean + b(sigma) * noise

    F_pred = model(b(c_in) * x_noisy, c_noise, act)
    # Stable target: F should match (clean - c_skip*x_noisy)/c_out.
    F_target = (clean - b(c_skip) * x_noisy) / b(c_out)
    # MSE on F == lambda * ||D - clean||^2 with lambda = 1/c_out^2 (the c_out's cancel).
    se = (F_pred - F_target) ** 2                 # [B,T,Cz,H,W]
    loss_tf = se.mean(dim=(2, 3, 4))              # [B,T]
    if return_denoised:
        # EDM denoised estimate D = c_skip*x_noisy + c_out*F_pred (grad flows via F_pred).
        D = b(c_skip) * x_noisy + b(c_out) * F_pred
        return loss_tf, x_noisy, D
    return loss_tf, x_noisy
