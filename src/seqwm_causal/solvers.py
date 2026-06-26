#!/usr/bin/env python3
"""EDM-compatible samplers for the short-T inference path.
Inference-only, NO training. Heun (reference) vs DPM-Solver++(2M), a multistep
data-prediction solver. Same EDM rho=7 sigma schedule for all (fair). NFE
accounting: Heun = 2*N-1 (skips final correction at sigma=0); DPM++(2M) = N
(one model eval per step, reuses previous x0 estimate via Lagrange extrapolation).

DPM++(2M) update (k-diffusion sample_dpmpp_2m, verified): with t=-log(sigma),
h=t_next-t, sigma ratio = exp(-h); denoised_d = (1+1/2r)D - (1/2r)D_prev where
r = h_last/h; x = exp(-h)*x - expm1(-h)*denoised_d.
"""
from __future__ import annotations
import torch
from src.seqwm_causal import diffusion as dfn


def heun_sigmas(num_steps: int, sigma_max: float, edm_cfg, device) -> torch.Tensor:
    """EDM Heun sigma schedule -- identical to the production sample_clip."""
    rho, smin, smax = 7.0, edm_cfg.sigma_min, float(sigma_max)
    i = torch.arange(num_steps, device=device, dtype=torch.float64)
    t = (smax ** (1 / rho) + i / (num_steps - 1)
         * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return torch.cat([t, torch.zeros(1, device=device, dtype=torch.float64)]).to(torch.float32)


@torch.no_grad()
def sample_clip_solver(model, ctx_latents, act, cfg, edm_cfg, n_ctx, num_steps,
                       sigma_max, device, solver, gen=None, horizon=None,
                       ctx_noise=None, tgt_noise=None):
    B = ctx_latents.shape[0]
    Cz, H, W = cfg.cz, cfg.h, cfg.w
    horizon = (cfg.t - n_ctx) if horizon is None else int(horizon)
    T = n_ctx + horizon
    if T > cfg.t:
        raise ValueError(f"T={T} exceeds cfg.t={cfg.t}")
    if act.shape[1] < T:
        raise ValueError(f"act length {act.shape[1]} < T={T}")
    act_T = act[:, :T].contiguous()
    sd = edm_cfg.sigma_data
    t = heun_sigmas(num_steps, sigma_max, edm_cfg, device)

    if ctx_noise is None:
        ctx_noise = torch.randn(ctx_latents.shape, device=device, generator=gen)
    ctx_noisy = ctx_latents + edm_cfg.sigma_ctx * ctx_noise.to(device)
    if tgt_noise is None:
        tgt_noise = torch.randn(B, horizon, Cz, H, W, device=device, generator=gen)
    x = tgt_noise[:, :horizon].to(device) * t[0]

    def D_of(x_tgt, sig):
        full = torch.cat([ctx_noisy, x_tgt], dim=1)
        sigma = torch.empty(B, T, device=device)
        sigma[:, :n_ctx] = edm_cfg.sigma_ctx
        sigma[:, n_ctx:] = sig
        return dfn.denoise(model, full, sigma, act_T, sd)[:, n_ctx:]

    if solver == "heun":
        for k in range(num_steps):
            sc, sn = t[k], t[k + 1]
            D = D_of(x, sc)
            d = (x - D) / sc
            xn = x + (sn - sc) * d
            if sn > 0:
                D2 = D_of(xn, sn)
                d2 = (xn - D2) / sn
                x = x + (sn - sc) * 0.5 * (d + d2)
            else:
                x = xn
    elif solver == "dpmpp2m":
        old = None
        for k in range(num_steps):
            sc, sn = t[k], t[k + 1]
            D = D_of(x, sc)
            if sn <= 0:                      # final step -> x0 estimate
                x = D
                old = D
                continue
            tc = -torch.log(sc)
            tn = -torch.log(sn)
            h = tn - tc
            if old is None:                  # first step: 1st-order (DPM++ EMA)
                x = (sn / sc) * x - torch.expm1(-h) * D
            else:
                h_last = tc - (-torch.log(t[k - 1]))
                r = h_last / h
                Dd = (1 + 1 / (2 * r)) * D - (1 / (2 * r)) * old
                x = (sn / sc) * x - torch.expm1(-h) * Dd
            old = D
    else:
        raise ValueError(f"unknown solver {solver}")
    return torch.cat([ctx_latents, x], dim=1)


def nfe_of(solver: str, num_steps: int) -> int:
    return 2 * num_steps - 1 if solver == "heun" else num_steps


@torch.no_grad()
def heun_trace(model, ctx_latents, act, cfg, edm_cfg, n_ctx, num_steps, sigma_max,
               device, horizon=1, ctx_noise=None, tgt_noise=None):
    """Heun sampler that, at each step, logs the EMBEDDED local-error proxy:
    the disagreement between the 1st-order (Euler) and 2nd-order (Heun-corrected)
    update, ||x_heun - x_euler|| normalized by ||x_heun||. This is the standard
    embedded error estimate an adaptive ODE controller would key on. Also logs
    per-step sigma and update magnitude ||x_{i+1}-x_i||. Returns (final ctx+x,
    trace) where trace[i] = {sigma, update_mag, embedded_err, embedded_err_rel}.
    Used by the NFE-difficulty study (B8). NO training."""
    B = ctx_latents.shape[0]
    Cz, H, W = cfg.cz, cfg.h, cfg.w
    T = n_ctx + horizon
    act_T = act[:, :T].contiguous()
    sd = edm_cfg.sigma_data
    t = heun_sigmas(num_steps, sigma_max, edm_cfg, device)
    if ctx_noise is None:
        ctx_noise = torch.randn(ctx_latents.shape, device=device)
    ctx_noisy = ctx_latents + edm_cfg.sigma_ctx * ctx_noise.to(device)
    if tgt_noise is None:
        tgt_noise = torch.randn(B, horizon, Cz, H, W, device=device)
    x = tgt_noise[:, :horizon].to(device) * t[0]

    def D_of(x_tgt, sig):
        full = torch.cat([ctx_noisy, x_tgt], dim=1)
        sigma = torch.empty(B, T, device=device)
        sigma[:, :n_ctx] = edm_cfg.sigma_ctx
        sigma[:, n_ctx:] = sig
        return dfn.denoise(model, full, sigma, act_T, sd)[:, n_ctx:]

    trace = []
    for k in range(num_steps):
        sc, sn = t[k], t[k + 1]
        D = D_of(x, sc)
        d = (x - D) / sc
        x_euler = x + (sn - sc) * d
        if sn > 0:
            D2 = D_of(x_euler, sn)
            d2 = (x_euler - D2) / sn
            x_heun = x + (sn - sc) * 0.5 * (d + d2)
            emb = (x_heun - x_euler).flatten(1).norm(dim=1)          # [B] per-example
            ref = x_heun.flatten(1).norm(dim=1) + 1e-9
            x_next = x_heun
        else:
            emb = torch.zeros(B, device=device)
            ref = x_euler.flatten(1).norm(dim=1) + 1e-9
            x_next = x_euler
        upd = (x_next - x).flatten(1).norm(dim=1)
        trace.append({"sigma": float(sc),
                      "update_mag": upd.detach().cpu().tolist(),
                      "embedded_err": emb.detach().cpu().tolist(),
                      "embedded_err_rel": (emb / ref).detach().cpu().tolist()})
        x = x_next
    return torch.cat([ctx_latents, x], dim=1), trace


@torch.no_grad()
def dpmpp2m_trace(model, ctx_latents, act, cfg, edm_cfg, n_ctx, num_steps, sigma_max,
                  device, horizon=1, ctx_noise=None, tgt_noise=None):
    """DPM++(2M) sampler with a SOLVER-NATIVE difficulty/local-error proxy (NOT
    Heun's embedded estimator, which is invalid for adaptive DPM++).
    Per step logs: x0_change_rel = ||D_i - D_{i-1}|| / ||D_i|| (how much the
    data-prediction is still moving = trajectory not yet converged), and
    correction_rel = ||denoised_d - D_i|| / ||D_i|| (magnitude of the 2nd-order
    Lagrange correction the multistep solver applies = local curvature/stiffness).
    Both are the natural quantities a DPM++ adaptive controller would key on.
    Returns (ctx+x, trace)."""
    B = ctx_latents.shape[0]
    Cz, H, W = cfg.cz, cfg.h, cfg.w
    T = n_ctx + horizon
    act_T = act[:, :T].contiguous()
    sd = edm_cfg.sigma_data
    t = heun_sigmas(num_steps, sigma_max, edm_cfg, device)
    if ctx_noise is None:
        ctx_noise = torch.randn(ctx_latents.shape, device=device)
    ctx_noisy = ctx_latents + edm_cfg.sigma_ctx * ctx_noise.to(device)
    if tgt_noise is None:
        tgt_noise = torch.randn(B, horizon, Cz, H, W, device=device)
    x = tgt_noise[:, :horizon].to(device) * t[0]

    def D_of(x_tgt, sig):
        full = torch.cat([ctx_noisy, x_tgt], dim=1)
        sigma = torch.empty(B, T, device=device)
        sigma[:, :n_ctx] = edm_cfg.sigma_ctx
        sigma[:, n_ctx:] = sig
        return dfn.denoise(model, full, sigma, act_T, sd)[:, n_ctx:]

    trace = []
    old = None
    for k in range(num_steps):
        sc, sn = t[k], t[k + 1]
        D = D_of(x, sc)
        dn = D.flatten(1).norm(dim=1) + 1e-9
        x0_change = (torch.zeros(B, device=device) if old is None
                     else (D - old).flatten(1).norm(dim=1))
        if sn <= 0:
            corr = torch.zeros(B, device=device)
            x_next = D
        else:
            tc = -torch.log(sc); tn = -torch.log(sn); h = tn - tc
            if old is None:
                Dd = D
            else:
                r = (tc - (-torch.log(t[k - 1]))) / h
                Dd = (1 + 1 / (2 * r)) * D - (1 / (2 * r)) * old
            corr = (Dd - D).flatten(1).norm(dim=1)
            x_next = (sn / sc) * x - torch.expm1(-h) * Dd
        trace.append({"sigma": float(sc),
                      "update_mag": (x_next - x).flatten(1).norm(dim=1).detach().cpu().tolist(),
                      "x0_change_rel": (x0_change / dn).detach().cpu().tolist(),
                      "correction_rel": (corr / dn).detach().cpu().tolist()})
        old = D
        x = x_next
    return torch.cat([ctx_latents, x], dim=1), trace
