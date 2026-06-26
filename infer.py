#!/usr/bin/env python3
"""
hyrule-dreamer-wm — turnkey inference.

Load the ego world model + KL-VAE tokenizer, seed it with a short clip of real
latent frames, then dream forward under a scripted action sequence and write an
MP4. The world model never sees pixels: it rolls forward in the tokenizer's
latent space (receding-horizon, causal), and the tokenizer decodes each latent
frame to RGB for the video.

Example — walk DOWN for 6s, then UP for 6s:

    python infer.py \
        --wm weights/hyrule_dreamer_wm.pt \
        --tokenizer weights/f4_ego_tokenizer.pt \
        --seed assets/seeds/overworld_00.pt \
        --actions "down:6,up:6" --fps 10 \
        --out out/walk_down_up.mp4

Action tokens: noop=0 A=1 B=2 select=3 start=4 UP=5 DOWN=6 LEFT=7 RIGHT=8
A --actions script is a comma list of `name:seconds` segments; frame count per
segment = seconds * fps. The first --n-ctx frames are the real seed (context);
everything after is closed-loop generation under your script.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import torch

# repo root on sys.path so `from src...` resolves when run from anywhere
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seqwm_causal.model import SeqWM, SeqWMConfig          # noqa: E402
from src.seqwm_causal import diffusion as dfn                  # noqa: E402
from src.seqwm_causal.solvers import sample_clip_solver        # noqa: E402
from src.tokenizer.kl_vae_f4 import KLVAEF4, KLVAEF4Config      # noqa: E402

ACTION = {"noop": 0, "a": 1, "b": 2, "select": 3, "start": 4,
          "up": 5, "down": 6, "left": 7, "right": 8}


def load_wm(ckpt_path: Path, device, weights: str = "ema"):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfgd = dict(sd["cfg"])
    causal_in_ckpt = "causal" in cfgd
    fields = {f.name for f in dataclasses.fields(SeqWMConfig)}
    cfg = SeqWMConfig(**{k: v for k, v in cfgd.items() if k in fields})
    if not causal_in_ckpt:
        cfg.causal = False
    model = SeqWM(cfg).to(device).eval()
    state = sd.get("ema", sd.get("model")) if weights == "ema" else sd.get("model", sd.get("ema"))
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[wm] {ckpt_path.name} step={sd.get('step','?')} weights={weights} "
          f"causal={cfg.causal} h={cfg.h} w={cfg.w} cz={cfg.cz}", flush=True)
    return model, cfg


def load_tokenizer(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfgd = ck.get("cfg", {})
    cfg = KLVAEF4Config(z_channels=int(cfgd.get("z_channels", 16)),
                        spatial=int(cfgd.get("spatial", 32)),
                        beta=float(cfgd.get("beta", 1e-4)))
    m = KLVAEF4(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    print(f"[tok] {ckpt_path.name} step={ck.get('step','?')} "
          f"z={cfg.z_channels} spatial={cfg.spatial}", flush=True)
    return m


@torch.no_grad()
def decode_latents(tok, lat: torch.Tensor, device, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, lat.shape[0], bs):
        rec = tok.decode(lat[i:i + bs].to(device)).clamp(-1, 1)
        out.append((((rec + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()))
    return np.transpose(np.concatenate(out, 0), (0, 2, 3, 1)).copy()  # -> [T,H,W,C]


def parse_actions(script: str, fps: int) -> list[int]:
    seq: list[int] = []
    for seg in script.split(","):
        name, _, secs = seg.strip().partition(":")
        name = name.strip().lower()
        if name not in ACTION:
            raise SystemExit(f"unknown action '{name}' (choose from {list(ACTION)})")
        seq += [ACTION[name]] * max(1, round(float(secs) * fps))
    return seq


@torch.no_grad()
def rollout(model, cfg, edm, ctx_lat: torch.Tensor, actions: torch.Tensor,
            n_ctx: int, num_steps: int, sigma_max: float, solver: str,
            device, seed: int) -> torch.Tensor:
    """Causal receding-horizon rollout: keep n_ctx context frames, generate one
    frame at a time under the action window, slide forward."""
    target_len = int(actions.shape[0])
    frames = [ctx_lat[i].to(device) for i in range(n_ctx)]
    cstart = 0
    cuda = device.type == "cuda"
    while len(frames) < target_len:
        need = n_ctx + 1
        a_win = actions[cstart:cstart + need]
        if a_win.shape[0] < need:
            a_win = torch.cat([a_win, a_win[-1:].expand(need - a_win.shape[0])], 0)
        ctx = torch.stack(frames[-n_ctx:], 0).unsqueeze(0)
        gen = torch.Generator(device=device).manual_seed(seed + cstart)
        ctx_noise = torch.randn(1, n_ctx, cfg.cz, cfg.h, cfg.w, device=device, generator=gen)
        tgt_noise = torch.randn(1, 1, cfg.cz, cfg.h, cfg.w, device=device, generator=gen)
        with torch.autocast(device_type="cuda", enabled=cuda):
            full = sample_clip_solver(model, ctx, a_win.unsqueeze(0).to(device), cfg, edm,
                                      n_ctx, num_steps, sigma_max, device, solver,
                                      horizon=1, ctx_noise=ctx_noise, tgt_noise=tgt_noise)
        frames.append(full[0, n_ctx].float())
        cstart += 1
        if cstart % 50 == 0:
            print(f"  rollout {len(frames)}/{target_len}", flush=True)
    return torch.stack(frames[:target_len], 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", required=True, type=Path, help="ego96 world-model checkpoint")
    ap.add_argument("--tokenizer", required=True, type=Path, help="f4-96 tokenizer checkpoint")
    ap.add_argument("--seed", required=True, type=Path, help="seed clip (.pt with obs_latent, act)")
    ap.add_argument("--actions", required=True, help='e.g. "down:30,up:30"')
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=int, default=3, help="nearest-neighbour upscale for the video")
    ap.add_argument("--n-ctx", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--sigma-max", type=float, default=8.0)
    ap.add_argument("--sigma-data", type=float, default=0.179)
    ap.add_argument("--sigma-ctx", type=float, default=0.05)
    ap.add_argument("--solver", choices=["heun", "dpmpp2m"], default="heun")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--seed-rng", type=int, default=20260615)
    args = ap.parse_args()

    import imageio.v2 as imageio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip = torch.load(args.seed, map_location="cpu", weights_only=False)
    seed_lat = clip["obs_latent"].float()
    seed_act = clip["act"].long()

    held = parse_actions(args.actions, args.fps)
    actions = torch.full((len(held),), held[0], dtype=torch.long)
    actions[:] = torch.tensor(held, dtype=torch.long)
    # context frames keep the seed's own (shifted) actions; script takes over after
    shifted_seed = dfn.shift_actions(seed_act.unsqueeze(0))[0]
    actions[:args.n_ctx] = shifted_seed[:args.n_ctx]

    model, cfg = load_wm(args.wm, device, args.weights)
    tok = load_tokenizer(args.tokenizer, device)
    edm = dfn.EDMConfig(sigma_data=args.sigma_data, sigma_ctx=args.sigma_ctx)

    print(f"[rollout] {len(held)} frames ({args.actions}) @ {args.fps}fps "
          f"= {len(held)/args.fps:.0f}s, solver={args.solver}{args.num_steps}", flush=True)
    lat = rollout(model, cfg, edm, seed_lat, actions, args.n_ctx,
                  args.num_steps, args.sigma_max, args.solver, device, args.seed_rng)
    frames = decode_latents(tok, lat, device)
    if args.scale > 1:
        frames = np.repeat(np.repeat(frames, args.scale, 1), args.scale, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, list(frames), fps=args.fps, codec="libx264",
                    quality=8, macro_block_size=1)
    print(f"[done] wrote {args.out}  ({len(frames)} frames, {frames.shape[1]}x{frames.shape[2]})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
