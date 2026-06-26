# Architecture

## Tokenizer — KL-VAE

A convolutional KL-regularized autoencoder. It maps a 64×64 ego-centric RGB frame to a
`32×32×16` latent grid and decodes it back. The world model operates **only** on these
latents; the tokenizer is trained first and then **frozen** — latent-space dynamics
models are invalid if the tokenizer keeps moving underneath them.

## World model — ego causal space-time DiT

A factorized space-time transformer over a clip of `T` latent frames. Each spatial cell
of a frame is a token; per block:

1. **spatial self-attention** within a frame (all cells see each other),
2. **temporal self-attention** across frames — **causal**: frame *t* attends only to
   frames ≤ *t*,
3. MLP, with pre-norm + AdaLN-Zero modulation conditioned on the per-frame EDM noise
   level.

Causal temporal attention is the key choice: it makes the model **online-samplable**
(rolling / receding-horizon) at inference, trading away the non-causal "see a future
anchor" trick that earlier non-causal experiments relied on.

**Conditioning added to tokens:** learned spatial pos-emb, learned temporal pos-emb,
and a per-frame **action embedding** (NES controller byte → vector), added to all
spatial tokens of its frame.

## Diffusion — EDM

Training and sampling use the EDM formulation. Context frames are held at a low fixed
noise (`sigma_ctx=0.05`); target frames are denoised from a lognormal noise schedule.
Inference defaults: `sigma_data=0.179` (the tokenizer's latent scale), `sigma_max=8.0`,
Heun solver, 20 steps, `n_ctx=8`.

## Rollout

```
frames = first n_ctx seed latents
while len(frames) < target:
    window  = last n_ctx frames
    action  = action[t]                 # your controller input
    next    = denoise(window, action)   # EDM sampler, horizon = 1
    frames.append(next)                 # feed back in
```

The fed-back context is the model's own output — this closed loop is where drift /
exposure bias lives, and is the central research problem of the project.
