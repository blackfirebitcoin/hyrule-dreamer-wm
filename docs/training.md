# Training notes

> This is an **inference + weights** release: the training/capture **harness** and the
> training **data** (captured NES frames, derived from a copyrighted ROM) are **not**
> included. This document records the recipe behind the published weights — the model
> definition in `src/` is the architecture-of-record.

## Data

- **Ego-centric re-render.** Frames are re-centered on Link so the world scrolls around
  him (rather than the native fixed-screen camera). This makes action→motion a
  consistent translation the dynamics model can learn.
- **Route corpus.** First-room-rooted outward exploration with balanced coverage
  (overworld traversal, cave approach + interior). This is *coverage*, not learned
  spatial memory — persistence is explicitly out of scope.

## Two-stage, frozen-tokenizer discipline

1. **Train the KL-VAE tokenizer** to convergence; gate on reconstruction quality.
2. **Freeze it**, cache all frames to latents, then **train the world model** against
   those frozen latents. Never co-train a dynamics model and its own tokenizer — the
   latent geometry drifts and the dynamics weights become throwaway.

## World-model training

- EDM diffusion objective, per-frame action conditioning, causal temporal attention.
- Context frames conditioned at low fixed noise; targets on a lognormal schedule.
- The published checkpoint is the EMA weights at ~20k steps — the point where
  held-direction rollouts stay coherent for several seconds.

## What did *not* make it in

- A later **96px** iteration of this model traded resolution for stability and
  **collapsed faster** under closed-loop rollout — it is *not* the published checkpoint.
- Fine-tune branches on this 64px base (context-noise FT, generated-history recovery FT)
  did not beat the base on coherence, so the **base** is the released model.
- Drift-reduction work (self-forcing, distribution matching) and a real-time distilled
  descendant are ongoing and beyond this checkpoint.
