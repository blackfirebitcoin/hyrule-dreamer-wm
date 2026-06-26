# Seed clips

Each `.pt` is a short clip of **pre-encoded latent frames** (`obs_latent`) plus the recorded controller bytes (`act`) for those frames. `infer.py` uses the first `--n-ctx` frames as context to seed a dream; everything after is generated.

These are latent tensors (32×32×16) — not pixels and not the ROM. They start in the overworld.
