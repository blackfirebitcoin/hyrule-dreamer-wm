"""CAUSAL sequence-level latent diffusion world model (src/seqwm_causal).

Parallel branch of src/seqwm_spread. The ONLY architectural change is that the
temporal self-attention is causal (frame t attends to frames <= t), which makes
the model samplable online via a rolling / receding-horizon loop. Front frames
act as a noised history window (ctx), later frames are diffused targets; loss on
targets only. The non-causal "spread future anchor" trick is deliberately NOT
available here -- this branch is the playable-direction prototype.

NOTE on evaluation: the single-shot eval last/first-target ratio will look like
the v3 (front-anchor) failure case for this branch, BECAUSE far targets cannot
see future context AND are far from the front history. That is EXPECTED and is
NOT the success metric. The verdict for the causal branch is FREE-RUNNING ROLLOUT
coherence (foreground motion recall over 1s/3s/10s with the rolling sampler),
measured by the rollout harness -- not the in-training clean-context eval.
"""
