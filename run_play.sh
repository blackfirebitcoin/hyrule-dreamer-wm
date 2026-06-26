#!/usr/bin/env bash
# Launch the live play harness on http://localhost:9300
# Needs a CUDA GPU for smooth (~10fps) play. Extra args pass through to play.py.
set -e
cd "$(dirname "$0")"
python play.py \
  --wm weights/hyrule_dreamer_wm.pt \
  --tokenizer weights/f4_ego_tokenizer.pt \
  --seeds-dir assets/seeds --n-seeds 3 \
  --host 127.0.0.1 --port 9300 "$@"
