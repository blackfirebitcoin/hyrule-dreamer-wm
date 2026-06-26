#!/usr/bin/env bash
# Download the hyrule-dreamer-wm model weights from the GitHub Release.
# Weights are not stored in git (too large); they are Release assets.
#
# Usage:  bash weights/fetch_weights.sh
#
# Edit RELEASE_BASE to point at your published release tag.
set -euo pipefail

RELEASE_BASE="${RELEASE_BASE:-https://github.com/blackfirebitcoin/hyrule-dreamer-wm/releases/download/v1.0}"
cd "$(dirname "$0")"

fetch () {
  local name="$1"
  if [ -f "$name" ]; then echo "  have $name"; return; fi
  echo "  downloading $name ..."
  curl -fL "$RELEASE_BASE/$name" -o "$name"
}

fetch hyrule_dreamer_wm.pt   # 70 MB — ego world model (EMA weights)
fetch f4_ego_tokenizer.pt    # 6 MB  — KL-VAE tokenizer

echo "done. weights in $(pwd)"
