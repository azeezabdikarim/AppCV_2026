#!/usr/bin/env bash
set -euo pipefail

NUM_IMAGES="${1:-80}"
DELAY="${2:-0.5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/capture_training_images.py" \
  --num-images "${NUM_IMAGES}" \
  --delay "${DELAY}"
