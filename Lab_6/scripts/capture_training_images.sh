#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-train}"
NUM_IMAGES="${2:-20}"
DELAY="${3:-0.5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/capture_training_images.py" \
  --session "${SESSION}" \
  --num-images "${NUM_IMAGES}" \
  --delay "${DELAY}"

