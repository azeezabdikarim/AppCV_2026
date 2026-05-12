#!/usr/bin/env bash
set -euo pipefail

BOARD=${1:-}
NUM_IMAGES=${2:-25}

if [[ -z "${BOARD}" ]]; then
  echo "Usage: bash capture.sh <checker|charuco> [num_images]" >&2
  exit 1
fi

if [[ "${BOARD}" != "checker" && "${BOARD}" != "charuco" ]]; then
  echo "Unsupported board '${BOARD}'. Use 'checker' or 'charuco'." >&2
  exit 1
fi

if command -v rpicam-still >/dev/null 2>&1; then
  CAMERA_CMD="rpicam-still"
elif command -v libcamera-still >/dev/null 2>&1; then
  CAMERA_CMD="libcamera-still"
else
  echo "Neither rpicam-still nor libcamera-still was found." >&2
  exit 1
fi

OUT_DIR="data/${BOARD}"
mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/img_*.jpg

for i in $(seq -w 0 $((NUM_IMAGES - 1))); do
  echo "Capturing frame $i..."
  "$CAMERA_CMD" -n \
    --width 640 \
    --height 480 \
    --timeout 100 \
    -o "${OUT_DIR}/img_${i}.jpg"
  sleep 0.1
done

count=$(find "${OUT_DIR}" -maxdepth 1 -name 'img_*.jpg' | wc -l | tr -d ' ')
echo "Captured $count images into ${OUT_DIR}/"
