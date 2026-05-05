#!/usr/bin/env bash
set -euo pipefail

NUM_IMAGES=${1:-25}

if command -v rpicam-still >/dev/null 2>&1; then
  CAMERA_CMD="rpicam-still"
elif command -v libcamera-still >/dev/null 2>&1; then
  CAMERA_CMD="libcamera-still"
else
  echo "Neither rpicam-still nor libcamera-still was found." >&2
  exit 1
fi

mkdir -p data
rm -f data/img_*.jpg

for i in $(seq -w 0 $((NUM_IMAGES - 1))); do
  echo "Capturing frame $i..."
  "$CAMERA_CMD" -n \
    --width 640 \
    --height 480 \
    --timeout 100 \
    -o "data/img_${i}.jpg"
  sleep 0.1
done

count=$(find data -maxdepth 1 -name 'img_*.jpg' | wc -l | tr -d ' ')
echo "Captured $count images into data/"
