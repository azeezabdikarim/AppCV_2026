#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-recordings}"
mkdir -p "$OUT"

record() {
  local name=$1
  echo ">>> ${name}: 640x480 @ 30 fps"
  rpicam-vid -t 5000 -n --width 640 --height 480 --framerate 30 -o "$OUT/${name}.h264"
  ffmpeg -y -loglevel error -framerate 30 -i "$OUT/${name}.h264" -c copy "$OUT/${name}.mp4"
  rm "$OUT/${name}.h264"
}

echo "Static clip: place the Pi flat on the table, point it at a still scene (no people, no motion)."
read -rp "Press Enter when ready: " _
record static

echo
echo "Dynamic clip: keep the Pi still, but have a person walk across the frame (or a hand move through it)."
read -rp "Press Enter when ready: " _
record dynamic

echo
ls -lh "$OUT/"
