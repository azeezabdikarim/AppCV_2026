#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-recordings}"
mkdir -p "$OUT"

record() {
  local name=$1 W=$2 H=$3 FPS=$4
  echo ">>> ${name}: ${W}x${H} @ ${FPS} fps"
  rpicam-vid -t 5000 -n --width "$W" --height "$H" --framerate "$FPS" -o "$OUT/${name}.h264"
  ffmpeg -y -loglevel error -framerate "$FPS" -i "$OUT/${name}.h264" -c copy "$OUT/${name}.mp4"
  rm "$OUT/${name}.h264"
}

record m1_320_4    320  240   4
record m2_320_30   320  240  30
record m3_640_4    640  480   4
record m4_640_30   640  480  30
record m5_1280_4  1280  720   4
record m6_1280_30 1280  720  30

echo
ls -lh "$OUT/"
