# Lab 1b — Streaming (placeholder)

This folder holds the streaming code from the 2025 combined Lab 1, moved here because the 2026 three-lab arc splits that material into:

- **Lab 1a** — image as data (laptop-side notebook analysis). See `../Lab_1a/`.
- **Lab 1b** — introductory streaming (this folder).
- **Lab 2** — camera calibration and measurement.

See [`../../notes/04_labs_1_to_3_plan.md`](../../notes/04_labs_1_to_3_plan.md) for the full curricular plan.

## What's here

- `streamserver.py` — Flask + `picamera2` MJPEG server, lifted directly from 2025. Serves a single client over `http://cvpiXX.local:8000/video_feed` at 854×480, ~20 fps (hard-coded 50 ms sleep between frames). Renders an "no camera detected" placeholder if `Picamera2` fails to initialise.
- `templates/index.html` — minimal Flask template that displays the stream in a browser and provides a JavaScript snapshot button.

The code is functional as a *starting point*. A student who clones and runs it will see a live stream in their browser within minutes, which is the right first moment for a streaming lab.

## What's missing relative to the 2026 Lab 1b / Lab 2 plan

This file exists so that whoever revisits this folder — me in August, a TA, or a future-semester instructor — has a concrete list of deltas to close before this code is ready to hand to students.

**For Lab 1b (streaming intro) — session objectives per `notes/04_labs_1_to_3_plan.md` §2.1–2.7:**

1. **Instrumentation.** `streamserver.py` currently has no metrics. Add a frame counter and byte counter inside `generate_frames()` and a background thread (or a request hook) that prints "served N frames, M bytes in last second" once per second. This becomes the student's first source of truth for "what is my stream actually delivering". Plan §2.8.1 calls for this explicitly.
2. **Resolution / framerate as CLI args.** Right now the size is hard-coded to `(854, 480)` and the frame-rate cap is hard-coded to `time.sleep(0.05)`. For the three-configuration measurement (320×240 @ 4 fps, 640×480 @ 15 fps, 1280×720 @ 30 fps in plan §2.8.1), students need to change these between runs. Add `--width`, `--height`, `--fps` flags; drop the `time.sleep` in favour of a `1/fps` budget check.
3. **Local video recording exercises.** Plan §2.2–2.3 has the class record 5-second clips at six (resolution × framerate) combinations with `rpicam-vid`, then tabulate file size vs bitrate. No code artefact is strictly needed — this is command-line work on the Pi — but a small helper `scripts/record_matrix.sh` that loops over the six configs and saves clips to `recordings/` would save students time and make the deliverable reproducible.

**For the classroom stress test (plan §2.8.2):** no new code is required on the server side; it's a coordinated in-session exercise. The deliverable is a table of achieved-fps "alone" vs "all 12 streaming".

**For the optional frame-differencing extension (plan §2.9):** students modify `generate_frames()` to yield `abs(current_frame - prev_frame)` thresholded. Worth a second file (`streamserver_motion.py` or a `--mode motion` flag) so the baseline server stays clean for the main exercises.

**Cleanup items in the existing code before student handoff:**

- The HTML in `templates/index.html` references `stream_server.py` (snake\_case with underscore), but the actual file is `streamserver.py`. Pick one and make them consistent — I'd keep the filename `streamserver.py` and fix the template.
- The "Camera Controls" help text in the template lists `camera.resolution`, `camera.framerate`, `camera.hflip`, `camera.rotation` — those are the *old* `picamera` (v1) API. The code uses `picamera2`, which does not expose those attributes that way. Either rewrite the help to show the correct `picam2.create_video_configuration(...)` arguments, or drop that panel — it'll mislead students who try to follow it verbatim.
- The snapshot button is cute but not part of the curriculum. Safe to keep; not worth teaching.

## Required Pi-side packages

Per plan §0.2, Lab 1b is the lab that adds:

```bash
sudo apt install -y python3-opencv python3-picamera2 python3-flask ffmpeg
```

Current Raspberry Pi OS images already include the basic `rpicam-apps` in the base image.

These are **not** pre-installed on the course SD-card image. The install is done in session, staggered by row (odd rows first, even rows ~2 min later) to avoid saturating the CV-PI-NET uplink — which is itself on-message for a lab about bandwidth.

## Status

Code is 2025-vintage, drop-in functional, not yet adapted for the 2026 objectives. Do not hand this folder to students as-is for Lab 1b; the instrumentation in item 1 above is what makes the lab pedagogically honest.
