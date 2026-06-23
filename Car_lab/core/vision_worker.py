#!/usr/bin/env python3

import threading
import time

import cv2


class CoreVisionWorker:
    """Runs the shared car vision/control loop once and caches JPEG output."""

    def __init__(self, camera, robot, jpeg_quality=75):
        self.camera = camera
        self.robot = robot
        self.jpeg_quality = jpeg_quality

        self.running = False
        self.thread = None
        self.latest_jpeg = None
        self.frame_id = 0
        self.fps = 0.0
        self.error = ""
        self.last_frame_time = 0.0

        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_fps_time = time.time()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            start = time.time()

            try:
                frame = self.camera.get_frame()
                processed_frame = self.robot.process_autonomous_frame(frame)

                ok, buffer = cv2.imencode(
                    ".jpg",
                    processed_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if ok:
                    with self._lock:
                        self.latest_jpeg = buffer.tobytes()
                        self.frame_id += 1
                        self.last_frame_time = time.time()
                        self.error = ""
            except Exception as exc:
                with self._lock:
                    self.error = f"vision loop error: {exc}"

            self._update_fps()

            elapsed = time.time() - start
            target_fps = self._target_fps()
            time.sleep(max(0.0, (1.0 / target_fps) - elapsed))

    def _target_fps(self):
        return max(1, min(15, int(getattr(self.robot, "target_fps", 10))))

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            with self._lock:
                self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    def get_jpeg(self):
        with self._lock:
            return self.frame_id, self.latest_jpeg

    def get_status(self):
        with self._lock:
            last_frame_time = self.last_frame_time
            return {
                "vision_fps": round(self.fps, 1),
                "vision_target_fps": self._target_fps(),
                "vision_frame_id": self.frame_id,
                "vision_frame_age": round(time.time() - last_frame_time, 2)
                if last_frame_time
                else None,
                "vision_error": self.error,
                "jpeg_quality": self.jpeg_quality,
            }

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
