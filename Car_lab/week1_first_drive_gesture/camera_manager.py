#!/usr/bin/env python3

import subprocess
import threading
import time

import cv2
import numpy as np


class CameraManager:
    """Background camera capture for Pi 5/libcamera with OpenCV fallback."""

    def __init__(self, width=320, height=240, fps=10):
        self.width = width
        self.height = height
        self.fps = fps
        self.current_frame = None
        self.running = False
        self.process = None
        self.thread = None
        self.method = "not_started"
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        if self._libcamera_available():
            if self._run_libcamera_loop():
                return

        if self._run_opencv_loop():
            return

        self._placeholder_loop("Camera unavailable")

    def _libcamera_available(self):
        try:
            result = subprocess.run(["which", "libcamera-vid"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def _run_libcamera_loop(self):
        cmd = [
            "libcamera-vid",
            "--timeout", "0",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--output", "-",
            "--codec", "mjpeg",
            "--inline",
            "-n",
        ]

        try:
            self.method = "libcamera"
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            print(f"libcamera-vid startup failed: {exc}")
            return False

        buffer = b""
        while self.running and self.process.poll() is None:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk

            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")
            if start == -1 or end == -1 or end <= start:
                continue

            jpeg_data = buffer[start:end + 2]
            buffer = buffer[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with self._lock:
                    self.current_frame = frame

        return False

    def _run_opencv_loop(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                return False

            self.method = "opencv"
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            while self.running:
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self.current_frame = cv2.resize(frame, (self.width, self.height))
                time.sleep(1.0 / max(1, self.fps))

            cap.release()
            return True
        except Exception as exc:
            print(f"OpenCV camera fallback failed: {exc}")
            return False

    def _placeholder_loop(self, message):
        self.method = "placeholder"
        while self.running:
            with self._lock:
                self.current_frame = self._placeholder_frame(message)
            time.sleep(0.2)

    def _placeholder_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(frame, message, (16, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (8, self.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
        return frame

    def get_frame(self):
        with self._lock:
            if self.current_frame is None:
                return self._placeholder_frame("Waiting for camera")
            return self.current_frame.copy()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=1)
