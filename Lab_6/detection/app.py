#!/usr/bin/env python3
"""Standalone ONNX sign detector for Lab 6 model verification."""

import argparse
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template


LOGGER = logging.getLogger("lab6_detector")
DEFAULT_CLASSES = ["Stop_Sign", "TU_Logo", "Stahp", "Falling_Cows"]

app = Flask(__name__)
camera = None
detector = None


class YOLODetector:
    """Run a raw Ultralytics YOLOv8 ONNX detection model."""

    def __init__(self, model_path, class_names, confidence=0.5, nms_iou=0.45):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is missing. Activate the app_cv conda environment."
            ) from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        options = ort.SessionOptions()
        options.inter_op_num_threads = 2
        options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape
        self.class_names = list(class_names)
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)

        height = self.input_shape[2] if len(self.input_shape) == 4 else None
        width = self.input_shape[3] if len(self.input_shape) == 4 else None
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                f"The course detector requires a fixed BCHW model input; got {self.input_shape}"
            )
        self.input_height = height
        self.input_width = width

        LOGGER.info("Loaded %s", path)
        LOGGER.info("Input: %s %s", self.input_name, self.input_shape)

    def _preprocess(self, frame):
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )
        resized_width = int(round(frame_width * scale))
        resized_height = int(round(frame_height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height))

        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.full(
            (self.input_height, self.input_width, 3), 114, dtype=np.uint8
        )
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def detect(self, frame):
        tensor, scale, pad_x, pad_y = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(output, frame.shape, scale, pad_x, pad_y)

    def _decode(self, output, frame_shape, scale, pad_x, pad_y):
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            raise ValueError(f"Unexpected model output shape: {output.shape}")

        expected_columns = 4 + len(self.class_names)
        if predictions.shape[0] == expected_columns:
            predictions = predictions.T
        if predictions.shape[1] < expected_columns:
            raise ValueError(
                f"Output has {predictions.shape[1]} values per candidate; "
                f"expected at least {expected_columns}"
            )

        class_scores = predictions[:, 4:expected_columns]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        keep = confidences >= self.confidence

        predictions = predictions[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        if not len(predictions):
            return []

        frame_height, frame_width = frame_shape[:2]
        boxes = []
        for center_x, center_y, width, height in predictions[:, :4]:
            x = (center_x - width / 2 - pad_x) / scale
            y = (center_y - height / 2 - pad_y) / scale
            box_width = width / scale
            box_height = height / scale

            x = max(0.0, min(float(x), frame_width - 1.0))
            y = max(0.0, min(float(y), frame_height - 1.0))
            box_width = max(0.0, min(float(box_width), frame_width - x))
            box_height = max(0.0, min(float(box_height), frame_height - y))
            boxes.append([int(x), int(y), int(box_width), int(box_height)])

        selected = cv2.dnn.NMSBoxes(
            boxes,
            confidences.astype(float).tolist(),
            self.confidence,
            self.nms_iou,
        )
        selected = np.asarray(selected).reshape(-1) if len(selected) else []

        detections = []
        frame_area = float(frame_width * frame_height)
        for index in selected:
            x, y, width, height = boxes[int(index)]
            class_id = int(class_ids[int(index)])
            detections.append(
                {
                    "bbox": [x, y, width, height],
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": float(confidences[int(index)]),
                    "area_ratio": (width * height) / frame_area,
                }
            )
        return detections

    @staticmethod
    def draw(frame, detections):
        output = frame.copy()
        for detection in detections:
            x, y, width, height = detection["bbox"]
            is_stop = detection["class_name"] in {"Stop_Sign", "Stahp"}
            color = (0, 0, 255) if is_stop else (0, 200, 0)
            cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
            label = (
                f'{detection["class_name"]} '
                f'{detection["confidence"]:.2f} '
                f'{100 * detection["area_ratio"]:.1f}%'
            )
            cv2.putText(
                output,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return output


class CameraStream:
    """Read MJPEG frames from rpicam/libcamera in one background thread."""

    def __init__(self, width=320, height=240, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.running = False
        self.process = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        command = next(
            (name for name in ("rpicam-vid", "libcamera-vid") if shutil.which(name)),
            None,
        )
        if command is None:
            raise RuntimeError("rpicam-vid/libcamera-vid was not found")

        args = [
            command,
            "--timeout",
            "0",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.fps),
            "--codec",
            "mjpeg",
            "--inline",
            "--nopreview",
            "--output",
            "-",
        ]
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )

        buffer = b""
        while self.running and self.process.poll() is None:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start < 0 or end < 0:
                continue
            jpeg = buffer[start : end + 2]
            buffer = buffer[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=1)


def frame_generator():
    last_detection = 0.0
    cached_detections = []
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_detection >= 0.5:
            cached_detections = detector.detect(frame)
            last_detection = now
        display = detector.draw(frame, cached_detections)
        ok, encoded = cv2.imencode(".jpg", display)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template(
        "index.html",
        input_shape=detector.input_shape,
        confidence=detector.confidence,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument(
        "--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated names"
    )
    return parser.parse_args()


def main():
    global camera, detector
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    detector = YOLODetector(
        args.model,
        class_names,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
    )
    camera = CameraStream()
    camera.start()
    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()

