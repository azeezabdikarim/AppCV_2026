#!/usr/bin/env python3
"""Provided ONNX and geometry helpers for the Week 4 student exercise."""

from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASS_NAMES = ["Stop_Sign", "TU_Logo", "Stahp", "Falling_Cows"]


class SignDetectorBase:
    """Own model loading, preprocessing, geometry conversion and NMS."""

    def __init__(
        self,
        model_path=None,
        class_names=None,
        confidence_threshold=0.5,
        nms_iou_threshold=0.45,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is missing. Activate the app_cv conda environment."
            ) from exc

        module_dir = Path(__file__).resolve().parent
        self.model_path = Path(model_path or module_dir / "models" / "best.onnx")
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.inter_op_num_threads = 2
        options.intra_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape

        if len(self.input_shape) != 4:
            raise ValueError(f"Expected BCHW model input, got {self.input_shape}")
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        if not isinstance(self.input_height, int) or not isinstance(self.input_width, int):
            raise ValueError(
                f"Week 4 requires a fixed-size ONNX input, got {self.input_shape}"
            )

    def run_model(self, camera_frame):
        """Return raw model output and the letterbox transform."""
        input_tensor, transform = self._preprocess(camera_frame)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        return output, transform

    def prediction_rows(self, raw_output):
        """Normalize YOLO output to one `[cx, cy, w, h, scores...]` row per box."""
        predictions = np.squeeze(raw_output)
        if predictions.ndim != 2:
            raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

        expected_columns = 4 + len(self.class_names)
        if predictions.shape[0] == expected_columns:
            predictions = predictions.T
        if predictions.shape[1] < expected_columns:
            raise ValueError(
                f"Model output has {predictions.shape[1]} columns; "
                f"expected at least {expected_columns}"
            )
        return predictions[:, :expected_columns]

    def model_box_to_frame(self, model_box, frame_shape, transform):
        """Convert model `[cx, cy, w, h]` to clipped frame `[x, y, w, h]`."""
        center_x, center_y, width, height = [float(value) for value in model_box]
        scale, pad_x, pad_y = transform
        frame_height, frame_width = frame_shape[:2]

        x = (center_x - width / 2 - pad_x) / scale
        y = (center_y - height / 2 - pad_y) / scale
        width /= scale
        height /= scale

        x = max(0.0, min(x, frame_width - 1.0))
        y = max(0.0, min(y, frame_height - 1.0))
        width = max(0.0, min(width, frame_width - x))
        height = max(0.0, min(height, frame_height - y))
        return [int(x), int(y), int(width), int(height)]

    def apply_nms(self, detections):
        """Remove duplicate overlapping detections."""
        if not detections:
            return []
        boxes = [detection["bbox"] for detection in detections]
        confidences = [detection["confidence"] for detection in detections]
        selected = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_iou_threshold,
        )
        indices = np.asarray(selected).reshape(-1) if len(selected) else []
        return [detections[int(index)] for index in indices]

    def _preprocess(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Camera frame is empty")

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
            (self.input_height, self.input_width, 3),
            114,
            dtype=np.uint8,
        )
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), (scale, pad_x, pad_y)

