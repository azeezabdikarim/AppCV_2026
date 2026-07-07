#!/usr/bin/env python3
"""Student work for Week 4: filter model output and make a stop decision."""

import numpy as np

from week4_object_detection.sign_detector_base import SignDetectorBase


class SignDetector(SignDetectorBase):
    """Custom-sign detector used by the shared car controller."""

    def __init__(self, model_path=None):
        super().__init__(
            model_path=model_path,
            confidence_threshold=0.50,
            nms_iou_threshold=0.45,
        )

        # Student-tunable decision settings.
        # Week 4 stops for any of the four custom signs.
        self.stop_classes = set(self.class_names)
        self.stop_area_ratio = 0.05

    def detect_signs(self, camera_frame):
        """Run ONNX inference and return filtered detection dictionaries."""
        raw_output, transform = self.run_model(camera_frame)
        predictions = self.prediction_rows(raw_output)
        return self._filter_predictions(predictions, camera_frame.shape, transform)

    def _filter_predictions(self, predictions, frame_shape, transform):
        """Convert confident prediction rows into detection dictionaries.

        Return dictionaries with these fields:

        ```text
        bbox        [x, y, width, height]
        class_id    integer class index
        class_name  class name from self.class_names
        confidence  highest class confidence
        area_ratio  bounding-box area / complete frame area
        ```
        """
        detections = []
        frame_height, frame_width = frame_shape[:2]
        frame_area = float(frame_width * frame_height)

        # TODO 1: For every prediction row:
        #   - read the class scores from row[4:]
        #   - find the highest score and its class ID
        #   - reject scores below self.confidence_threshold
        #   - convert row[:4] with self.model_box_to_frame(...)
        #   - calculate area_ratio
        #   - append a detection dictionary in the format above

        # TODO 2: Apply self.apply_nms(detections) before returning.
        return detections

    def should_stop(self, detected_signs, camera_frame=None):
        """Return True when a target sign occupies enough of the image."""
        # TODO 3: Return True if any detection:
        #   - has a class_name in self.stop_classes, and
        #   - has area_ratio >= self.stop_area_ratio.
        return False
