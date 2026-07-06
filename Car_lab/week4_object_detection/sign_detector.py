#!/usr/bin/env python3
"""Instructor solution for the Week 4 object-detection exercise."""

import numpy as np

from week4_object_detection.sign_detector_base import SignDetectorBase


class SignDetector(SignDetectorBase):
    def __init__(self, model_path=None):
        super().__init__(
            model_path=model_path,
            confidence_threshold=0.05,
            nms_iou_threshold=0.45,
        )
        # Week 4 stops for any of the four custom signs.
        self.stop_classes = set(self.class_names)
        self.stop_area_ratio = 0.05

    def detect_signs(self, camera_frame):
        raw_output, transform = self.run_model(camera_frame)
        predictions = self.prediction_rows(raw_output)
        return self._filter_predictions(predictions, camera_frame.shape, transform)

    def _filter_predictions(self, predictions, frame_shape, transform):
        detections = []
        frame_height, frame_width = frame_shape[:2]
        frame_area = float(frame_width * frame_height)

        for prediction in predictions:
            class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.confidence_threshold:
                continue

            bbox = self.model_box_to_frame(
                prediction[:4],
                frame_shape,
                transform,
            )
            _, _, width, height = bbox
            if width <= 0 or height <= 0:
                continue

            detections.append(
                {
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": confidence,
                    "area_ratio": (width * height) / frame_area,
                }
            )

        return self.apply_nms(detections)

    def should_stop(self, detected_signs, camera_frame=None):
        return any(
            detection["class_name"] in self.stop_classes
            and detection["area_ratio"] >= self.stop_area_ratio
            for detection in detected_signs
        )
