#!/usr/bin/env python3

import cv2
import numpy as np
import time


class DebugVisualizer:
    """Render the debug view selected in the shared car UI."""

    def create_object_detection_debug_frame(
        self,
        original_frame,
        cache_manager,
        timing_utils,
        status_manager,
        sign_detector,
    ):
        """Draw Week 4 detection confidence and box-area decisions."""
        output = original_frame.copy()
        detections = cache_manager.cached_detections
        if detections:
            output = self._draw_detection_overlay(output, detections, sign_detector)

        status = status_manager.get_current_status(time.time())
        interval = getattr(timing_utils, "detection_interval", 0.5)
        scheduled_hz = 1.0 / interval if interval > 0 else 0.0
        threshold = getattr(sign_detector, "stop_area_ratio", 0.0)
        status_text = (
            f"{status} | detect {scheduled_hz:.1f} Hz / "
            f"{timing_utils.last_detection_inference_ms:.0f} ms | "
            f"stop area {100 * threshold:.1f}%"
        )
        cv2.putText(
            output,
            status_text,
            (8, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return output

    def create_speed_estimation_debug_frame(self, original_frame, flow_vectors, enabled):
        """Draw the latest optical-flow tracks without covering the feed."""
        debug_frame = original_frame.copy()
        if not enabled:
            return debug_frame

        for previous_point, current_point in flow_vectors:
            points = np.asarray([previous_point, current_point], dtype=np.float32)
            if points.shape != (2, 2) or not np.all(np.isfinite(points)):
                continue

            previous = tuple(np.rint(points[0]).astype(int))
            current = tuple(np.rint(points[1]).astype(int))
            cv2.arrowedLine(
                debug_frame,
                previous,
                current,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.3,
            )
            cv2.circle(debug_frame, current, 2, (0, 255, 0), -1, cv2.LINE_AA)
        return debug_frame

    def _draw_detection_overlay(self, frame, detections, sign_detector):
        stop_classes = getattr(sign_detector, "stop_classes", set())
        stop_threshold = getattr(sign_detector, "stop_area_ratio", 1.0)

        for detection in detections:
            x, y, width, height = detection["bbox"]
            class_name = detection.get("class_name", "object")
            confidence = detection.get("confidence", 0.0)
            area_ratio = detection.get(
                "area_ratio",
                (width * height) / float(frame.shape[0] * frame.shape[1]),
            )
            triggers_stop = class_name in stop_classes and area_ratio >= stop_threshold
            color = (0, 0, 255) if triggers_stop else (0, 200, 0)

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            label = f"{class_name} {confidence:.2f} area={100 * area_ratio:.1f}%"
            cv2.putText(
                frame,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame
