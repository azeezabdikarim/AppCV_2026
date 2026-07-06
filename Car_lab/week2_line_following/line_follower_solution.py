#!/usr/bin/env python3
"""Week 2: Line Following -- student implementation file (REFERENCE SOLUTION).

This is the only file you edit. You implement the four image functions and the
``process()`` pipeline that chains them together. Everything else -- the PID
controller, the debug overlay, and the web-server glue -- is provided in
``line_follower_base.py`` and ``debug_overlay.py`` and does not need to be
touched.

Pipeline:  frame -> ROI -> grayscale -> line center -> error -> steering
"""

import cv2
import numpy as np

from week2_line_following.line_follower_base import LineFollowerBase


class LineFollower(LineFollowerBase):
    def __init__(self):
        super().__init__()

        # ================================================================
        # TUNABLE PARAMETERS -- adjust here, or live in the web UI.
        # ================================================================
        # Region of interest, as fractions of the frame.
        self.roi_top = 0.6
        self.roi_bottom = 0.9
        self.roi_left = 0.1
        self.roi_right = 0.9

        # Canny edge-detection thresholds.
        self.canny_low = 100
        self.canny_high = 150

        # PID gains.
        self.Kp = 0.8
        self.Ki = 0.1
        self.Kd = 0.3

    def process(self, frame):
        """Run the pipeline: each step's output feeds the next.

        The values stored on ``self`` are the latest pipeline outputs; the
        provided debug overlay reads them to draw the ROI box, the centers,
        and the edges.
        """
        roi, self.roi_bounds = self.extract_roi(frame)
        gray = self.convert_to_grayscale(roi)
        self.line_center_x, self.line_center_y, self.edge_image = self.detect_line_center(gray)
        self.current_error = self.calculate_error(roi, self.line_center_x)
        return self.apply_pid(self.current_error)

    # ====================================================================
    # THE FOUR FUNCTIONS YOU IMPLEMENT
    # ====================================================================

    def extract_roi(self, frame):
        """Crop to the region of interest. Return ``(roi, bounds)`` where
        ``bounds`` is ``(top, bottom, left, right)`` in pixels."""
        height, width = frame.shape[:2]
        top = int(height * self.roi_top)
        bottom = int(height * self.roi_bottom)
        left = int(width * self.roi_left)
        right = int(width * self.roi_right)
        roi = frame[top:bottom, left:right]

        # add this import to the top 
        # from utils.console_logger import console_logger

        # then you can write something like 

        # console_logger.info(
        #      f"ROI bounds: top={top}, 
        #       bottom={bottom}, left={left}, right={right}")
        return roi, (top, bottom, left, right)

    def convert_to_grayscale(self, roi):
        """Convert a BGR ROI to grayscale. Return the gray image."""
        if roi is None or roi.size == 0:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def detect_line_center(self, gray):
        """Estimate the line center from edge pixels. Return
        ``(center_x, center_y, edges)`` in ROI coordinates."""
        if gray is None or gray.size == 0:
            return None, None, None
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        ys, xs = np.where(edges > 0)
        if xs.size == 0:
            return None, None, edges
        return int(np.mean(xs)), int(np.mean(ys)), edges

    def calculate_error(self, roi, line_center_x):
        """Steering error in pixels: ``line_center_x - image_center_x``.

        Positive => line is right of center (steer right); negative => left.
        """
        if roi is None or roi.size == 0:
            self.image_center_x = None
            return 0.0
        self.image_center_x = roi.shape[1] // 2
        if line_center_x is None:
            return 0.0
        return line_center_x - self.image_center_x
