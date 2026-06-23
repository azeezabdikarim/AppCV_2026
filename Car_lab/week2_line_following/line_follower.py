#!/usr/bin/env python3
"""Week 2: Line Following -- student implementation file.

This is the only file you edit. You implement the four image functions below
and they slot into the ``process()`` pipeline. Everything else -- the PID
controller, the debug overlay, and the web-server glue -- is provided in
``line_follower_base.py`` and ``debug_overlay.py`` and does not need to be
touched.

Pipeline:  frame -> ROI -> grayscale -> line center -> error -> steering

The placeholders below let the web UI run immediately; the car will not follow
the line until you replace them with real implementations.
"""

import cv2
import numpy as np

from week2_line_following.line_follower_base import LineFollowerBase


class LineFollower(LineFollowerBase):
    def __init__(self):
        super().__init__()

        # ================================================================
        # TUNABLE PARAMETERS -- fill these in to tune (or set them live in
        # the web UI). The car will not drive until they have values.
        # ================================================================
        # Region of interest, as fractions of the frame. Range [0, 1], where
        # 0 is the top/left edge and 1 is the bottom/right edge.
        self.roi_top = None       # top of ROI,    e.g. 0.6
        self.roi_bottom = None    # bottom of ROI, e.g. 0.9
        self.roi_left = None      # left of ROI,   e.g. 0.1
        self.roi_right = None     # right of ROI,  e.g. 0.9

        # Canny edge-detection thresholds. Pixel-gradient bounds in [0, 255],
        # with low < high.
        self.canny_low = None     # lower threshold, e.g. 100
        self.canny_high = None    # upper threshold, e.g. 150

        # PID gains (>= 0). Raise Kp until the car responds, then add Ki/Kd.
        self.Kp = None            # proportional gain, e.g. 0.8
        self.Ki = None            # integral gain,     e.g. 0.1
        self.Kd = None            # derivative gain,   e.g. 0.3

    def process(self, frame):
        """Run the pipeline: each step's output feeds the next.

        The values stored on ``self`` are the latest pipeline outputs; the
        provided debug overlay reads them to draw the ROI box, the centers,
        and the edges. You should not need to change this method.
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
        """Crop to the region of interest.

        TODO: use self.roi_top / roi_bottom / roi_left / roi_right (fractions
        of the frame) to compute pixel bounds, then crop with array slicing:
            frame[top:bottom, left:right]
        Return ``(roi, (top, bottom, left, right))``.
        """
        # Placeholder: return the whole frame so the pipeline still runs.
        height, width = frame.shape[:2]
        return frame, (0, height, 0, width)

    def convert_to_grayscale(self, roi):
        """Convert a BGR ROI to grayscale and return it.

        TODO: cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        """
        # Placeholder: no grayscale yet.
        return None

    def detect_line_center(self, gray):
        """Estimate the line center from edge pixels.

        TODO:
          1. edges = cv2.Canny(gray, self.canny_low, self.canny_high)
          2. ys, xs = np.where(edges > 0)
          3. if any edge pixels: center_x = int(np.mean(xs)),
             center_y = int(np.mean(ys))
        Return ``(center_x, center_y, edges)`` in ROI coordinates. Returning the
        edges enables the "Canny edges" debug view.
        """
        # Placeholder: no line detected yet.
        return None, None, None

    def calculate_error(self, roi, line_center_x):
        """Steering error in pixels.

        TODO:
          - image_center_x = roi.shape[1] // 2   (store it on self.image_center_x)
          - error = line_center_x - image_center_x
        Positive => line is right of center (steer right); negative => left.
        Return 0.0 when there is no line.
        """
        # Placeholder: keep the image center for the debug view, no error yet.
        self.image_center_x = (roi.shape[1] // 2) if (roi is not None and roi.size) else None
        return 0.0
