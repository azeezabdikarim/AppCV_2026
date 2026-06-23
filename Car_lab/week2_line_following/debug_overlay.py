#!/usr/bin/env python3
"""Debug overlay rendering for Week 2 line following (PROVIDED).

This file is provided and does not need to be edited. It draws the optional
debug views onto a copy of the camera frame. Which views are drawn is chosen
by a bitmask (the debug-view checkboxes in the web UI add up to this number):

    ROI box        = 1
    Center & error = 2
    Canny edges    = 4

Nothing selected (0) means no overlay. The error and steering values are shown
in the web UI sidebar, so they are never drawn on the video.
"""

import cv2

ROI_BOX = 1
CENTER_ERROR = 2
CANNY_EDGES = 4


def render_overlay(frame, debug_level, state):
    """Return a copy of ``frame`` with the selected debug overlays drawn.

    ``state`` is the LineFollower instance; we read the latest pipeline values
    it stored this frame (roi_bounds, image_center_x, line_center_x/y,
    edge_image).
    """
    debug_frame = frame.copy()
    bounds = getattr(state, "roi_bounds", None)
    has_bounds = isinstance(bounds, (tuple, list)) and len(bounds) == 4

    # ROI box (yellow).
    if (debug_level & ROI_BOX) and has_bounds:
        top, bottom, left, right = bounds
        cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(debug_frame, "ROI", (left, max(15, top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Center & error: image center (blue), line center (green), error line (red).
    if (debug_level & CENTER_ERROR) and has_bounds:
        top, bottom, left, _ = bounds
        center_global = None
        if state.image_center_x is not None:
            cgx, cgy = left + state.image_center_x, (top + bottom) // 2
            center_global = (cgx, cgy)
            cv2.circle(debug_frame, center_global, 8, (255, 0, 0), -1)
            cv2.putText(debug_frame, "IMG CENTER", (cgx - 40, cgy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        if state.line_center_x is not None and state.line_center_y is not None:
            lgx, lgy = left + state.line_center_x, top + state.line_center_y
            cv2.circle(debug_frame, (lgx, lgy), 8, (0, 255, 0), -1)
            cv2.putText(debug_frame, "LINE CENTER", (lgx - 40, lgy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if center_global is not None:
                cv2.line(debug_frame, center_global, (lgx, lgy), (0, 0, 255), 2)

    # Canny edges: overlay the raw edge map in magenta inside the ROI.
    edge_image = getattr(state, "edge_image", None)
    if (debug_level & CANNY_EDGES) and has_bounds and edge_image is not None:
        top, bottom, left, right = bounds
        roi_region = debug_frame[top:bottom, left:right]
        if roi_region.shape[:2] == edge_image.shape[:2]:
            roi_region[edge_image > 0] = (255, 0, 255)

    return debug_frame
