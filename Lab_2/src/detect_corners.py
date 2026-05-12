#!/usr/bin/env python3
"""
Extract checkerboard or ChArUco corners from calibration images.

Usage:
    python3 src/detect_corners.py --board checker
    python3 src/detect_corners.py --board charuco

Output:
    Saves corner detections to captured_points/<board>/corners.npz.
"""

import argparse

import cv2
import numpy as np

from board_config import (
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CHECKERBOARD_INNER_CORNERS,
    board_calibration_images,
    board_corners_path,
    board_captured_points_dir,
    checkerboard_object_points,
    create_charuco_board,
    detect_charuco_board,
)


def detect_checkerboard(images):
    pattern = CHECKERBOARD_INNER_CORNERS
    objp = checkerboard_object_points()

    objpoints, imgpoints = [], []
    valid_files = []
    img_shape = None

    for image_path in images:
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(refined)
        valid_files.append(image_path.name)

    return {
        "valid_files": valid_files,
        "objpoints": np.stack(objpoints) if objpoints else np.empty((0, 0, 3), np.float32),
        "imgpoints": np.stack(imgpoints) if imgpoints else np.empty((0, 0, 1, 2), np.float32),
        "img_shape": img_shape,
    }


def detect_charuco(images):
    board = create_charuco_board()

    all_corners, all_ids = [], []
    valid_files = []
    img_shape = None
    expected_corners = (CHARUCO_SQUARES_X - 1) * (CHARUCO_SQUARES_Y - 1)

    for image_path in images:
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        marker_corners, marker_ids, charuco_corners, charuco_ids = detect_charuco_board(
            gray, board=board
        )
        if marker_ids is None or len(marker_ids) == 0:
            continue

        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_corners) != expected_corners
        ):
            continue

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        valid_files.append(image_path.name)

    return {
        "valid_files": valid_files,
        "corners": np.stack(all_corners) if all_corners else np.empty((0, 0, 1, 2), np.float32),
        "ids": np.stack(all_ids) if all_ids else np.empty((0, 0, 1), np.int32),
        "img_shape": img_shape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=["checker", "charuco"], required=True)
    args = parser.parse_args()

    images = sorted(board_calibration_images(args.board))
    if not images:
        raise SystemExit(f"No calibration images found in data/{args.board}/.")

    output_dir = board_captured_points_dir(args.board)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.board == "checker":
        result = detect_checkerboard(images)
    else:
        result = detect_charuco(images)

    out_path = board_corners_path(args.board)
    np.savez(out_path, **result)

    print(
        f"Corner detection complete. {len(result['valid_files'])} valid frames saved."
    )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
