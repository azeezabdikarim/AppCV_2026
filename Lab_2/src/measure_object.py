#!/usr/bin/env python3
"""
Prototype measurement script with stronger board detection and a manual fallback.

If automatic board detection fails, the user can click the four outer board
corners to define the board plane manually. The measurement itself is still
done by clicking two points on the object.
"""

import argparse
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from board_config import (
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CHARUCO_MARKER_SIZE_M,
    CHECKERBOARD_INNER_CORNERS,
    CHECKERBOARD_SQUARES_X,
    CHECKERBOARD_SQUARES_Y,
    SQUARE_SIZE_M,
    board_intrinsics_path,
    charuco_object_points_for_ids,
    create_charuco_board,
    detect_charuco_board,
)


class DetectionError(RuntimeError):
    pass


def load_intrinsics(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        intr = yaml.safe_load(f)
    k = np.asarray(intr["K"]["data"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(intr["D"]["data"], dtype=np.float64).reshape(-1, 1)
    return k, d


def undistort_pixel_points(points, k, d):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, k, d, P=k).reshape(-1, 2)


def checker_inner_world_points(cols, rows, square_size_m):
    objp = np.zeros((rows * cols, 2), np.float32)
    objp[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def board_outer_world_points(board_name, square_size_m):
    if board_name == "checker":
        width = CHECKERBOARD_SQUARES_X * square_size_m
        height = CHECKERBOARD_SQUARES_Y * square_size_m
    else:
        width = CHARUCO_SQUARES_X * square_size_m
        height = CHARUCO_SQUARES_Y * square_size_m
    return np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float32,
    )


def preprocess_variants(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yield "gray", gray
    yield "equalized", cv2.equalizeHist(gray)
    yield "clahe", clahe.apply(gray)


def detect_checkerboard_auto(gray, cols, rows, square_size_m, debug=False):
    pattern = (cols, rows)
    attempts = []

    for label, image in preprocess_variants(gray):
        if hasattr(cv2, "findChessboardCornersSB"):
            sb_variants = [("sb", 0)]
            if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
                sb_variants.append(("sb_exhaustive", cv2.CALIB_CB_EXHAUSTIVE))
            if hasattr(cv2, "CALIB_CB_ACCURACY"):
                sb_variants.append(("sb_accuracy", cv2.CALIB_CB_ACCURACY))
            if hasattr(cv2, "CALIB_CB_EXHAUSTIVE") and hasattr(
                cv2, "CALIB_CB_ACCURACY"
            ):
                sb_variants.append(
                    (
                        "sb_exhaustive_accuracy",
                        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                    )
                )
            for name, flags in sb_variants:
                found, corners = cv2.findChessboardCornersSB(image, pattern, flags=flags)
                attempts.append(f"{label}:{name}={found}")
                if found:
                    return (
                        corners.reshape(-1, 2).astype(np.float32),
                        checker_inner_world_points(cols, rows, square_size_m),
                        f"{label}:{name}",
                    )

        legacy_variants = [
            ("legacy_adaptive", cv2.CALIB_CB_ADAPTIVE_THRESH),
            (
                "legacy_adaptive_fast",
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            ),
            ("legacy_default", 0),
        ]
        for name, flags in legacy_variants:
            found, corners = cv2.findChessboardCorners(image, pattern, flags=flags)
            attempts.append(f"{label}:{name}={found}")
            if found:
                cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                return (
                    corners.reshape(-1, 2).astype(np.float32),
                    checker_inner_world_points(cols, rows, square_size_m),
                    f"{label}:{name}",
                )

    if debug:
        print("Checkerboard detector attempts:")
        for item in attempts:
            print("  ", item)
    raise DetectionError(
        f"Checkerboard pattern {pattern} was not detected in the image."
    )


def detect_charuco_auto(gray, debug=False):
    board = create_charuco_board()
    attempts = []

    for label, image in preprocess_variants(gray):
        marker_corners, marker_ids, charuco_corners, charuco_ids = detect_charuco_board(
            image, board=board
        )
        marker_count = 0 if marker_ids is None else len(marker_ids)
        attempts.append(f"{label}:markers={marker_count}")
        if marker_ids is None or marker_count < 4:
            continue
        charuco_count = 0 if charuco_corners is None else len(charuco_corners)
        attempts.append(f"{label}:charuco={charuco_count}")
        if charuco_corners is None or charuco_ids is None or charuco_count < 4:
            continue
        objp = charuco_object_points_for_ids(board, charuco_ids)[:, :2]
        return (
            charuco_corners.reshape(-1, 2).astype(np.float32),
            objp.astype(np.float32),
            f"{label}:charuco={charuco_count}",
        )

    if debug:
        print("ChArUco detector attempts:")
        for item in attempts:
            print("  ", item)
    raise DetectionError("ChArUco markers or corners were not detected in the image.")


def prompt_manual_board_corners(img, board_name, square_size_m):
    world_points = board_outer_world_points(board_name, square_size_m)

    plt.figure(figsize=(7, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(
        "Click board corners in order:\n"
        "top-left, top-right, bottom-right, bottom-left, then press Enter"
    )
    image_points = plt.ginput(4, timeout=0)
    plt.close()

    if len(image_points) != 4:
        sys.exit("Need exactly four board-corner clicks.")

    return np.asarray(image_points, dtype=np.float32), world_points


def fit_image_to_world_homography(k, d, image_points, world_points):
    image_points_ud = undistort_pixel_points(image_points, k, d)
    h, _ = cv2.findHomography(image_points_ud, np.asarray(world_points, np.float32), 0)
    if h is None:
        raise DetectionError("Could not compute a plane homography from the board.")
    return h


def pixel_to_world(h_img_to_world, k, d, point):
    point_ud = undistort_pixel_points([point], k, d)[0]
    uv1 = np.array([point_ud[0], point_ud[1], 1.0], dtype=np.float64)
    world = h_img_to_world @ uv1
    world /= world[2]
    return world[:2]


def prompt_measurement_points(img):
    plt.figure(figsize=(7, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Click TWO points on the object, then press Enter")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        sys.exit("Need exactly two measurement clicks.")
    return np.asarray(pts, dtype=np.float32)


def auto_board_correspondences(gray, board_name, cols, rows, square_size_m, debug=False):
    if board_name == "checker":
        return detect_checkerboard_auto(gray, cols, rows, square_size_m, debug=debug)
    return detect_charuco_auto(gray, debug=debug)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--img", required=True, help="Path to photograph")
    parser.add_argument(
        "--yaml",
        default=None,
        help="YAML file containing K and D. If omitted, the path is derived from --board.",
    )
    parser.add_argument(
        "--board",
        choices=["checker", "charuco"],
        default="checker",
        help="Pattern used during calibration",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=CHECKERBOARD_INNER_CORNERS[0],
        help="Inner corners along X for checkerboard",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=CHECKERBOARD_INNER_CORNERS[1],
        help="Inner corners along Y for checkerboard",
    )
    parser.add_argument(
        "--sq",
        type=float,
        default=SQUARE_SIZE_M,
        help="Square size in metres",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Do not fall back to manual board-corner clicks if auto detection fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test board detection and homography setup without opening click prompts",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detector attempts",
    )
    args = parser.parse_args()

    yaml_path = args.yaml or str(board_intrinsics_path(args.board))
    k, d = load_intrinsics(yaml_path)

    img = cv2.imread(args.img)
    if img is None:
        sys.exit(f"Cannot read {args.img}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    used_manual_fallback = False
    try:
        image_points, world_points, method = auto_board_correspondences(
            gray,
            args.board,
            args.cols,
            args.rows,
            args.sq,
            debug=args.debug,
        )
        print(f"Automatic board detection succeeded with {method}.")
    except DetectionError as exc:
        print(str(exc))
        if args.auto_only:
            sys.exit("Automatic board detection failed and --auto-only was set.")
        if args.dry_run:
            print("Automatic detection failed. Manual board-corner fallback would be used.")
            return
        used_manual_fallback = True
        image_points, world_points = prompt_manual_board_corners(
            img, args.board, args.sq
        )
        method = "manual_outer_corners"
        print("Using manual board-corner fallback.")

    h_img_to_world = fit_image_to_world_homography(k, d, image_points, world_points)

    if args.dry_run:
        print(f"Homography setup succeeded with {method}.")
        return

    measurement_points = prompt_measurement_points(img)
    p1_world = pixel_to_world(h_img_to_world, k, d, measurement_points[0])
    p2_world = pixel_to_world(h_img_to_world, k, d, measurement_points[1])
    distance_m = np.linalg.norm(p1_world - p2_world)

    print(f"Distance = {distance_m * 1000:.1f} mm   (Board: {args.board})")
    if used_manual_fallback:
        print("Board pose was estimated from four manual board-corner clicks.")
    else:
        print("Board pose was estimated from automatic board feature detection.")


if __name__ == "__main__":
    main()
