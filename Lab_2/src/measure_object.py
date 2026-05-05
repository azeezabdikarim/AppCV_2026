#!/usr/bin/env python3
"""
Estimate the real-world distance between two clicked points that lie on the
calibration-board plane.
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
    SQUARE_SIZE_M,
    charuco_object_points_for_ids,
    create_charuco_board,
    get_aruco_dictionary,
)


def load_intrinsics(yaml_path):
    with open(yaml_path, "r") as f:
        intr = yaml.safe_load(f)
    k = np.asarray(intr["K"]["data"]).reshape(3, 3)
    d = np.asarray(intr["D"]["data"]).reshape(-1, 1)
    return k, d


def detect_checkerboard(gray, cols, rows, square_size_m):
    pattern = (cols, rows)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        raise SystemExit("Checkerboard not found.")

    cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    objp *= square_size_m
    return corners, objp


def detect_charuco(gray):
    dictionary = get_aruco_dictionary()
    board = create_charuco_board()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) < 4:
        raise SystemExit("ChArUco markers not detected.")

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        raise SystemExit("Not enough ChArUco corners.")

    objp = charuco_object_points_for_ids(board, charuco_ids)
    return charuco_corners, objp


def pix_to_world(h_inv, undistorted_point):
    uv1 = np.array([undistorted_point[0], undistorted_point[1], 1.0])
    world = h_inv @ uv1
    world /= world[2]
    return world[:2]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--img", required=True, help="Path to photograph")
    parser.add_argument(
        "--yaml",
        default="captured_points/intrinsics.yml",
        help="YAML file containing K and D",
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
    args = parser.parse_args()

    k, d = load_intrinsics(args.yaml)

    img = cv2.imread(args.img)
    if img is None:
        sys.exit(f"Cannot read {args.img}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.board == "checker":
        corners, objp = detect_checkerboard(gray, args.cols, args.rows, args.sq)
    else:
        corners, objp = detect_charuco(gray)

    _, rvec, tvec = cv2.solvePnP(objp, corners, k, d, flags=cv2.SOLVEPNP_ITERATIVE)

    r, _ = cv2.Rodrigues(rvec)
    rt = np.hstack((r[:, :2], tvec))
    h = k @ rt
    h_inv = np.linalg.inv(h)

    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Click TWO points on the object, then press Enter")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        sys.exit("Need exactly two clicks.")

    pts_ud = cv2.undistortPoints(
        np.array(pts, dtype=np.float32).reshape(-1, 1, 2),
        k,
        d,
        P=k,
    ).reshape(-1, 2)

    p1_world = pix_to_world(h_inv, pts_ud[0])
    p2_world = pix_to_world(h_inv, pts_ud[1])
    distance_m = np.linalg.norm(p1_world - p2_world)

    print(f"Distance = {distance_m * 1000:.1f} mm   (Board: {args.board})")


if __name__ == "__main__":
    main()
