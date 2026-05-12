from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CAPTURED_POINTS_DIR = BASE_DIR / "captured_points"


CHECKERBOARD_SQUARES_X = 9
CHECKERBOARD_SQUARES_Y = 6
CHECKERBOARD_INNER_CORNERS = (
    CHECKERBOARD_SQUARES_X - 1,
    CHECKERBOARD_SQUARES_Y - 1,
)

CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7

SQUARE_SIZE_M = 0.03
CHARUCO_MARKER_SIZE_M = 0.022

ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50

VALID_BOARDS = ("checker", "charuco")


def normalize_board_name(board: str) -> str:
    board_name = board.strip().lower()
    if board_name not in VALID_BOARDS:
        raise ValueError(
            f"Unsupported board '{board}'. Expected one of: {', '.join(VALID_BOARDS)}."
        )
    return board_name


def board_data_dir(board: str) -> Path:
    return DATA_DIR / normalize_board_name(board)


def board_captured_points_dir(board: str) -> Path:
    return CAPTURED_POINTS_DIR / normalize_board_name(board)


def board_corners_path(board: str) -> Path:
    return board_captured_points_dir(board) / "corners.npz"


def board_intrinsics_path(board: str) -> Path:
    return board_captured_points_dir(board) / "intrinsics.yml"


def board_measurement_image_path(board: str) -> Path:
    return board_data_dir(board) / "object_on_board.jpg"


def board_clip_path(board: str) -> Path:
    return board_data_dir(board) / "board_clip.mp4"


def board_calibration_images(board: str):
    return board_data_dir(board).glob("img_*.jpg")


def get_aruco_dictionary():
    aruco = cv2.aruco
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    return aruco.Dictionary_get(ARUCO_DICTIONARY_ID)


def create_charuco_board():
    aruco = cv2.aruco
    dictionary = get_aruco_dictionary()
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            SQUARE_SIZE_M,
            CHARUCO_MARKER_SIZE_M,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        SQUARE_SIZE_M,
        CHARUCO_MARKER_SIZE_M,
        dictionary,
    )


def checkerboard_object_points():
    pattern = CHECKERBOARD_INNER_CORNERS
    num_corners = pattern[0] * pattern[1]
    objp = np.zeros((num_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def charuco_object_points_for_ids(board, charuco_ids):
    ids = charuco_ids.flatten()
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    return corners[ids].astype(np.float32)
