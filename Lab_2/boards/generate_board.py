#!/usr/bin/env python3
"""
Generate printable checkerboard and ChArUco boards on A4 paper.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board_config import (  # noqa: E402
    CHARUCO_MARKER_SIZE_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CHECKERBOARD_SQUARES_X,
    CHECKERBOARD_SQUARES_Y,
    SQUARE_SIZE_M,
    create_charuco_board,
)


DPI = 300
MARGIN_PX = 50
OUT_DIR = Path(__file__).resolve().parent


def mm_to_px(mm):
    return int(round(mm * DPI / 25.4))


def make_checkerboard(nx, ny, square_mm):
    square_px = mm_to_px(square_mm)
    width_px, height_px = nx * square_px, ny * square_px
    img = np.zeros((height_px, width_px), np.uint8)
    for y in range(ny):
        for x in range(nx):
            if (x + y) % 2 == 0:
                img[y * square_px : (y + 1) * square_px, x * square_px : (x + 1) * square_px] = 255
    return img


def make_charuco(nx, ny, square_mm):
    board = create_charuco_board()
    width_px, height_px = mm_to_px(nx * square_mm), mm_to_px(ny * square_mm)
    if hasattr(board, "generateImage"):
        return board.generateImage((width_px, height_px))
    return board.draw((width_px, height_px))


def assemble_pdf(images, out_path):
    a4_width_px, a4_height_px = mm_to_px(210), mm_to_px(297)
    canvas = Image.new("RGB", (a4_width_px, a4_height_px), "white")
    y = MARGIN_PX
    for img in images:
        pil = Image.fromarray(
            img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        )
        x = (a4_width_px - pil.width) // 2
        canvas.paste(pil, (x, y))
        y += pil.height + MARGIN_PX
    canvas.save(out_path, "PDF", resolution=DPI)


def main():
    square_mm = SQUARE_SIZE_M * 1000.0
    marker_mm = CHARUCO_MARKER_SIZE_M * 1000.0

    checker = make_checkerboard(
        CHECKERBOARD_SQUARES_X,
        CHECKERBOARD_SQUARES_Y,
        square_mm,
    )
    charuco = make_charuco(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        square_mm,
    )

    cv2.imwrite(str(OUT_DIR / "checkerboard.png"), checker)
    cv2.imwrite(str(OUT_DIR / "charuco.png"), charuco)
    assemble_pdf([checker, charuco], OUT_DIR / "calibration_boards.pdf")

    print("Saved checkerboard.png, charuco.png, and calibration_boards.pdf")
    print(f"Checkerboard square size: {square_mm:.1f} mm")
    print(f"ChArUco marker size: {marker_mm:.1f} mm")


if __name__ == "__main__":
    main()
