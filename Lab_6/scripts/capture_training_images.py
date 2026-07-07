#!/usr/bin/env python3
"""Capture one image dataset with the Raspberry Pi camera CLI."""

import argparse
import shutil
import subprocess
import time
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = LAB_DIR / "captured_images"


def available_camera_command():
    """Return the still-camera command provided by the installed Pi OS."""
    for command in ("rpicam-still", "libcamera-still"):
        if shutil.which(command):
            return command
    return None


def next_image_number(output_dir):
    numbers = []
    for image_path in output_dir.glob("img_*.jpg"):
        try:
            numbers.append(int(image_path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=-1) + 1


def capture_images(num_images, delay, width, height):
    camera_command = available_camera_command()
    if camera_command is None:
        raise RuntimeError(
            "Neither rpicam-still nor libcamera-still is installed. "
            "Check the Raspberry Pi camera software."
        )

    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    start_number = next_image_number(output_dir)

    print(f"Camera command: {camera_command}")
    print(f"Saving to: {output_dir}")
    print(f"Capturing {num_images} images at {width}x{height}")
    print("Starting in 3 seconds...")
    for remaining in range(3, 0, -1):
        print(remaining)
        time.sleep(1)

    successful = 0
    for offset in range(num_images):
        image_number = start_number + offset
        output_path = output_dir / f"img_{image_number:03d}.jpg"
        command = [
            camera_command,
            "--nopreview",
            "--width",
            str(width),
            "--height",
            str(height),
            "--timeout",
            "100",
            "--output",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"[failed] {output_path.name}: camera command timed out")
        else:
            if result.returncode == 0 and output_path.exists():
                successful += 1
                print(f"[ok] {successful}/{num_images}: {output_path.name}")
            else:
                message = result.stderr.strip().splitlines()
                detail = message[-1] if message else "camera command failed"
                print(f"[failed] {output_path.name}: {detail}")

        if offset < num_images - 1:
            time.sleep(delay)

    print(f"Capture complete: {successful}/{num_images} images saved")
    return successful == num_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one image dataset for CVAT annotation."
    )
    parser.add_argument(
        "-n", "--num-images", type=int, default=80, help="Number of images"
    )
    parser.add_argument(
        "-d", "--delay", type=float, default=0.5, help="Seconds between images"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error("--num-images must be greater than zero")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.width <= 0 or args.height <= 0:
        parser.error("image dimensions must be greater than zero")
    return args


def main():
    args = parse_args()
    complete = capture_images(
        num_images=args.num_images,
        delay=args.delay,
        width=args.width,
        height=args.height,
    )
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
