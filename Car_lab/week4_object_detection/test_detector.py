#!/usr/bin/env python3
"""Run the student detector on one saved image without starting the motors."""

import argparse
from pathlib import Path

import cv2

from week4_object_detection.sign_detector import SignDetector


def draw_detections(image, detections):
    output = image.copy()
    for detection in detections:
        x, y, width, height = detection["bbox"]
        color = (0, 0, 255) if detection["class_name"] in {"Stop_Sign", "Stahp"} else (0, 200, 0)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        label = (
            f'{detection["class_name"]} {detection["confidence"]:.2f} '
            f'{100 * detection["area_ratio"]:.1f}%'
        )
        cv2.putText(
            output,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="detection_test.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = SignDetector(model_path=args.model)
    detections = detector.detect_signs(image)
    output = draw_detections(image, detections)
    output_path = Path(args.output).resolve()
    if not cv2.imwrite(str(output_path), output):
        raise SystemExit(f"Could not write output: {output_path}")

    print(f"Detections: {len(detections)}")
    for detection in detections:
        print(
            f'- {detection["class_name"]}: '
            f'confidence={detection["confidence"]:.3f}, '
            f'area={100 * detection["area_ratio"]:.2f}%'
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

