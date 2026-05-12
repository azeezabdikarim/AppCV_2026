from pathlib import Path

import cv2
import numpy as np
import yaml

from board_config import (
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CHECKERBOARD_INNER_CORNERS,
    CHECKERBOARD_SQUARES_X,
    CHECKERBOARD_SQUARES_Y,
    SQUARE_SIZE_M,
    charuco_object_points_for_ids,
    checkerboard_object_points,
    create_charuco_board,
    detect_charuco_board,
)


def load_intrinsics(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    k = np.asarray(payload["K"]["data"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(payload["D"]["data"], dtype=np.float64).reshape(-1, 1)
    return k, d


def _count_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = 0
    first_frame_shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if first_frame_shape is None:
            first_frame_shape = frame.shape

    cap.release()
    return frame_count, first_frame_shape


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_count = reported_frame_count
    frame_count_source = "metadata"

    if frame_count <= 0:
        frame_count, first_frame_shape = _count_video_frames(video_path)
        frame_count_source = "decoded"
        if first_frame_shape is not None and (width <= 0 or height <= 0):
            height, width = first_frame_shape[:2]

    if frame_count <= 0:
        raise ValueError(
            f"Video appears to contain zero readable frames: {video_path}"
        )

    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "frame_count": frame_count,
        "reported_frame_count": reported_frame_count,
        "frame_count_source": frame_count_source,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def sample_frame_indices(frame_count, num_samples=9):
    if frame_count <= 0:
        raise ValueError(
            "Cannot sample frames because the video frame count is zero."
        )
    num_samples = min(num_samples, frame_count)
    return np.linspace(0, frame_count - 1, num_samples, dtype=int).tolist()


def read_frame(video_path, frame_index):
    if frame_index < 0:
        raise ValueError(f"Frame index must be non-negative, not {frame_index}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Some OpenCV/video backend combinations ignore random frame seeks for mp4.
    # These lab clips are short, so decode sequentially for predictable behavior.
    current_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"Could not read frame {frame_index} from {video_path}. "
                f"The clip ended at frame {current_index}."
            )
        if current_index == frame_index:
            cap.release()
            return frame
        current_index += 1


def read_frames(video_path, frame_indices):
    if not frame_indices:
        raise ValueError("No frame indices were provided for reading.")
    requested = [int(idx) for idx in frame_indices]
    if min(requested) < 0:
        raise ValueError("Frame indices must all be non-negative.")

    requested_set = set(requested)
    max_index = max(requested)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    found = {}
    current_index = 0
    while current_index <= max_index:
        ok, frame = cap.read()
        if not ok:
            break
        if current_index in requested_set:
            found[current_index] = frame.copy()
            if len(found) == len(requested_set):
                break
        current_index += 1

    cap.release()

    missing = [idx for idx in requested if idx not in found]
    if missing:
        raise RuntimeError(
            f"Could not read requested frames {missing} from {video_path}. "
            f"The clip ended after frame {current_index - 1}."
        )

    return [(idx, found[idx]) for idx in requested]


def detect_charuco(gray):
    board = create_charuco_board()
    marker_corners, marker_ids, charuco_corners, charuco_ids = detect_charuco_board(
        gray, board=board
    )
    if marker_ids is None or len(marker_ids) < 4:
        return None

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None

    object_points = charuco_object_points_for_ids(board, charuco_ids).astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    return {
        "board_type": "charuco",
        "object_points": object_points,
        "image_points": image_points,
        "charuco_ids": charuco_ids.astype(np.int32),
    }


def detect_checkerboard(gray):
    pattern = CHECKERBOARD_INNER_CORNERS

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=0)
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not found:
        return None

    return {
        "board_type": "checker",
        "object_points": checkerboard_object_points().astype(np.float32),
        "image_points": corners.reshape(-1, 2).astype(np.float32),
    }


def detect_board(frame_bgr, board_type="charuco"):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if board_type == "charuco":
        return detect_charuco(gray)
    if board_type == "checker":
        return detect_checkerboard(gray)
    raise ValueError(f"Unsupported board_type: {board_type}")


def estimate_pose(detection, k, d):
    ok, rvec, tvec = cv2.solvePnP(
        detection["object_points"],
        detection["image_points"].reshape(-1, 1, 2),
        k,
        d,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return {
        "rvec": rvec.reshape(3, 1).astype(np.float64),
        "tvec": tvec.reshape(3, 1).astype(np.float64),
    }


def track_poses(video_path, board_type, k, d):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    poses = []
    frame_index = 0
    valid_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection = detect_board(frame, board_type=board_type)
        if detection is None:
            poses.append(None)
        else:
            pose = estimate_pose(detection, k, d)
            poses.append(pose)
            if pose is not None:
                valid_count += 1

        frame_index += 1

    cap.release()
    return poses, valid_count


def draw_detected_points(frame_bgr, detection, color=(0, 255, 0)):
    output = frame_bgr.copy()
    for x, y in detection["image_points"]:
        cv2.circle(output, (int(round(x)), int(round(y))), 4, color, 1)
    return output


def draw_axes(frame_bgr, pose, k, d, axis_length_m=0.06):
    output = frame_bgr.copy()
    cv2.drawFrameAxes(output, k, d, pose["rvec"], pose["tvec"], axis_length_m, 2)
    return output


def board_extent_m(board_type):
    if board_type == "charuco":
        return CHARUCO_SQUARES_X * SQUARE_SIZE_M, CHARUCO_SQUARES_Y * SQUARE_SIZE_M
    if board_type == "checker":
        return CHECKERBOARD_SQUARES_X * SQUARE_SIZE_M, CHECKERBOARD_SQUARES_Y * SQUARE_SIZE_M
    raise ValueError(f"Unsupported board_type: {board_type}")


def make_centered_cuboid(board_type="charuco", size_x=None, size_y=None, height=None):
    board_width, board_height = board_extent_m(board_type)
    size_x = 0.4 * board_width if size_x is None else float(size_x)
    size_y = 0.3 * board_height if size_y is None else float(size_y)
    height = 0.6 * min(size_x, size_y) if height is None else float(height)

    x0 = 0.5 * (board_width - size_x)
    y0 = 0.5 * (board_height - size_y)
    x1 = x0 + size_x
    y1 = y0 + size_y
    z0 = 0.0
    z1 = -height

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return vertices, edges


def draw_cuboid(frame_bgr, pose, k, d, vertices, edges):
    projected, _ = cv2.projectPoints(vertices, pose["rvec"], pose["tvec"], k, d)
    projected = projected.reshape(-1, 2)
    output = frame_bgr.copy()

    for start, end in edges:
        p0 = projected[start]
        p1 = projected[end]
        cv2.line(
            output,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (0, 128, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return output


def annotate_frame(frame_bgr, text):
    output = frame_bgr.copy()
    cv2.putText(
        output,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        1,
        lineType=cv2.LINE_AA,
    )
    return output


def render_overlay_video(
    video_path,
    output_path,
    poses,
    k,
    d,
    vertices=None,
    edges=None,
    axis_length_m=0.06,
    draw_mode="cuboid",
):
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        info["fps"] if info["fps"] > 0 else 15.0,
        (info["width"], info["height"]),
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pose = poses[frame_index] if frame_index < len(poses) else None
        if pose is not None:
            if draw_mode in {"axes", "axes+cuboid"}:
                frame = draw_axes(frame, pose, k, d, axis_length_m=axis_length_m)
            if draw_mode in {"cuboid", "axes+cuboid"} and vertices is not None and edges is not None:
                frame = draw_cuboid(frame, pose, k, d, vertices, edges)
        else:
            frame = annotate_frame(frame, "Board pose unavailable in this frame")

        writer.write(frame)
        frame_index += 1

    writer.release()
    cap.release()
    return output_path
