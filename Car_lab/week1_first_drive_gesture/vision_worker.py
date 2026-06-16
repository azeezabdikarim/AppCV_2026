#!/usr/bin/env python3

import importlib
import os
import threading
import time

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


class VisionWorker:
    """Runs MediaPipe once per frame and caches debug JPEGs for Flask clients."""

    VALID_DIRECTIONS = {"left", "right", "unknown"}

    def __init__(self, camera, movement_controller, target_fps=8, jpeg_quality=65):
        self.camera = camera
        self.movement_controller = movement_controller
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.running = False
        self.thread = None
        self.latest_jpeg = None
        self.gesture_direction = "unknown"
        self.gesture_error = ""
        self.keypoints_used = []
        self.hand_detected = False
        self.fps = 0.0
        self.frame_count = 0
        self._last_fps_time = time.time()
        self._lock = threading.Lock()
        self._gesture_module = None
        self._gesture_mtime = None

        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        self.mediapipe_error = ""
        if mp is not None:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as exc:
                self.mediapipe_error = f"mediapipe init failed: {exc}"
                print(self.mediapipe_error)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            start = time.time()
            frame = self.camera.get_frame()
            debug_frame = self._process_frame(frame)
            ok, buffer = cv2.imencode(".jpg", debug_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self._lock:
                    self.latest_jpeg = buffer.tobytes()

            elapsed = time.time() - start
            time.sleep(max(0.0, (1.0 / self.target_fps) - elapsed))

    def _process_frame(self, frame):
        debug_frame = frame.copy()
        direction = "unknown"
        keypoints_used = []
        hand_detected = False
        error = ""

        if self.hands is None:
            error = self.mediapipe_error or "mediapipe is not installed"
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                direction, keypoints_used, error = self._call_student_logic(hand_landmarks.landmark)
                self._draw_landmarks(debug_frame, hand_landmarks, keypoints_used)

        if direction in {"left", "right"}:
            self.movement_controller.apply_gesture_direction(direction)
        else:
            self.movement_controller.center_steering(source="gesture_unknown")

        self._update_fps()
        self._draw_overlay(debug_frame, direction, hand_detected, error)

        with self._lock:
            self.gesture_direction = direction
            self.keypoints_used = keypoints_used
            self.hand_detected = hand_detected
            self.gesture_error = error

        return debug_frame

    def _call_student_logic(self, hand_landmarks):
        try:
            module = self._load_gesture_module()
            result = module.detect_steering_gesture(hand_landmarks)
        except Exception as exc:
            return "unknown", [], f"gesture_logic.py error: {exc}"

        if not isinstance(result, tuple) or len(result) != 2:
            return "unknown", [], "return (direction, keypoints_used)"

        direction, keypoints_used = result
        direction = str(direction).lower()
        if direction not in self.VALID_DIRECTIONS:
            return "unknown", [], f"invalid direction: {direction}"

        if keypoints_used is None:
            keypoints_used = []
        try:
            keypoints_used = [int(idx) for idx in keypoints_used]
        except Exception:
            keypoints_used = []

        return direction, keypoints_used, ""

    def _load_gesture_module(self):
        module_path = os.path.join(os.path.dirname(__file__), "gesture_logic.py")
        mtime = os.path.getmtime(module_path)
        if self._gesture_module is None:
            import gesture_logic
            self._gesture_module = gesture_logic
            self._gesture_mtime = mtime
        elif mtime != self._gesture_mtime:
            self._gesture_module = importlib.reload(self._gesture_module)
            self._gesture_mtime = mtime
        return self._gesture_module

    def _draw_landmarks(self, frame, hand_landmarks, keypoints_used):
        h, w = frame.shape[:2]
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(230, 230, 230), thickness=1),
        )

        keypoint_set = set(keypoints_used)
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if idx in keypoint_set:
                cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 9, (255, 255, 255), 2)
                cv2.putText(frame, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 4, (140, 140, 140), -1)

    def _draw_overlay(self, frame, direction, hand_detected, error):
        status = self.movement_controller.get_status()
        moving_text = "MOVING" if status["forward_active"] else "STOPPED"
        line_1 = f"Gesture: {direction.upper()}  Hand: {'YES' if hand_detected else 'NO'}"
        line_2 = f"Car: {moving_text}  Steering: {status['current_direction']} ({status['current_angle']} deg)"
        line_3 = f"FPS: {self.fps:.1f}  Camera: {self.camera.method}"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (20, 24, 30), -1)
        cv2.putText(frame, line_1, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line_2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 240, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, line_3, (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 210, 255), 1, cv2.LINE_AA)

        if error:
            cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (35, 25, 25), -1)
            cv2.putText(frame, error[:58], (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (120, 190, 255), 1, cv2.LINE_AA)

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self._last_fps_time = now

    def get_jpeg(self):
        with self._lock:
            return self.latest_jpeg

    def get_status(self):
        with self._lock:
            status = {
                "mediapipe_available": self.hands is not None,
                "hand_detected": self.hand_detected,
                "gesture_direction": self.gesture_direction,
                "keypoints_used": self.keypoints_used,
                "gesture_error": self.gesture_error,
                "fps": round(self.fps, 1),
                "target_fps": self.target_fps,
                "camera_method": self.camera.method,
                "camera_error": self.camera.error,
                "car_network": "CV-PI-NET",
                "viewer_network": "CV-CAR-VIEW-5G",
            }
        status.update(self.movement_controller.get_status())
        return status

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.hands:
            self.hands.close()
