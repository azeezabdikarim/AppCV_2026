#!/usr/bin/env python3

import atexit
import signal
import socket
import sys
import time

from flask import Flask, Response, jsonify, render_template

from camera_manager import CameraManager
from movement_controller import MovementController
from vision_worker import VisionWorker


app = Flask(__name__)

camera = CameraManager(width=320, height=240, fps=10)
movement = MovementController(forward_speed=8, pulse_speed=10, pulse_duration=0.45)
vision = VisionWorker(camera, movement, target_fps=8, jpeg_quality=65)


def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


def cleanup():
    vision.stop()
    camera.stop()
    movement.cleanup()


atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda _sig, _frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(0))


@app.route("/")
def index():
    return render_template("control.html")


def generate_frames():
    while True:
        frame = vision.get_jpeg()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(1.0 / 8.0)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start_forward", methods=["GET", "POST"])
def start_forward():
    success, message = movement.start_forward()
    return jsonify({"success": success, "message": message, "status": vision.get_status()})


@app.route("/api/stop", methods=["GET", "POST"])
def stop():
    success, message = movement.stop()
    return jsonify({"success": success, "message": message, "status": vision.get_status()})


@app.route("/api/move/<direction>", methods=["GET", "POST"])
def move(direction):
    success, message = movement.manual_pulse(direction)
    return jsonify({"success": success, "message": message, "status": vision.get_status()})


@app.route("/api/steer/<direction>", methods=["GET", "POST"])
def steer(direction):
    success, message = movement.apply_gesture_direction(direction)
    return jsonify({"success": success, "message": message, "status": vision.get_status()})


@app.route("/api/status")
def status():
    return jsonify(vision.get_status())


if __name__ == "__main__":
    camera.start()
    vision.start()

    # Give the background threads a moment to fill the first cached frame.
    time.sleep(1.0)
    local_ip = get_local_ip()

    print("Week 1 PiCar-X gesture steering app")
    print("Cars should be on: CV-PI-NET")
    print("Laptop viewers should be on: CV-CAR-VIEW-5G")
    print("Use one live video UI per car.")
    print("")
    print("Open one of these URLs from the controlling laptop:")
    print("  http://localhost:5000")
    print(f"  http://{local_ip}:5000")
    print("")
    print("Press Ctrl+C to stop the server.")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
