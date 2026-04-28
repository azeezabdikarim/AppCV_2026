import argparse
import threading
import time
import logging
from flask import Flask, render_template, Response

from picamera2 import Picamera2
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

picam2 = None
camera_available = False
stream_width = 640
stream_height = 480
stream_fps = 15

frame_count = 0
byte_count = 0
counter_lock = threading.Lock()


def init_camera(width, height):
    global picam2, camera_available
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (width, height)})
        picam2.configure(config)
        picam2.start()
        camera_available = True
        logger.info(f"Picamera2 initialized at {width}x{height}.")
    except Exception as exc:
        picam2 = None
        camera_available = False
        logger.error(f"Failed to initialize Picamera2: {exc}")


def throughput_logger():
    last_frames = 0
    last_bytes = 0
    while True:
        time.sleep(1.0)
        with counter_lock:
            frames = frame_count
            total_bytes = byte_count
        df = frames - last_frames
        db = total_bytes - last_bytes
        last_frames, last_bytes = frames, total_bytes
        if df > 0:
            logger.info(
                f"served {df:.1f} fps, {db/1024:.0f} KB/s, {db*8/1e6:.2f} Mbit/s"
            )


def generate_frames():
    global frame_count, byte_count
    frame_period = 1.0 / stream_fps
    next_deadline = time.perf_counter()

    while True:
        if camera_available:
            frame = picam2.capture_array()
        else:
            frame = np.zeros((stream_height, stream_width, 3), dtype=np.uint8)
            text = "No camera is detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            size, _ = cv2.getTextSize(text, font, 1, 2)
            x = (stream_width - size[0]) // 2
            y = (stream_height + size[1]) // 2
            cv2.putText(frame, text, (x, y), font, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.warning("Frame encoding failed, skipping frame.")
            continue
        frame_bytes = buffer.tobytes()

        with counter_lock:
            frame_count += 1
            byte_count += len(frame_bytes)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        next_deadline += frame_period
        sleep_for = next_deadline - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_deadline = time.perf_counter()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Raspberry Pi Camera Stream Server using Picamera2'
    )
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    stream_width = args.width
    stream_height = args.height
    stream_fps = args.fps

    init_camera(stream_width, stream_height)

    threading.Thread(target=throughput_logger, daemon=True).start()

    logger.info(f"Starting stream server on port {args.port} ({stream_width}x{stream_height} @ {stream_fps} fps)")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}/video_feed")
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
