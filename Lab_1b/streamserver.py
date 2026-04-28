import argparse
import time
import logging
from flask import Flask, render_template, Response

# picamera2 for Raspberry Pi OS Bookworm camera support
from picamera2 import Picamera2
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Attempt to initialize the Pi camera
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (854, 480)})
    picam2.configure(config)
    picam2.start()
    camera_available = True
    logger.info("Picamera2 initialized successfully.")
except Exception as exc:
    picam2 = None
    camera_available = False
    logger.error(f"Failed to initialize Picamera2: {exc}")


def generate_frames():
    """
    Generator yielding JPEG-encoded frames from the camera, or a placeholder if unavailable.
    """
    while True:
        if camera_available:
            # Capture a frame from the Pi camera
            frame = picam2.capture_array()
        else:
            # Create a blank frame and overlay text
            height, width = 480, 854
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            text = "No camera is detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2
            size, _ = cv2.getTextSize(text, font, scale, thickness)
            x = (width - size[0]) // 2
            y = (height + size[1]) // 2
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.warning("Frame encoding failed, skipping frame.")
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Control frame rate
        time.sleep(0.05)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Raspberry Pi Camera Stream Server using Picamera2'
    )
    parser.add_argument(
        '--port', type=int, default=8000,
        help='Port to run the server on (default: 8000)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting camera stream server on port {args.port}")
    logger.info(f"Access the stream at http://<YOUR_PI_IP>:{args.port}")
    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server error: {e}")
