#!/usr/bin/env python3
"""
libcamera-compatible MJPEG video streaming server for Raspberry Pi
Works with latest Raspberry Pi OS that uses libcamera
"""

import cv2
from flask import Flask, Response
import subprocess
import threading
import time
import os
import signal
import sys

app = Flask(__name__)

class LibCameraVideoStream:
    def __init__(self):
        self.frame = None
        self.running = False
        self.process = None
        self.thread = None
        
    def start_libcamera_stream(self):
        """Start libcamera-vid and capture frames"""
        try:
            # Use libcamera-vid to stream to stdout
            cmd = [
                'libcamera-vid',
                '--timeout', '0',           # Run indefinitely  
                '--width', '320',
                '--height', '240',
                '--framerate', '15',
                '--output', '-',            # Output to stdout
                '--codec', 'mjpeg',         # MJPEG format
                '--inline'                  # Include headers in stream
            ]
            
            print("Starting libcamera stream...")
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._read_frames)
            self.thread.daemon = True
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start libcamera: {e}")
            return False
    
    def _read_frames(self):
        """Read frames from libcamera process"""
        buffer = b''
        
        while self.running and self.process.poll() is None:
            try:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Look for JPEG frame boundaries
                start = buffer.find(b'\xff\xd8')  # JPEG start
                end = buffer.find(b'\xff\xd9')    # JPEG end
                
                if start != -1 and end != -1 and end > start:
                    # Extract complete JPEG frame
                    jpeg_frame = buffer[start:end+2]
                    self.frame = jpeg_frame
                    
                    # Remove processed data from buffer
                    buffer = buffer[end+2:]
                    
            except Exception as e:
                print(f"Error reading frame: {e}")
                break
    
    def get_frame(self):
        return self.frame
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

# Try libcamera first, fallback to OpenCV
camera_stream = None

def initialize_camera():
    global camera_stream
    
    # Method 1: Try libcamera (modern RPi OS)
    try:
        # Check if libcamera is available
        result = subprocess.run(['which', 'libcamera-vid'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("libcamera detected, using libcamera stream...")
            camera_stream = LibCameraVideoStream()
            if camera_stream.start_libcamera_stream():
                return True
        
        print("libcamera not available or failed")
        
    except Exception as e:
        print(f"libcamera initialization failed: {e}")
    
    # Method 2: Fallback to OpenCV (older systems)
    try:
        print("Falling back to OpenCV...")
        import cv2
        
        class OpenCVCamera:
            def __init__(self):
                self.video = cv2.VideoCapture(0, cv2.CAP_V4L2)
                self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                
            def get_frame(self):
                success, image = self.video.read()
                if success:
                    ret, jpeg = cv2.imencode('.jpg', image, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 30])
                    return jpeg.tobytes()
                return None
            
            def stop(self):
                self.video.release()
        
        camera_stream = OpenCVCamera()
        return True
        
    except Exception as e:
        print(f"OpenCV fallback failed: {e}")
        return False

def generate_frames():
    while True:
        if camera_stream:
            frame = camera_stream.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.067)  # ~15 FPS

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    if camera_stream:
        return {'status': 'Camera server running', 'method': 'libcamera' if isinstance(camera_stream, LibCameraVideoStream) else 'opencv'}
    else:
        return {'status': 'Camera error'}, 500

@app.route('/')
def index():
    return '''
    <h1>Raspberry Pi Camera Server (libcamera compatible)</h1>
    <p>Camera feed: <a href="/video">/video</a></p>
    <p>Status: <a href="/status">/status</a></p>
    <img src="/video" width="320" height="240">
    '''

def signal_handler(sig, frame):
    print("\nShutting down...")
    if camera_stream:
        camera_stream.stop()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    
    if not initialize_camera():
        print("ERROR: No camera method worked!")
        print("\nTroubleshooting steps:")
        print("1. Check camera connection: ls /dev/video*")
        print("2. Test libcamera: libcamera-hello --help")
        print("3. Check permissions: groups $USER")
        exit(1)
    
    print("Camera initialized successfully!")
    print("Students can connect using: http://cvpi33.local:5000/video")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)