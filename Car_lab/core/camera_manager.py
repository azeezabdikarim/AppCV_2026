#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import time
import subprocess
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, width=320, height=240, fps=15):
        """Initialize camera using rpicam/libcamera streaming with OpenCV fallback."""
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        self.lock = threading.Lock()
        self.process = None
        self.current_frame = None
        self.capture_thread = None
        self.method = "not_started"
        self.error = ""
        
        logger.info("Camera manager initialized")
    
    def _create_placeholder_frame(self, message="Waiting for camera..."):
        """Create a frame with a message"""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # Calculate text size and position for centering
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (5, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def start_streaming(self):
        """Start the camera streaming in a background thread."""
        if not self.is_running:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info("Camera streaming started")

    def _camera_command(self):
        """Return the camera CLI available on this Pi OS image."""
        for command in ("rpicam-vid", "libcamera-vid"):
            if shutil.which(command):
                return command
        return None
    
    def _capture_frames(self):
        """Continuously capture frames using Pi camera CLI, then OpenCV fallback."""
        camera_command = self._camera_command()
        if camera_command and self._run_camera_command_loop(camera_command):
            return

        if not camera_command:
            self.error = "rpicam-vid/libcamera-vid not found"
            logger.error(self.error)

        if self._run_opencv_loop():
            return

        self._fallback_placeholder_loop("Camera unavailable")

    def _run_camera_command_loop(self, camera_command):
        logger.info(f"Starting {camera_command} streaming...")

        try:
            cmd = [
                camera_command,
                "--timeout", "0",
                "--width", str(self.width),
                "--height", str(self.height),
                "--framerate", str(self.fps),
                "--output", "-",
                "--codec", "mjpeg",
                "--inline",
                "-n",
            ]
            
            self.method = camera_command
            self.error = ""
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            
            time.sleep(1)
            if self.process.poll() is not None:
                self.error = f"{camera_command} failed to start"
                logger.error(self.error)
                return False
            
            logger.info(f"{camera_command} streaming started successfully")
            
            buffer = b""
            while self.is_running and self.process.poll() is None:
                try:
                    chunk = self.process.stdout.read(4096)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # Look for JPEG boundaries
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9')
                    
                    if start != -1 and end != -1 and end > start:
                        # Extract JPEG frame
                        jpeg_data = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        
                        # Decode JPEG
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            with self.lock:
                                self.current_frame = frame
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    break
            
            if self.process:
                self.process.terminate()

            self.error = f"{camera_command} stopped before producing frames"
            return False
                
        except Exception as e:
            self.error = f"{camera_command} streaming setup failed: {e}"
            logger.error(self.error)
            return False

    def _run_opencv_loop(self):
        """Fallback for USB/V4L2 camera access."""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                self.error = "OpenCV could not open /dev/video0"
                logger.error(self.error)
                return False

            self.method = "opencv"
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            failed_reads = 0
            while self.is_running:
                ok, frame = cap.read()
                if ok and frame is not None:
                    failed_reads = 0
                    with self.lock:
                        self.current_frame = cv2.resize(frame, (self.width, self.height))
                else:
                    failed_reads += 1
                    if failed_reads >= max(10, self.fps * 3):
                        cap.release()
                        self.error = "OpenCV opened /dev/video0 but read no frames"
                        logger.error(self.error)
                        return False
                time.sleep(1.0 / max(1, self.fps))

            cap.release()
            return True
        except Exception as e:
            self.error = f"OpenCV camera fallback failed: {e}"
            logger.error(self.error)
            return False
    
    def _fallback_placeholder_loop(self, message):
        """Generate placeholder frames when camera fails"""
        self.method = "placeholder"
        logger.info(f"Using placeholder frames due to camera failure: {self.error}")
        frame_counter = 0
        while self.is_running:
            frame_counter += 1
            frame_message = f"{message} - Frame {frame_counter}"
            with self.lock:
                self.current_frame = self._create_placeholder_frame(frame_message)
            time.sleep(0.1)
    
    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            else:
                return self._create_placeholder_frame()
    
    def get_jpeg_frame(self):
        """Get current frame as JPEG bytes for streaming"""
        frame = self.get_frame()
        
        # Add minimal status overlay (just timestamp)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (5, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return buffer.tobytes()
        else:
            # Return placeholder frame if encoding fails
            placeholder_frame = self._create_placeholder_frame("Encoding Error")
            ret, buffer = cv2.imencode('.jpg', placeholder_frame)
            return buffer.tobytes()
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_running = False
        if self.process:
            self.process.terminate()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)
        logger.info("Camera streaming stopped")
    
    def cleanup(self):
        """Clean shutdown of camera"""
        self.stop_streaming()
        logger.info("Camera cleaned up")

# Global camera instance
camera = CameraManager()
