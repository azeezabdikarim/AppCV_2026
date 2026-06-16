#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import time
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        """Initialize camera using libcamera-vid streaming"""
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.process = None
        self.current_frame = None
        
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
        """Start the camera streaming using libcamera-vid"""
        if not self.is_running:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info("Camera streaming started")
    
    def _capture_frames(self):
        """Continuously capture frames using libcamera-vid streaming (from working example)"""
        logger.info("Starting libcamera-vid streaming...")
        
        try:
            # Start libcamera-vid streaming to stdout (exact command from working example)
            cmd = [
                "libcamera-vid",
                "-t", "0",  # Infinite timeout
                "--width", "320",
                "--height", "240",
                "--framerate", "15",
                "-o", "-",  # Output to stdout
                "--codec", "mjpeg",
                "--inline",
                "-n"  # No preview
            ]
            
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Test if the process starts successfully
            time.sleep(1)
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read().decode('utf-8')
                logger.error(f"libcamera-vid failed to start. Error: {stderr_output}")
                self._fallback_placeholder_loop()
                return
            
            logger.info("libcamera-vid streaming started successfully")
            
            buffer = b""
            while self.is_running:
                try:
                    chunk = self.process.stdout.read(1024)
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
                                self.current_frame = frame.copy()
                
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    break
            
            if self.process:
                self.process.terminate()
                
        except Exception as e:
            logger.error(f"libcamera streaming setup failed: {e}")
            self._fallback_placeholder_loop()
    
    def _fallback_placeholder_loop(self):
        """Generate placeholder frames when camera fails"""
        logger.info("Using placeholder frames due to camera failure")
        frame_counter = 0
        while self.is_running:
            frame_counter += 1
            message = f"Camera Error - Frame {frame_counter}"
            with self.lock:
                self.current_frame = self._create_placeholder_frame(message)
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