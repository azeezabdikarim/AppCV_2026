import cv2
import time

class FPSCalculator:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
    
    def update(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps

def setup_camera(camera_index=0):
    """Initialize camera with optimal settings"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def cleanup_camera(cap):
    """Properly release camera resources"""
    cap.release()
    cv2.destroyAllWindows()