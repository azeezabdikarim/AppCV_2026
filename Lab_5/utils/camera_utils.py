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

def setup_camera(camera_index=0, raspberry_pi_url=None):
    """
    Initialize camera with optimal settings
    
    Args:
        camera_index: Local camera index (0, 1, 2, etc.) 
        raspberry_pi_url: URL for Raspberry Pi camera stream
                         Example: "http://cvpi33.local:5000/video"
    
    Returns:
        cv2.VideoCapture object
    """
    if raspberry_pi_url:
        print(f"Connecting to Raspberry Pi camera: {raspberry_pi_url}")
        cap = cv2.VideoCapture(raspberry_pi_url)
        # Give it a moment to connect
        time.sleep(1)
        if not cap.isOpened():
            print(f"Failed to connect to {raspberry_pi_url}")
            print("Falling back to local camera...")
            cap = cv2.VideoCapture(camera_index)
    else:
        print(f"Using local camera index: {camera_index}")
        cap = cv2.VideoCapture(camera_index)
    
    # Set resolution (will work for both local and stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def cleanup_camera(cap):
    """Properly release camera resources"""
    cap.release()
    cv2.destroyAllWindows()