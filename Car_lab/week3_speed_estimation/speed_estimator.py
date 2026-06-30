#!/usr/bin/env python3

import cv2
import numpy as np
import json
import os

class SpeedEstimator:
    """
    Week 3 Implementation: Optical Flow Speed Estimation
    
    You will implement optical flow analysis to estimate robot speed
    from camera frames, following the Lucas-Kanade method from the academic paper.
    """
    
    def __init__(self):
        """Initialize optical flow parameters and load calibration"""
        
        # Optical flow parameters (you may need to tune these)
        self.feature_params = {
            'maxCorners': 100,      # Maximum number of features to track
            'qualityLevel': 0.3,    # Quality level for corner detection
            'minDistance': 7,       # Minimum distance between features
            'blockSize': 7          # Size of averaging block for corner detection
        }
        
        self.lk_params = {
            'winSize': (15, 15),    # Window size for Lucas-Kanade
            'maxLevel': 2,          # Maximum pyramid levels
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Speed calibration parameters (loaded from calibration file)
        self.calibration_loaded = False
        self.flow_to_speed_slope = 1.0
        self.flow_to_speed_intercept = 0.0
        
        # Internal state for tracking
        self.previous_gray = None
        self.previous_features = None

        self.speed_history = []
        self.max_history_length = 10

        # Latest tracked points are retained only for the optional Web UI overlay.
        self.last_flow_vectors = []
        self.last_flow_magnitude = 0.0
        self.last_features_tracked = 0
        
        # Exponential smoothing for speed
        self.smoothed_speed = 0.0
        self.alpha = 0.3  # Smoothing factor: 0.1=very smooth, 0.9=very responsive

        # Lucas-Kanade reports small sub-pixel displacements even for a stationary
        # camera. Treat coherent motion below this value as tracking jitter.
        self.stationary_flow_threshold = 0.5  # pixels per processed frame
        self.stationary_speed_threshold = 0.02  # m/s after calibration

        # Load calibration parameters
        self._load_calibration()
        
        print("SpeedEstimator initialized - you: Implement the helper functions!")
    
    def _load_calibration(self):
        """Load speed calibration parameters from file"""
        try:
            calibration_file = os.path.join(os.path.dirname(__file__), 'calibration_params.json')
            
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    params = json.load(f)
                
                self.flow_to_speed_slope = params.get('slope', 1.0)
                self.flow_to_speed_intercept = params.get('intercept', 0.0)
                self.calibration_loaded = True
                
                print(f"✅ Calibration loaded: slope={self.flow_to_speed_slope:.3f}, intercept={self.flow_to_speed_intercept:.3f}")
            else:
                print(f"⚠️  No calibration file found: {calibration_file}")
                print("   Using default parameters. Run speed calibration first!")
                
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
    
    def estimate_speed(self, current_frame, previous_frame):
        """
        Main speed estimation function - calls your-implemented helper functions
        
        Args:
            current_frame: numpy array of shape (240, 320, 3) - current RGB image
            previous_frame: numpy array of shape (240, 320, 3) - previous RGB image
            
        Returns:
            speed: float - estimated speed in units/second (calibrated)
        """
        
        self.last_flow_vectors = []
        self.last_flow_magnitude = 0.0
        self.last_features_tracked = 0

        if previous_frame is None:
            return self._record_stationary_speed()
        
        try:
            # Step 1: Convert to grayscale
            current_gray, previous_gray = self._convert_to_grayscale(current_frame, previous_frame)
            if current_gray is None or previous_gray is None:
                return self._record_stationary_speed()
            
            # Step 2: Detect features in previous frame
            features_prev = self._detect_features(previous_gray)
            if features_prev is None or len(features_prev) == 0:
                return self._record_stationary_speed()
            
            # Step 3: Track features using optical flow
            features_curr, status, error = self._track_features(previous_gray, current_gray, features_prev)
            if features_curr is None or status is None:
                return self._record_stationary_speed()
            
            # Step 4: Filter good features
            good_prev, good_curr = self._filter_good_features(features_prev, features_curr, status)
            self.last_features_tracked = len(good_prev)
            previous_points = np.asarray(good_prev).reshape(-1, 2)
            current_points = np.asarray(good_curr).reshape(-1, 2)
            self.last_flow_vectors = [
                (previous_point.tolist(), current_point.tolist())
                for previous_point, current_point in zip(previous_points, current_points)
            ]
            if len(good_prev) < 10:  # Need minimum features for reliable estimation
                return self._record_stationary_speed()
            
            # Step 5: Calculate flow magnitudes
            flow_magnitudes = self._calculate_flow_magnitudes(good_prev, good_curr)
            if not flow_magnitudes:
                return self._record_stationary_speed()
            
            # Step 6: Average flow magnitude
            finite_flow = np.asarray(flow_magnitudes, dtype=np.float32)
            finite_flow = finite_flow[np.isfinite(finite_flow)]
            if finite_flow.size == 0:
                return self._record_stationary_speed()

            avg_flow_magnitude = float(np.mean(finite_flow))
            median_flow_magnitude = float(np.median(finite_flow))
            self.last_flow_magnitude = float(avg_flow_magnitude)

            # The median prevents a few mistracked features from making a still
            # camera look as though it is moving.
            if median_flow_magnitude < self.stationary_flow_threshold:
                return self._record_stationary_speed()
            
            # Step 7: Convert to calibrated speed
            speed = self._convert_flow_to_speed(avg_flow_magnitude)

            if speed < self.stationary_speed_threshold:
                return self._record_stationary_speed()

            return self._record_moving_speed(speed)
            
        except Exception as e:
            print(f"Speed estimation error: {e}")
            return self._record_stationary_speed()

    def _append_speed_history(self, speed):
        """Store one bounded speed sample for the Web UI."""
        self.speed_history.append(float(speed))
        if len(self.speed_history) > self.max_history_length:
            self.speed_history.pop(0)

    def _record_stationary_speed(self):
        """Clear stale smoothing as soon as the optical flow is stationary."""
        self.smoothed_speed = 0.0
        self._append_speed_history(0.0)
        return 0.0

    def _record_moving_speed(self, speed):
        """Smooth and store a speed reading that passed the noise gates."""
        speed = float(speed)
        self.smoothed_speed = (
            self.alpha * speed + (1 - self.alpha) * self.smoothed_speed
        )
        self._append_speed_history(speed)
        return speed
    
    def _convert_to_grayscale(self, current_frame, previous_frame):
        """
        TODO: Convert RGB frames to grayscale for optical flow
        
        Args:
            current_frame: numpy array (240, 320, 3) - current RGB image
            previous_frame: numpy array (240, 320, 3) - previous RGB image
            
        Returns:
            tuple: (current_gray, previous_gray) both as numpy arrays
            
        Hint: Use cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """
        # TODO: Convert current_frame to grayscale
        current_gray = None
        
        # TODO: Convert previous_frame to grayscale
        previous_gray = None
        
        return current_gray, previous_gray
    
    def _detect_features(self, gray_image):
        """
        TODO: Detect good features to track in the grayscale image
        
        Args:
            gray_image: numpy array - grayscale image
            
        Returns:
            features: numpy array of feature points, or None if no features found
            
        Hint: Use cv2.goodFeaturesToTrack(gray_image, **self.feature_params)
        """
        # TODO: Detect corner features using cv2.goodFeaturesToTrack
        features = None
        
        return features
    
    def _track_features(self, previous_gray, current_gray, features_prev):
        """
        TODO: Track features from previous frame to current frame using Lucas-Kanade
        
        Args:
            previous_gray: numpy array - previous grayscale image
            current_gray: numpy array - current grayscale image  
            features_prev: numpy array - features from previous frame
            
        Returns:
            tuple: (features_curr, status, error) from optical flow
            
        Hint: Use cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, features_prev, None, **self.lk_params)
        """
        # TODO: Track features using Lucas-Kanade optical flow
        features_curr = None
        status = None
        error = None
        
        return features_curr, status, error
    
    def _filter_good_features(self, features_prev, features_curr, status):
        """
        TODO: Filter out features where tracking failed
        
        Args:
            features_prev: numpy array - features from previous frame
            features_curr: numpy array - features from current frame
            status: numpy array - tracking status (1=good, 0=failed)
            
        Returns:
            tuple: (good_prev, good_curr) - lists of successfully tracked features
            
        Hint: Loop through status array, keep features where status[i] == 1
        """
        good_prev = []
        good_curr = []
        
        # TODO: Filter features based on tracking status
        # Keep only features where status == 1 (successful tracking)
        
        return good_prev, good_curr
    
    def _calculate_flow_magnitudes(self, good_prev, good_curr):
        """
        TODO: Calculate the magnitude of optical flow vectors
        
        Args:
            good_prev: list of feature points from previous frame
            good_curr: list of feature points from current frame
            
        Returns:
            list: flow magnitudes for each feature pair
            
        Hint: For each point pair, calculate dx = curr_x - prev_x, dy = curr_y - prev_y
              Then magnitude = sqrt(dx*dx + dy*dy)
        """
        flow_magnitudes = []
        
        # TODO: Calculate flow magnitude for each feature pair
        # For each (prev_pt, curr_pt) pair:
        #   1. Calculate dx = curr_pt[0][0] - prev_pt[0][0]
        #   2. Calculate dy = curr_pt[0][1] - prev_pt[0][1]  
        #   3. Calculate magnitude = np.sqrt(dx*dx + dy*dy)
        #   4. Add magnitude to flow_magnitudes list
        
        return flow_magnitudes
    
    def _convert_flow_to_speed(self, avg_flow_magnitude):
        """
        TODO: Convert average flow magnitude to calibrated speed
        
        Args:
            avg_flow_magnitude: float - average optical flow magnitude
            
        Returns:
            float: calibrated speed in real-world units
            
        Hint: Apply linear calibration: speed = slope * flow + intercept
        """
        # TODO: Apply calibration to convert flow to speed
        if self.calibration_loaded:
            speed = None  # Use self.flow_to_speed_slope and self.flow_to_speed_intercept
        else:
            speed = avg_flow_magnitude  # Raw flow if no calibration
        
        # Ensure speed is non-negative
        speed = max(0.0, speed)
        return speed
    
    def is_calibrated(self):
        """Check if speed calibration parameters are available"""
        return self.calibration_loaded
    
    def get_speed_history(self):
        """Get speed history for web interface and smoothing"""
        if not self.speed_history:
            return {
                'current_speed': 0.0,
                'smoothed_speed': 0.0,
                'speed_history': [],
                'avg_speed': 0.0
            }
        
        return {
            'current_speed': self.speed_history[-1],
            'smoothed_speed': self.smoothed_speed,
            'speed_history': self.speed_history.copy(),
            'avg_speed': sum(self.speed_history) / len(self.speed_history)
        }

    def get_flow_debug_data(self):
        """Return the latest optical-flow tracks for visualization."""
        return {
            'vectors': self.last_flow_vectors.copy(),
            'flow_magnitude': self.last_flow_magnitude,
            'features_tracked': self.last_features_tracked,
        }

# =============================================================================
# QUICK REFERENCE FOR
# =============================================================================

"""
KEY OPENCV FUNCTIONS TO USE:

1. cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   - Converts color image to grayscale

2. cv2.goodFeaturesToTrack(gray_image, **params)
   - Detects corner features good for tracking
   - Returns array of (x,y) coordinates

3. cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_features, None, **params)
   - Tracks features between frames using Lucas-Kanade
   - Returns: (new_features, status, error)

4. Feature point access:
   - prev_pt[0][0] = x coordinate
   - prev_pt[0][1] = y coordinate

5. Flow magnitude calculation:
   - dx = new_x - old_x
   - dy = new_y - old_y  
   - magnitude = sqrt(dx² + dy²)
"""
