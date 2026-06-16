#!/usr/bin/env python3

import time
import cv2
import numpy as np

class TimingUtils:
    """Handles timing and performance tracking for inference operations"""
    
    def __init__(self):
        self.last_detection_time = 0
        self.last_depth_time = 0
        self.last_detection_inference_ms = 0
        self.last_depth_inference_ms = 0
        self.detection_interval = 0.5
        self.depth_interval = 1.0
    
    def run_detection_with_timing(self, frame, sign_detector, detection_interval, depth_interval, cache_manager):
        """Run object detection with timing and caching (single inference execution)"""
        current_time = time.time()
        
        # Store intervals for later use
        self.detection_interval = detection_interval
        self.depth_interval = depth_interval
        
        # Only run detection if enough time has passed
        if current_time - self.last_detection_time >= detection_interval:
            start_time = time.perf_counter()
            detected_signs = sign_detector.detect_signs(frame)
            self.last_detection_inference_ms = (time.perf_counter() - start_time) * 1000
            
            # Cache results for debugging
            cache_manager.update_detections(detected_signs)
            self.last_detection_time = current_time
            
            # Run depth analysis if we have detections and it's time
            if detected_signs and self._should_run_depth(current_time):
                self._run_depth_with_timing(frame, detected_signs, cache_manager)
            
            return detected_signs
        else:
            # Return cached results - no additional inference
            return cache_manager.cached_detections
    
    def _should_run_depth(self, current_time):
        """Check if it's time to run depth analysis"""
        return current_time - self.last_depth_time >= self.depth_interval
    
    def _run_depth_with_timing(self, frame, detections, cache_manager):
        """Run depth estimation with timing (placeholder for MiDaS integration)"""
        start_time = time.perf_counter()
        
        # Placeholder: Simple area-based "depth" for now
        # Students will replace this with actual MiDaS integration
        depth_map = self._simple_area_based_depth(frame, detections)
        
        self.last_depth_inference_ms = (time.perf_counter() - start_time) * 1000
        cache_manager.update_depth(depth_map)
        self.last_depth_time = time.time()
    
    def _simple_area_based_depth(self, frame, detections):
        """Simple area-based depth estimation (students will replace with MiDaS)"""
        # Create a simple depth visualization based on bounding box areas
        height, width = frame.shape[:2]
        depth_map = np.zeros((height, width), dtype=np.uint8)
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            area = w * h
            
            # Larger bounding box = closer = higher depth value
            depth_value = min(255, int(area / 100))  # Simple area-to-depth conversion
            cv2.rectangle(depth_map, (x, y), (x + w, y + h), depth_value, -1)
        
        return depth_map

class CacheManager:
    """Manages cached results for debugging and visualization"""
    
    def __init__(self):
        self.cached_detections = []
        self.cached_depth_map = None
        self.detection_frame_counter = 0
        self.depth_frame_counter = 0
    
    def update_detections(self, detections):
        """Update cached detection results"""
        self.cached_detections = detections
        self.detection_frame_counter += 1
    
    def update_depth(self, depth_map):
        """Update cached depth results"""
        self.cached_depth_map = depth_map
        self.depth_frame_counter = self.detection_frame_counter

class StatusManager:
    """Manages robot status and anti-infinite-stop logic"""
    
    def __init__(self):
        self.recently_stopped_for_sign = False
        self.stop_cooldown_until = None
    
    def set_recently_stopped(self, value):
        """Set the recently stopped flag"""
        self.recently_stopped_for_sign = value
    
    def start_cooldown(self, current_time, cooldown_duration):
        """Start the stop cooldown period"""
        self.stop_cooldown_until = current_time + cooldown_duration
        self.recently_stopped_for_sign = False
    
    def is_in_cooldown(self, current_time):
        """Check if currently in stop cooldown period"""
        return (self.stop_cooldown_until is not None and 
                current_time < self.stop_cooldown_until)
    
    def get_stop_status(self, current_time, sign_stop_until=None):
        """Get current stopping status"""
        if sign_stop_until is not None and current_time < sign_stop_until:
            return "STOPPED"
        elif self.is_in_cooldown(current_time):
            return "COOLDOWN"
        else:
            return "ACTIVE"
    
    def get_current_status(self, current_time):
        """Get current status for debug display"""
        if self.is_in_cooldown(current_time):
            return "COOLDOWN"
        else:
            return "ACTIVE"