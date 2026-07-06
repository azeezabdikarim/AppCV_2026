#!/usr/bin/env python3

import time

class TimingUtils:
    """Handles timing and performance tracking for inference operations"""
    
    def __init__(self):
        self.last_detection_time = 0
        self.last_detection_inference_ms = 0
        self.detection_interval = 0.5
    
    def run_detection_with_timing(self, frame, sign_detector, detection_interval, cache_manager):
        """Run object detection with timing and caching (single inference execution)"""
        current_time = time.time()
        
        # Store intervals for later use
        self.detection_interval = detection_interval
        
        # Only run detection if enough time has passed
        if current_time - self.last_detection_time >= detection_interval:
            start_time = time.perf_counter()
            detected_signs = sign_detector.detect_signs(frame)
            self.last_detection_inference_ms = (time.perf_counter() - start_time) * 1000
            
            # Cache results for debugging
            cache_manager.update_detections(detected_signs)
            self.last_detection_time = current_time
            return detected_signs
        else:
            # Return cached results - no additional inference
            return cache_manager.cached_detections

class CacheManager:
    """Manages cached results for debugging and visualization"""
    
    def __init__(self):
        self.cached_detections = []
        self.detection_frame_counter = 0
    
    def update_detections(self, detections):
        """Update cached detection results"""
        self.cached_detections = detections
        self.detection_frame_counter += 1

class StatusManager:
    """Manages robot status and anti-infinite-stop logic"""
    
    def __init__(self):
        self.recently_stopped_for_sign = False
        self.stop_cooldown_until = None
    
    def set_recently_stopped(self, value):
        """Set the recently stopped flag"""
        self.recently_stopped_for_sign = value

    def reset(self):
        """Clear sign-stop and cooldown state before a new run."""
        self.recently_stopped_for_sign = False
        self.stop_cooldown_until = None
    
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
