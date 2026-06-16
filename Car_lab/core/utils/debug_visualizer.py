#!/usr/bin/env python3

import cv2
import numpy as np
import time

class DebugVisualizer:
    """Handles all debug visualization and overlay creation"""
    
    def __init__(self):
        """Initialize the debug visualizer"""
        pass
    
    def create_week2_debug_frame(self, original_frame, cache_manager, timing_utils, status_manager, sign_detector):
        """Create dual-panel visualization for Week 2 object detection and depth"""
        height, width = original_frame.shape[:2]
        
        # Left panel: Object detection overlay
        left_panel = original_frame.copy()
        if cache_manager.cached_detections:
            left_panel = self._draw_detection_overlay(left_panel, cache_manager.cached_detections)
        
        # Add detection panel label
        cv2.putText(left_panel, "Object Detection", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Right panel: Depth analysis
        right_panel = original_frame.copy()
        
        # Get cached depth map from SignDetector
        cached_depth_map = sign_detector.get_cached_depth_map() if sign_detector else None
        
        if cached_depth_map is not None:
            # Apply depth colormap
            depth_normalized = cv2.normalize(cached_depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
            right_panel = cv2.addWeighted(right_panel, 0.6, depth_colored, 0.4, 0)
            
            # Show frame counter alignment
            cv2.putText(right_panel, "[0]", (width - 40, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        else:
            # Show waiting status
            cv2.putText(right_panel, "Depth: Enable advanced mode", (10, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add depth panel label
        cv2.putText(right_panel, "Depth Analysis", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine panels side by side
        combined = np.hstack([left_panel, right_panel])
        
        # Add comprehensive debug information
        return self._add_week2_debug_overlay(combined, cache_manager, timing_utils, status_manager)
    
    def create_week3_debug_frame(self, original_frame, current_speed):
        """Simple placeholder for Week 3 - actual speed estimation uses create_speed_estimation_debug_frame"""
        speed_text = f"Speed: {current_speed:.1f} units/s"
        cv2.putText(original_frame, speed_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(original_frame, "Speed Estimation Mode (Week 3)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return original_frame

    def create_speed_estimation_debug_frame(self, original_frame, current_speed, speed_data):
        """Create comprehensive debug frame for Week 3 speed estimation mode"""
        debug_frame = original_frame.copy()
        height, width = debug_frame.shape[:2]
        
        # Get speed data
        smoothed_speed = speed_data.get('smoothed_speed', current_speed)
        motor_power = speed_data.get('motor_power', 0)
        test_active = speed_data.get('test_active', False)
        
        # Color coding based on speed thresholds
        if smoothed_speed > 0.4:
            color = (0, 0, 255)  # Red - fast
            status = "FAST"
        elif smoothed_speed > 0.15:
            color = (0, 165, 255)  # Orange - medium  
        elif smoothed_speed > 0.05:
            color = (0, 255, 0)  # Green - slow
            status = "SLOW"
        else:
            color = (128, 128, 128)  # Gray - stopped
            status = "STOPPED"
        
        # Large speed display in center (48px equivalent)
        speed_text = f"{smoothed_speed:.2f} m/s"
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        # Background rectangle for speed text
        padding = 20
        cv2.rectangle(debug_frame, 
                    (text_x - padding, text_y - text_size[1] - padding),
                    (text_x + text_size[0] + padding, text_y + padding),
                    (0, 0, 0), -1)
        
        # Speed text
        cv2.putText(debug_frame, speed_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Status text below speed
        status_y = text_y + 50
        cv2.putText(debug_frame, status, (text_x + 20, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Motor power display (top left)
        if test_active and motor_power > 0:
            power_text = f"Motor: {motor_power}% (TEST ACTIVE)"
            cv2.putText(debug_frame, power_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Speed history indicator (bottom)
        history = speed_data.get('speed_history', [])
        if len(history) > 1:
            avg_speed = sum(history[-5:]) / min(5, len(history))
            cv2.putText(debug_frame, f"Avg (5 frames): {avg_speed:.2f} m/s", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Current values (top right)
        cv2.putText(debug_frame, f"Raw: {current_speed:.3f}", (width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Smooth: {smoothed_speed:.3f}", (width - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame
    
    def create_speed_estimation_debug_frame(self, original_frame, current_speed, speed_data):
        """Create debug frame for Week 3 speed estimation mode"""
        debug_frame = original_frame.copy()
        height, width = debug_frame.shape[:2]
        
        # Add speed information overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Large speed display in center
        speed_text = f"Speed: {current_speed:.2f} m/s"
        font_scale = 1.2
        text_size = cv2.getTextSize(speed_text, font, font_scale, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        # Background rectangle for speed text
        cv2.rectangle(debug_frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        # Speed text
        color = (0, 255, 0) if current_speed > 0.1 else (128, 128, 128)
        cv2.putText(debug_frame, speed_text, (text_x, text_y), 
                   font, font_scale, color, 2)
        
        # Status information
        status_y = 30
        
        # Calibration status
        cal_status = "Calibrated" if speed_data.get('calibrated', False) else "Not Calibrated"
        cal_color = (0, 255, 0) if speed_data.get('calibrated', False) else (0, 0, 255)
        cv2.putText(debug_frame, f"Status: {cal_status}", (10, status_y), 
                   font, 0.6, cal_color, 2)
        
        # Motor power
        motor_power = speed_data.get('motor_power', 0)
        if motor_power > 0:
            cv2.putText(debug_frame, f"Motor: {motor_power}%", (10, status_y + 25), 
                       font, 0.6, (255, 255, 0), 2)
        
        # Speed history indicator
        history = speed_data.get('speed_history', [])
        if len(history) > 1:
            cv2.putText(debug_frame, f"Avg: {np.mean(history[-5:]):.2f} m/s", 
                       (10, height - 20), font, 0.5, (255, 255, 255), 1)
        
        return debug_frame
    
    def _draw_detection_overlay(self, frame, detections):
        """Draw bounding boxes and labels for detected objects"""
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            confidence = detection['confidence']
            class_name = detection.get('class_name', detection.get('class', 'object'))
            
            # Color coding for different object types
            if 'stop' in class_name.lower():
                color = (0, 0, 255)  # Red for stop signs
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            font_scale = 0.5
            thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Background for label
            cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def _add_week2_debug_overlay(self, combined_frame, cache_manager, timing_utils, status_manager):
        """Add comprehensive Week 2 debug information"""
        height, width = combined_frame.shape[:2]
        
        # Performance metrics
        detection_interval = getattr(timing_utils, 'detection_interval', 0.5)
        depth_interval = getattr(timing_utils, 'depth_interval', 1.0)
        
        detection_fps = 1.0 / detection_interval if detection_interval > 0 else 0
        depth_fps = 1.0 / depth_interval if depth_interval > 0 else 0
        
        current_time = time.time()
        
        # Get status from status manager
        stop_status = status_manager.get_current_status(current_time)
        
        # Top status line
        status_text = f"Status: {stop_status} | Detection: {detection_fps:.1f}fps ({timing_utils.last_detection_inference_ms:.0f}ms) | Depth: {depth_fps:.1f}fps ({timing_utils.last_depth_inference_ms:.0f}ms)"
        cv2.putText(combined_frame, status_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection list
        detection_text = f"Detections: {len(cache_manager.cached_detections)}"
        if cache_manager.cached_detections:
            top_detections = cache_manager.cached_detections[:3]  # Show top 3
            detection_names = [f"{d.get('class_name', d.get('class', 'obj'))}({d['confidence']:.2f})" 
                             for d in top_detections]
            detection_text += f" - {', '.join(detection_names)}"
        
        cv2.putText(combined_frame, detection_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return combined_frame