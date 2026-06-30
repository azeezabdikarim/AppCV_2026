#!/usr/bin/env python3

import cv2
import numpy as np
import time

class DebugVisualizer:
    """Handles all debug visualization and overlay creation"""
    
    def __init__(self):
        """Initialize the debug visualizer"""
        pass
    
    def create_object_detection_debug_frame(self, original_frame, cache_manager, timing_utils, status_manager, sign_detector):
        """Create dual-panel visualization for object detection and depth"""
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
        return self._add_object_detection_debug_overlay(combined, cache_manager, timing_utils, status_manager)
    
    def create_speed_estimation_debug_frame(self, original_frame, flow_vectors, enabled):
        """Draw the latest optical-flow tracks without covering the camera feed."""
        debug_frame = original_frame.copy()
        if not enabled:
            return debug_frame

        for previous_point, current_point in flow_vectors:
            points = np.asarray([previous_point, current_point], dtype=np.float32)
            if points.shape != (2, 2) or not np.all(np.isfinite(points)):
                continue

            previous = tuple(np.rint(points[0]).astype(int))
            current = tuple(np.rint(points[1]).astype(int))
            cv2.arrowedLine(
                debug_frame,
                previous,
                current,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.3,
            )
            cv2.circle(debug_frame, current, 2, (0, 255, 0), -1, cv2.LINE_AA)

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
    
    def _add_object_detection_debug_overlay(self, combined_frame, cache_manager, timing_utils, status_manager):
        """Add comprehensive object-detection debug information"""
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
