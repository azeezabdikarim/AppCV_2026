#!/usr/bin/env python3

import threading
import time
import cv2
import numpy as np
import sys
import os

# Add the parent directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.movement_controls import MovementController
from utils.debug_visualizer import DebugVisualizer
from utils.utils import TimingUtils, CacheManager, StatusManager
from utils.console_logger import console_logger
from feature_config import FEATURES_ENABLED

try:
    from picarx import Picarx
except ImportError:
    print("WARNING: PicarX not available - running in simulation mode")
    Picarx = None

class RobotController:
    def __init__(self):
        """Initialize the robot controller"""
        try:
            # =================================================================
            # STUDENT TUNABLE PARAMETERS - Modify these as needed
            # =================================================================
            
            # Week 4: Object-detection performance and stopping behaviour
            self.detection_interval = 0.5    # Run object detection every 0.5 seconds
            
            # Debug and visualization settings
            self.debug_level = 0             # 0-4, higher = more debug info
            self.target_fps = 10             # Target frame rate for line following
            self.speed_flow_overlay_enabled = True
            
            # =================================================================
            # SYSTEM VARIABLES - Don't modify these directly
            # =================================================================
            
            # Initialize hardware controller
            self.movement_controller = MovementController()
            
            # Initialize utility managers
            self.timing_utils = TimingUtils()
            self.cache_manager = CacheManager()
            self.status_manager = StatusManager()
            
            # Initialize debug visualizer
            self.debug_visualizer = DebugVisualizer()
            
            # Debug mode system
            self.debug_mode = "object_detection"  # Week 4 default
            self.available_modes = ["line_following", "speed_estimation", "object_detection"]
            
            # Autonomous mode variables
            self.autonomous_mode = False
            self.frame_counter = 0
            self.previous_frame = None
            self.current_speed = 0.0
            self.sign_stop_until = None      # Time when sign stop expires
            
            # Performance tracking and timing
            self.last_frame_time = time.time()
            self.frame_interval = 1.0 / self.target_fps

            # Speed testing state
            self._current_test_speed = 0
            
            # Feature modules (loaded based on FEATURES_ENABLED)
            self.line_follower = None
            self.sign_detector = None
            self.speed_estimator = None
            
            # Feature status tracking
            self.feature_status = {
                'line_following': 'Disabled',
                'speed_estimation': 'Disabled',
                'sign_detection': 'Disabled'
            }
            
            # Debug data for sidebar (clean, minimal)
            self.debug_data = {
                'error_px': 0.0,
                'steering_angle': 0.0,
                'lines_detected': 0,
                'mode': 'Manual'
            }
            
            # Load enabled features
            self._load_enabled_features()

            # Week 4 observes signs in front of the car, not the floor.
            if FEATURES_ENABLED['sign_detection']:
                self.movement_controller.set_camera_pan(0)
                self.movement_controller.set_camera_tilt(0)
            
            print("✅ Robot controller initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing robot: {e}")
    
    def _load_enabled_features(self):
        """Load only the features that are explicitly enabled"""
        
        print("Loading enabled features...")
        
        # Week 2: Line Following
        if FEATURES_ENABLED['line_following']:
            try:
                # Clear any cached modules to force reload
                module_name = 'week2_line_following.line_follower'
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                from week2_line_following.line_follower import LineFollower
                self.line_follower = LineFollower()
                self.feature_status['line_following'] = 'Active'
                print("✅ Line following enabled and loaded")
                
            except Exception as e:
                self.feature_status['line_following'] = f'Error: {str(e)}'
                print(f"❌ Line following error: {e}")
        else:
            self.feature_status['line_following'] = 'Disabled'
            print("Line following disabled")
            
        # Week 4: Sign Detection
        if FEATURES_ENABLED['sign_detection']:
            try:
                if 'week4_object_detection.sign_detector' in sys.modules:
                    del sys.modules['week4_object_detection.sign_detector']
                    
                from week4_object_detection.sign_detector import SignDetector
                self.sign_detector = SignDetector()
                self.feature_status['sign_detection'] = 'Active'
                print("✅ Sign detection enabled and loaded")
            except Exception as e:
                self.feature_status['sign_detection'] = f'Error: {str(e)}'
                print(f"❌ Sign detection error: {e}")
        else:
            self.feature_status['sign_detection'] = 'Disabled'
            print("Sign detection disabled")
            
        # Week 3: Speed Estimation
        if FEATURES_ENABLED['speed_estimation']:
            try:
                if 'week3_speed_estimation.speed_estimator' in sys.modules:
                    del sys.modules['week3_speed_estimation.speed_estimator']
                    
                from week3_speed_estimation.speed_estimator import SpeedEstimator
                self.speed_estimator = SpeedEstimator()
                self.feature_status['speed_estimation'] = 'Active' 
                print("✅ Speed estimation enabled and loaded")
            except Exception as e:
                self.feature_status['speed_estimation'] = f'Error: {str(e)}'
                print(f"❌ Speed estimation error: {e}")
        else:
            self.feature_status['speed_estimation'] = 'Disabled'
            print("Speed estimation disabled")
        
        # Print final status
        print("Feature Status Summary:")
        for feature, status in self.feature_status.items():
            print(f"   {feature}: {status}")
    
    def start_autonomous_mode(self):
        """Start autonomous mode - either line following or straight movement"""
        console_logger.info("Starting autonomous mode...")
        
        if not self.movement_controller.is_hardware_connected():
            error_msg = "Robot hardware not connected"
            print(f"❌ {error_msg}")
            return False, error_msg
            
        self.autonomous_mode = True
        self.frame_counter = 0
        self.sign_stop_until = None
        self.status_manager.reset()
        
        # Describe the active autonomous exercise.
        if FEATURES_ENABLED['sign_detection'] and self.sign_detector:
            self.debug_data['mode'] = 'Sign-Stop Test'
            self.debug_mode = 'object_detection'
            console_logger.info("✅ Sign-stop test started at low speed")
            return True, "Sign-stop test started"
        elif FEATURES_ENABLED['line_following'] and self.line_follower:
            self.debug_data['mode'] = 'Line Following'
            console_logger.info("✅ Autonomous line following started")
            return True, "Line following started"
        else:
            self.debug_data['mode'] = 'Straight Movement'
            console_logger.info("✅ Autonomous straight movement started")
            return True, "Straight movement started"

    def calculate_speed_only(self, current_frame):
        """Minimal speed calculation without autonomous processing"""
        if self.speed_estimator and FEATURES_ENABLED['speed_estimation']:
            speed = self.speed_estimator.estimate_speed(current_frame, self.previous_frame)
            self.previous_frame = current_frame.copy()  # Store for next calculation
            return speed
        return 0.0
    
    def stop_autonomous_mode(self):
        """Stop autonomous mode and return to manual control"""
        self.autonomous_mode = False
        self.sign_stop_until = None
        self.status_manager.reset()
        self.debug_data['mode'] = 'Manual'
        self.movement_controller.emergency_stop()
        print("Autonomous mode stopped")
        return True, "Autonomous mode stopped"
    
    def process_autonomous_frame(self, frame):
        """Main processing pipeline with clean debug separation"""
        
        display_frame = frame.copy()
        
        # Initialize debug data
        self.debug_data = {
            'error_px': 0.0,
            'steering_angle': 0.0,
            'lines_detected': 0,
            'mode': 'Autonomous' if self.autonomous_mode else 'Manual'
        }
        
        # A Week 4 sign stop is latched until Start is pressed again.
        current_time = time.time()
        stopped_for_sign = (self.sign_stop_until is not None and current_time < self.sign_stop_until)
        
        # Week 4: Sign Detection (with timing and caching)
        if self.sign_detector and FEATURES_ENABLED['sign_detection'] and not stopped_for_sign:
            detected_signs = self._run_detection_with_timing(frame)
            if self.autonomous_mode and self.sign_detector.should_stop(detected_signs, frame):
                triggering_signs = [
                    detection
                    for detection in detected_signs
                    if detection['class_name'] in self.sign_detector.stop_classes
                    and detection['area_ratio'] >= self.sign_detector.stop_area_ratio
                ]
                trigger = max(
                    triggering_signs,
                    key=lambda detection: detection['area_ratio'],
                )
                self.sign_stop_until = float('inf')
                stopped_for_sign = True
                self.autonomous_mode = False
                self.debug_data['mode'] = 'Stopped for Sign'
                self.movement_controller.emergency_stop()
                console_logger.stop(
                    f"Stopped for {trigger['class_name']}: "
                    f"area={100 * trigger['area_ratio']:.1f}% "
                    f">= {100 * self.sign_detector.stop_area_ratio:.1f}%"
                )
        
        # Week 3: Speed Estimation
        if self.speed_estimator and FEATURES_ENABLED['speed_estimation']:
            self.current_speed = self.speed_estimator.estimate_speed(frame, self.previous_frame)
            self.debug_data['current_speed'] = round(self.current_speed, 1)
        
        # Week 2: Line Following (skip if stopped for sign)
        if self.line_follower and FEATURES_ENABLED['line_following']:
            try:
                steering_angle = self.line_follower.compute_steering_angle(frame, debug_level=self.debug_level)
                
                debug_frame = self.line_follower.get_debug_frame()
                if debug_frame is not None:
                    display_frame = debug_frame
                
                if hasattr(self.line_follower, 'current_debug_data'):
                    self.debug_data.update(self.line_follower.current_debug_data)
                
                # Apply control only if autonomous and not stopped for sign
                if self.autonomous_mode and not stopped_for_sign:
                    if current_time - self.last_frame_time >= self.frame_interval:
                        self.last_frame_time = current_time
                        self.frame_counter += 1
                        
                        self.movement_controller.apply_autonomous_control(steering_angle)
                
            except Exception as e:
                print(f"Line following error: {e}")
                self.feature_status['line_following'] = f'Runtime Error: {str(e)}'

        # FALLBACK: Straight movement if line following not available
        elif self.autonomous_mode and not stopped_for_sign:
            if current_time - self.last_frame_time >= self.frame_interval:
                self.last_frame_time = current_time
                self.frame_counter += 1
                
                # Just go straight (steering angle = 0)
                self.movement_controller.apply_autonomous_control(0)
        
        # Route debug visualization based on mode
        if self.debug_mode == "object_detection":
            display_frame = self.debug_visualizer.create_object_detection_debug_frame(
                display_frame,
                self.cache_manager,
                self.timing_utils,
                self.status_manager,
                self.sign_detector,
                self.status_manager.get_stop_status(time.time(), self.sign_stop_until),
            )
        elif self.debug_mode == "speed_estimation":
            flow_vectors = []
            if self.speed_estimator and hasattr(self.speed_estimator, 'get_flow_debug_data'):
                flow_vectors = self.speed_estimator.get_flow_debug_data().get('vectors', [])
            display_frame = self.debug_visualizer.create_speed_estimation_debug_frame(
                frame, flow_vectors, self.speed_flow_overlay_enabled
            )
        # Default: line_following mode uses existing debug frame
        
        # Store frame for next speed estimation
        self.previous_frame = frame.copy()
        
        return display_frame
    
    def _run_detection_with_timing(self, frame):
        """Run object detection with timing and caching (single inference execution)"""
        return self.timing_utils.run_detection_with_timing(
            frame,
            self.sign_detector,
            self.detection_interval,
            self.cache_manager,
        )
    
    # =============================================================================
    # DEBUG MODE CONTROL
    # =============================================================================
    
    def set_debug_mode(self, mode):
        """Switch debug visualization mode"""
        available_modes = self.get_available_debug_modes()
        if mode in available_modes:
            self.debug_mode = mode
            print(f"Debug mode set to: {mode}")
            return True
        else:
            print(f"❌ Invalid debug mode: {mode}. Available: {available_modes}")
            return False

    def get_available_debug_modes(self):
        """Return debug modes whose backing feature actually loaded."""
        mode_features = {
            "line_following": "line_following",
            "speed_estimation": "speed_estimation",
            "object_detection": "sign_detection",
        }
        return [
            mode
            for mode, feature in mode_features.items()
            if self.feature_status.get(feature) == "Active"
        ]
    
    def get_debug_mode_status(self):
        """Get current debug mode and performance metrics"""
        current_time = time.time()
        return {
            'debug_mode': self.debug_mode,
            'available_modes': self.get_available_debug_modes(),
            'detection_fps': 1.0 / self.detection_interval if self.detection_interval > 0 else 0,
            'last_detection_age': current_time - self.timing_utils.last_detection_time,
            'cached_detections': len(self.cache_manager.cached_detections),
            'last_detection_inference_ms': self.timing_utils.last_detection_inference_ms,
        }
    
    # =============================================================================
    # HARDWARE CONTROL DELEGATION
    # =============================================================================
    
    def set_camera_pan(self, angle):
        """Set camera pan angle (-90 to +90 degrees)"""
        return self.movement_controller.set_camera_pan(angle)
    
    def set_camera_tilt(self, angle):
        """Set camera tilt angle (-90 to +90 degrees)"""
        return self.movement_controller.set_camera_tilt(angle)
    
    def camera_look_down(self):
        """Preset: Point camera down for line following"""
        return self.movement_controller.camera_look_down()
    
    def camera_look_forward(self):
        """Preset: Point camera forward for obstacle detection"""
        return self.movement_controller.camera_look_forward()
    
    def move_forward(self, duration=0.5, speed=50):
        """Move robot forward for specified duration"""
        return self.movement_controller.move_forward(duration, speed, self.autonomous_mode)
    
    def move_backward(self, duration=0.5, speed=50):
        """Move robot backward for specified duration"""
        return self.movement_controller.move_backward(duration, speed, self.autonomous_mode)
    
    def turn_left(self, duration=0.5, speed=50, angle=-30):
        """Turn robot left while moving forward"""
        return self.movement_controller.turn_left(duration, speed, angle, self.autonomous_mode)
    
    def turn_right(self, duration=0.5, speed=50, angle=30):
        """Turn robot right while moving forward"""
        return self.movement_controller.turn_right(duration, speed, angle, self.autonomous_mode)
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        self.autonomous_mode = False
        self.movement_controller.emergency_stop()
    
    def cleanup(self):
        """Clean shutdown of robot"""
        self.emergency_stop()
        self.movement_controller.cleanup()
    
    # =============================================================================
    # DEBUG AND CONFIGURATION METHODS
    # =============================================================================
    
    def set_debug_level(self, level):
        """Set debug overlay selection as a bitmask (0-7).

        Views compose independently: ROI box=1, center+error=2, edges=4.
        0 means no overlay (off).
        """
        self.debug_level = max(0, min(7, level))
        print(f"Debug level set to: {self.debug_level}")
    
    def set_frame_rate(self, fps):
        """Set target frame rate"""
        self.target_fps = max(1, min(15, fps))
        self.frame_interval = 1.0 / self.target_fps
        print(f"Frame rate set to: {self.target_fps} fps")

    def set_speed_flow_overlay(self, enabled):
        """Show or hide optical-flow vectors in speed-estimation mode."""
        self.speed_flow_overlay_enabled = bool(enabled)
    
    def update_pid_parameters(self, kp=None, ki=None, kd=None):
        """Update PID parameters during runtime"""
        if self.line_follower and hasattr(self.line_follower, 'update_parameters'):
            self.line_follower.update_parameters(kp=kp, ki=ki, kd=kd)
            print(f"PID parameters updated: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            print("WARNING: Cannot update PID parameters - line follower not available")
    
    def get_debug_data(self):
        """Get clean debug data for sidebar"""
        data = self.debug_data.copy()
        
        # Add Week 4 specific data when in object detection mode
        if self.debug_mode == "object_detection":
            largest_area_ratio = max(
                (
                    detection.get('area_ratio', 0.0)
                    for detection in self.cache_manager.cached_detections
                ),
                default=0.0,
            )
            data.update({
                'detections_count': len(self.cache_manager.cached_detections),
                'detection_inference_ms': self.timing_utils.last_detection_inference_ms,
                'detection_fps': (
                    1.0 / self.detection_interval
                    if self.detection_interval > 0
                    else 0.0
                ),
                'largest_area_ratio': largest_area_ratio,
                'stop_area_ratio': getattr(self.sign_detector, 'stop_area_ratio', 0.0),
                'stop_status': self.status_manager.get_stop_status(time.time(), self.sign_stop_until)
            })
        
        return data
    
    def get_speed_data(self):
        """Get current speed data for speed estimation debug mode"""
        speed_data = {
            'current_speed': 0.0,
            'smoothed_speed': 0.0,
            'speed_history': [],
            'motor_power': 0,
            'test_active': False,
            'flow_magnitude': 0.0,
            'features_tracked': 0,
            'calibrated': False,
            'speed_thresholds': {
                'fast': 0.4,
                'medium': 0.15,
                'slow': 0.05
            }
        }
        
        try:
            if self.speed_estimator and FEATURES_ENABLED['speed_estimation']:
                # Get speed history from speed estimator
                if hasattr(self.speed_estimator, 'get_speed_history'):
                    history_data = self.speed_estimator.get_speed_history()
                    speed_data.update(history_data)
                
                # Add current speed
                speed_data['current_speed'] = round(self.current_speed, 3)
                speed_data['calibrated'] = self.speed_estimator.is_calibrated()

                if hasattr(self.speed_estimator, 'get_flow_debug_data'):
                    flow_debug_data = self.speed_estimator.get_flow_debug_data()
                    speed_data['flow_magnitude'] = flow_debug_data.get('flow_magnitude', 0.0)
                    speed_data['features_tracked'] = flow_debug_data.get('features_tracked', 0)
            
            # Add motor status from movement controller
            if hasattr(self, '_current_test_speed'):
                speed_data['motor_power'] = self._current_test_speed
                speed_data['test_active'] = self._current_test_speed > 0
            else:
                # Check if robot is moving manually
                if self.movement_controller.is_hardware_connected():
                    speed_data['test_active'] = self.movement_controller.is_moving
            
        except Exception as e:
            print(f"Speed data error: {e}")
        
        return speed_data
    
    def control_speed_test(self, action, speed_percent):
        """Control speed testing movements"""
        if not self.movement_controller.is_hardware_connected():
            return False, "Robot hardware not connected"
        
        if self.autonomous_mode:
            return False, "Cannot run speed test during autonomous mode"
        
        try:
            if action == 'start':
                # Start movement at specified speed
                self.movement_controller.picar.set_dir_servo_angle(0)  # Straight
                self.movement_controller.picar.forward(speed_percent)
                self._current_test_speed = speed_percent
                
                # Auto-stop after 3 seconds
                import threading
                def auto_stop():
                    try:
                        self.movement_controller.picar.stop()
                        self._current_test_speed = 0
                    except:
                        pass
                
                timer = threading.Timer(3.0, auto_stop)
                timer.start()
                self._speed_test_timer = timer
                
                return True, f"Speed test started at {speed_percent}% for 3 seconds"
                
            elif action == 'stop':
                # Manual stop
                self.movement_controller.picar.stop()
                self._current_test_speed = 0
                
                if hasattr(self, '_speed_test_timer'):
                    self._speed_test_timer.cancel()
                
                return True, "Speed test stopped"
                
        except Exception as e:
            return False, f"Speed test error: {str(e)}"
        
        return False, "Unknown action"
    
    def get_feature_status(self):
        """Return current status of all features"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'features': self.feature_status.copy(),
            'camera_position': self.movement_controller.get_camera_position(),
            'target_fps': self.target_fps,
            'debug_level': self.debug_level,
            'debug_mode': self.debug_mode,
            'available_modes': self.get_available_debug_modes(),
            'speed_flow_overlay_enabled': self.speed_flow_overlay_enabled
        }
    
    def get_autonomous_button_text(self):
        """Get the appropriate button text for autonomous mode"""
        if FEATURES_ENABLED['sign_detection'] and self.sign_detector:
            return "Start Sign-Stop Test"
        elif FEATURES_ENABLED['line_following'] and self.line_follower:
            return "Start Line Following"
        else:
            return "Start Straight Movement"

# Global robot instance
robot = RobotController()
