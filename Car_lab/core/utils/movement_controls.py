#!/usr/bin/env python3

import threading
import time
from .console_logger import console_logger


try:
    from picarx import Picarx
except ImportError:
    print("WARNING: PicarX not available - running in simulation mode")
    Picarx = None

class MovementController:
    """Handles all hardware movement and camera control operations"""
    
    def __init__(self):
        """Initialize the movement controller with hardware"""
        try:
            if Picarx:
                self.picar = Picarx()
                print("✅ PiCar-X hardware connected")
            else:
                self.picar = None
                print("WARNING: Running without PiCar-X hardware")
                
            self.is_moving = False
            
            # Camera positioning
            self.camera_pan_angle = 0   # -90 to +90 degrees
            self.camera_tilt_angle = -30  # Start looking down for line following
            
            # Initialize camera position
            self._set_camera_position()
            
        except Exception as e:
            print(f"❌ Error initializing movement hardware: {e}")
            self.picar = None
    
    def _set_camera_position(self):
        """Set camera to initial position"""
        if self.picar:
            try:
                self.picar.set_cam_pan_angle(self.camera_pan_angle)
                self.picar.set_cam_tilt_angle(self.camera_tilt_angle)
                print(f"Camera positioned: pan={self.camera_pan_angle}°, tilt={self.camera_tilt_angle}°")
            except Exception as e:
                print(f"WARNING: Camera positioning error: {e}")
    
    def is_hardware_connected(self):
        """Check if hardware is connected and available"""
        return self.picar is not None
    
    def get_camera_position(self):
        """Get current camera position"""
        return {
            'pan': self.camera_pan_angle,
            'tilt': self.camera_tilt_angle
        }
    
    # =============================================================================
    # CAMERA CONTROL METHODS
    # =============================================================================
    
    def set_camera_pan(self, angle):
        """Set camera pan angle (-90 to +90 degrees)"""
        angle = max(-90, min(90, angle))
        self.camera_pan_angle = angle
        
        if self.picar:
            try:
                self.picar.set_cam_pan_angle(angle)
                print(f"Camera pan set to {angle}°")
                return True
            except Exception as e:
                print(f"❌ Camera pan error: {e}")
                return False
        return False
    
    def set_camera_tilt(self, angle):
        """Set camera tilt angle (-90 to +90 degrees)"""
        angle = max(-90, min(90, angle))
        self.camera_tilt_angle = angle
        
        if self.picar:
            try:
                self.picar.set_cam_tilt_angle(angle)
                print(f"Camera tilt set to {angle}°")
                return True
            except Exception as e:
                print(f"❌ Camera tilt error: {e}")
                return False
        return False
    
    def camera_look_down(self):
        """Preset: Point camera down for line following"""
        return self.set_camera_pan(0) and self.set_camera_tilt(-30)
    
    def camera_look_forward(self):
        """Preset: Point camera forward for obstacle detection"""
        return self.set_camera_pan(0) and self.set_camera_tilt(0)
    
    # =============================================================================
    # AUTONOMOUS MOVEMENT CONTROL
    # =============================================================================
    
    def apply_autonomous_control(self, steering_angle):
        """Apply steering and forward movement for autonomous mode"""
        if self.picar and not self.is_moving:
            try:
                self.picar.set_dir_servo_angle(steering_angle)
                self.picar.forward(1)  # 1% speed
            except Exception as e:
                print(f"Autonomous control error: {e}")
    
    # =============================================================================
    # MANUAL MOVEMENT METHODS
    # =============================================================================
    
    def _auto_stop(self):
        """Automatically stop the robot and center wheels after movement"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False

    def stop(self):
        """Stop the robot (gentler than emergency_stop)"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)  # Center the wheels
            self.is_moving = False
        console_logger.info("Robot stopped")
    
    def move_forward(self, duration=0.5, speed=50, autonomous_mode=False):
        """Move robot forward for specified duration"""
        if not self.picar or self.is_moving or autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error moving forward: {e}")
            self._auto_stop()
            return False
    
    def move_backward(self, duration=0.5, speed=50, autonomous_mode=False):
        """Move robot backward for specified duration"""
        if not self.picar or self.is_moving or autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(0)
            self.picar.backward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error moving backward: {e}")
            self._auto_stop()
            return False
    
    def turn_left(self, duration=0.5, speed=50, angle=-30, autonomous_mode=False):
        """Turn robot left while moving forward"""
        if not self.picar or self.is_moving or autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error turning left: {e}")
            self._auto_stop()
            return False
    
    def turn_right(self, duration=0.5, speed=50, angle=30, autonomous_mode=False):
        """Turn robot right while moving forward"""
        if not self.picar or self.is_moving or autonomous_mode:
            return False
        
        try:
            self.is_moving = True
            self.picar.set_dir_servo_angle(angle)
            self.picar.forward(speed)
            timer = threading.Timer(duration, self._auto_stop)
            timer.start()
            return True
        except Exception as e:
            print(f"Error turning right: {e}")
            self._auto_stop()
            return False
    
    def emergency_stop(self):
        """Immediately stop the robot"""
        if self.picar:
            self.picar.stop()
            self.picar.set_dir_servo_angle(0)
            self.is_moving = False
        print("Emergency stop activated")
    
    def cleanup(self):
        """Clean shutdown of movement controller"""
        self.emergency_stop()
        print("Movement controller cleaned up")