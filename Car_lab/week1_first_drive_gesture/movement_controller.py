#!/usr/bin/env python3

import threading

try:
    from picarx import Picarx
except ImportError:
    Picarx = None


class MovementController:
    """Small PiCar-X movement layer for the first-drive gesture lab."""

    STEERING_ANGLES = {
        "left": -28,
        "straight": 0,
        "right": 28,
    }

    def __init__(self, forward_speed=8, pulse_speed=10, pulse_duration=0.45):
        self.forward_speed = forward_speed
        self.pulse_speed = pulse_speed
        self.pulse_duration = pulse_duration
        self.forward_active = False
        self.current_direction = "straight"
        self.current_angle = 0
        self.last_command = "none"
        self.command_count = 0
        self._pulse_timer = None
        self._lock = threading.Lock()

        if Picarx:
            try:
                self.picar = Picarx()
                self.picar.set_dir_servo_angle(0)
            except Exception as exc:
                print(f"PiCar-X hardware initialization failed: {exc}")
                self.picar = None
        else:
            print("picarx module not available; running movement controller in simulation mode.")
            self.picar = None

    def is_hardware_connected(self):
        return self.picar is not None

    def start_forward(self):
        with self._lock:
            self.forward_active = True
            self.last_command = "start_forward"
            self.command_count += 1
            self._apply_steering_locked(self.current_direction)
            if self.picar:
                self.picar.forward(self.forward_speed)
        return True, f"Forward drive started at {self.forward_speed}% speed"

    def stop(self):
        with self._lock:
            self.forward_active = False
            self.current_direction = "straight"
            self.current_angle = 0
            self.last_command = "stop"
            self.command_count += 1
            if self._pulse_timer:
                self._pulse_timer.cancel()
                self._pulse_timer = None
            if self.picar:
                self.picar.stop()
                self.picar.set_dir_servo_angle(0)
        return True, "Car stopped"

    def emergency_stop(self):
        return self.stop()

    def apply_gesture_direction(self, direction):
        if direction not in self.STEERING_ANGLES:
            return False, f"Unknown steering direction: {direction}"

        with self._lock:
            self._apply_steering_locked(direction)
            self.last_command = f"gesture_{direction}"
            if self.forward_active and self.picar:
                self.picar.forward(self.forward_speed)

        return True, f"Steering set to {direction}"

    def manual_pulse(self, direction):
        if direction == "stop":
            return self.stop()

        if direction not in {"forward", "backward", "left", "right"}:
            return False, f"Unknown manual direction: {direction}"

        with self._lock:
            if self.forward_active:
                return False, "Stop forward mode before using manual movement checks"

            self.command_count += 1
            self.last_command = f"manual_{direction}"

            if direction == "left":
                self._apply_steering_locked("left")
            elif direction == "right":
                self._apply_steering_locked("right")
            else:
                self._apply_steering_locked("straight")

            if self.picar:
                if direction == "backward":
                    self.picar.backward(self.pulse_speed)
                else:
                    self.picar.forward(self.pulse_speed)

            if self._pulse_timer:
                self._pulse_timer.cancel()
            self._pulse_timer = threading.Timer(self.pulse_duration, self._finish_pulse)
            self._pulse_timer.daemon = True
            self._pulse_timer.start()

        return True, f"Manual {direction} pulse started"

    def _finish_pulse(self):
        with self._lock:
            if self.forward_active:
                return
            if self.picar:
                self.picar.stop()
                self.picar.set_dir_servo_angle(0)
            self.current_direction = "straight"
            self.current_angle = 0
            self._pulse_timer = None

    def _apply_steering_locked(self, direction):
        self.current_direction = direction
        self.current_angle = self.STEERING_ANGLES[direction]
        if self.picar:
            self.picar.set_dir_servo_angle(self.current_angle)

    def get_status(self):
        with self._lock:
            return {
                "hardware_connected": self.is_hardware_connected(),
                "forward_active": self.forward_active,
                "current_direction": self.current_direction,
                "current_angle": self.current_angle,
                "forward_speed": self.forward_speed,
                "last_command": self.last_command,
                "command_count": self.command_count,
            }

    def cleanup(self):
        self.stop()
