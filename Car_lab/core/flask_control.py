#!/usr/bin/env python3

from flask import Flask, render_template, Response, jsonify, request
import atexit
import signal
import sys
import socket
from robot_controller import robot
from camera_manager import camera
from utils.console_logger import console_logger


# Initialize Flask app
app = Flask(__name__)

# Global state for tracking commands
last_command = "none"
command_count = 0

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip.startswith("127."):
                import subprocess
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                if result.returncode == 0:
                    addresses = result.stdout.strip().split()
                    if addresses:
                        local_ip = addresses[0]
            return local_ip
        except Exception:
            return "localhost"

def cleanup_handler():
    """Clean up resources on exit"""
    print("\nShutting down...")
    robot.cleanup()
    camera.cleanup()

# Register cleanup handlers
atexit.register(cleanup_handler)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

@app.route('/')
def index():
    """Main control page"""
    return render_template('control.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with autonomous processing"""
    def generate():
        while True:
            # Get raw frame from camera
            frame = camera.get_frame()
            
            # Process frame for autonomous features and debug visualization
            processed_frame = robot.process_autonomous_frame(frame)
            
            # Convert to JPEG for streaming
            import cv2
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = camera.get_jpeg_frame()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =============================================================================
# AUTONOMOUS CONTROL ROUTES
# =============================================================================

@app.route('/start_autonomous')
def start_autonomous():
    """Start autonomous line following mode"""
    success, message = robot.start_autonomous_mode()
    global last_command
    last_command = "autonomous started" if success else "autonomous failed"
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/stop_autonomous')
def stop_autonomous():
    """Stop autonomous mode"""
    success, message = robot.stop_autonomous_mode()
    global last_command
    last_command = "autonomous stopped"
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/autonomous_status')
def get_autonomous_status():
    """Get detailed autonomous system status"""
    status = robot.get_feature_status()
    
    # Let the robot controller determine the button text
    button_text = robot.get_autonomous_button_text()
    status['button_text'] = button_text
    
    return jsonify(status)

@app.route('/console_logs')
def get_console_logs():
    """Get recent console log messages"""
    messages = console_logger.get_recent_messages(limit=50)
    return jsonify({'messages': messages})

# =============================================================================
# CAMERA CONTROL ROUTES (FIXED - Using Query Parameters)
# =============================================================================

@app.route('/set_camera_pan')
def set_camera_pan():
    """Set camera pan angle using query parameter"""
    try:
        angle = request.args.get('angle', type=int)
        if angle is None:
            return jsonify({
                'success': False,
                'message': 'Angle parameter required'
            }), 400
        
        success = robot.set_camera_pan(angle)
        return jsonify({
            'success': success,
            'angle': angle,
            'message': f'Camera pan set to {angle}°' if success else 'Camera pan failed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting camera pan: {str(e)}'
        }), 500

@app.route('/set_camera_tilt')
def set_camera_tilt():
    """Set camera tilt angle using query parameter"""
    try:
        angle = request.args.get('angle', type=int)
        if angle is None:
            return jsonify({
                'success': False,
                'message': 'Angle parameter required'
            }), 400
        
        success = robot.set_camera_tilt(angle)
        return jsonify({
            'success': success,
            'angle': angle,
            'message': f'Camera tilt set to {angle}°' if success else 'Camera tilt failed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting camera tilt: {str(e)}'
        }), 500

@app.route('/camera_look_down')
def camera_look_down():
    """Preset: Point camera down for line following"""
    success = robot.camera_look_down()
    return jsonify({
        'success': success,
        'message': 'Camera pointed down for line following' if success else 'Camera positioning failed'
    })

@app.route('/camera_look_forward')
def camera_look_forward():
    """Preset: Point camera forward"""
    success = robot.camera_look_forward()
    return jsonify({
        'success': success,
        'message': 'Camera pointed forward' if success else 'Camera positioning failed'
    })

# =============================================================================
# DEBUG DATA ROUTES
# =============================================================================

@app.route('/debug_data')
def get_debug_data():
    """Get clean debug data for sidebar"""
    return jsonify(robot.get_debug_data())

@app.route('/set_debug_level')
def set_debug_level():
    """Set debugging visualization level using query parameter"""
    try:
        level = request.args.get('level', type=int)
        if level is None:
            return jsonify({
                'success': False,
                'message': 'Level parameter required'
            }), 400
        
        robot.set_debug_level(level)
        return jsonify({
            'success': True,
            'debug_level': robot.debug_level,
            'message': f'Debug level set to {level}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting debug level: {str(e)}'
        }), 500

@app.route('/set_frame_rate')
def set_frame_rate():
    """Set frame rate using query parameter"""
    try:
        fps = request.args.get('fps', type=int)
        if fps is None:
            return jsonify({
                'success': False,
                'message': 'FPS parameter required'
            }), 400
        
        robot.set_frame_rate(fps)
        return jsonify({
            'success': True,
            'frame_rate': robot.target_fps,
            'message': f'Frame rate set to {fps} fps'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting frame rate: {str(e)}'
        }), 500
    
@app.route('/set_debug_mode')
def set_debug_mode():
    """Set debug visualization mode using query parameter"""
    try:
        mode = request.args.get('mode', 'line_following')
        success = robot.set_debug_mode(mode)
        return jsonify({
            'success': success,
            'debug_mode': robot.debug_mode,
            'available_modes': robot.available_modes,
            'message': f'Debug mode set to {mode}' if success else f'Invalid debug mode: {mode}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error setting debug mode: {str(e)}'
        }), 500

@app.route('/get_week2_performance')
def get_week2_performance():
    """Get Week 2 specific performance metrics"""
    try:
        return jsonify(robot.get_debug_mode_status())
    except Exception as e:
        return jsonify({
            'error': f'Error getting performance data: {str(e)}'
        }), 500

@app.route('/update_pid_parameters')
def update_pid_parameters():
    """Update PID parameters via web interface"""
    kp = request.args.get('kp', type=float)
    ki = request.args.get('ki', type=float)
    kd = request.args.get('kd', type=float)
    
    robot.update_pid_parameters(kp=kp, ki=ki, kd=kd)
    
    return jsonify({
        'success': True,
        'message': 'PID parameters updated',
        'kp': kp,
        'ki': ki, 
        'kd': kd
    })

# =============================================================================
# EXISTING MANUAL CONTROL ROUTES (unchanged)
# =============================================================================

@app.route('/move/<direction>')
def move_robot(direction):
    """Handle movement commands"""
    global last_command, command_count
    
    if robot.autonomous_mode:
        return jsonify({
            'success': False,
            'message': 'Manual control disabled during autonomous mode',
            'command_count': command_count
        })
    
    command_count += 1
    success = False
    
    try:
        if direction == 'forward':
            success = robot.move_forward(duration=0.5, speed=50)
            last_command = "forward"
            
        elif direction == 'backward':
            success = robot.move_backward(duration=0.5, speed=50)
            last_command = "backward"
            
        elif direction == 'left':
            success = robot.turn_left(duration=0.5, speed=50, angle=-30)
            last_command = "turn left"
            
        elif direction == 'right':
            success = robot.turn_right(duration=0.5, speed=50, angle=30)
            last_command = "turn right"
            
        elif direction == 'stop':
            robot.emergency_stop()
            last_command = "emergency stop"
            success = True
            
        else:
            return jsonify({
                'success': False, 
                'message': f'Unknown direction: {direction}',
                'command_count': command_count
            })
        
        return jsonify({
            'success': success,
            'direction': direction,
            'message': f'Command {direction} {"executed" if success else "failed"}',
            'command_count': command_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error executing {direction}: {str(e)}',
            'command_count': command_count
        })

@app.route('/status')
def get_status():
    """Get current robot status"""
    return jsonify({
        'last_command': last_command,
        'command_count': command_count,
        'is_moving': robot.movement_controller.is_moving if robot.movement_controller.is_hardware_connected() else False,
        'robot_connected': robot.movement_controller.is_hardware_connected(),
        'autonomous_mode': robot.autonomous_mode
    })

@app.route('/test')
def test_robot():
    """Test basic robot functionality"""
    if robot.picar is None:
        return jsonify({
            'success': False,
            'message': 'Robot not connected'
        })
    
    if robot.autonomous_mode:
        return jsonify({
            'success': False,
            'message': 'Cannot test during autonomous mode'
        })
    
    try:
        robot.move_forward(duration=0.2, speed=30)
        return jsonify({
            'success': True,
            'message': 'Test movement completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Test failed: {str(e)}'
        })
    
@app.route('/speed_data')
def get_speed_data():
    """Get current speed data for speed estimation debug mode"""
    try:
        speed_data = robot.get_speed_data()
        
        # Add real-time timestamp
        import time
        speed_data['timestamp'] = time.time()
        
        return jsonify(speed_data)
    except Exception as e:
        return jsonify({
            'error': f'Speed data error: {str(e)}',
            'current_speed': 0.0,
            'smoothed_speed': 0.0,
            'speed_history': [],
            'motor_power': 0,
            'test_active': False,
            'timestamp': time.time()
        }), 500

@app.route('/speed_test_control')
def speed_test_control():
    """Control speed testing with discrete power levels and auto-stop"""
    try:
        action = request.args.get('action', 'stop')  # start, stop, emergency_stop
        speed = request.args.get('speed', type=int, default=30)
        
        if robot.autonomous_mode:
            return jsonify({
                'success': False,
                'message': 'Speed testing disabled during autonomous mode'
            })
        
        if action == 'start':
            # Validate speed levels
            if speed not in [30, 50, 70]:
                return jsonify({
                    'success': False,
                    'message': f'Invalid speed level: {speed}. Use 30, 50, or 70.'
                })
        
        success, message = robot.control_speed_test(action, speed)
        
        return jsonify({
            'success': success,
            'message': message,
            'action': action,
            'speed': speed if action == 'start' else 0,
            'duration': 3 if action == 'start' else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Speed test error: {str(e)}'
        }), 500

@app.route('/process_speed_frames')
def process_speed_frames():
    """Lightweight speed estimation without video streaming"""
    try:
        # Get raw frame from camera (no encoding)
        current_frame = camera.get_frame()
        
        # Run speed estimation only (no autonomous processing)
        speed = robot.calculate_speed_only(current_frame)
        
        # Return minimal JSON (not full debug data)
        return jsonify({
            'current_speed': speed,
            'smoothed_speed': robot.get_smoothed_speed(),
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'current_speed': 0.0, 'smoothed_speed': 0.0, 'error': str(e)})

if __name__ == '__main__':
    try:
        camera.start_streaming()
        local_ip = get_local_ip()
        
        print("Starting Flask server...")
        print("Open http://localhost:5000 in your browser")
        print(f"Or from another device: http://{local_ip}:5000")
        print("Use camera controls to position camera, then start autonomous mode")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down due to keyboard interrupt")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        cleanup_handler()