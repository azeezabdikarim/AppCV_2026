#!/usr/bin/env python3

"""
Simple script to run the PiCar-X control application
"""

import sys
import subprocess
import time
import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address (doesn't actually send data)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        try:
            # Fallback method
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            if local_ip.startswith("127."):
                # If we get localhost, try getting all addresses
                import subprocess
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
                if result.returncode == 0:
                    addresses = result.stdout.strip().split()
                    if addresses:
                        local_ip = addresses[0]
            return local_ip
        except Exception:
            return "localhost"

def check_requirements():
    """Check if required modules are available"""
    required_modules = ['flask', 'cv2', 'picarx']  # Removed vilib
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} - OK")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {', '.join(missing_modules)}")
        print("Please install missing modules before running.")
        return False
    
    return True

def main():
    print("=== PiCar-X Control Application ===\n")
    
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print("\nStarting application...")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Run the Flask application
        from flask_control import app, camera
        
        # Start camera
        print("📷 Initializing camera (may take a few seconds)...")
        camera.start_streaming()
        time.sleep(3)  # Give camera more time to start
        
        # Get the actual IP address
        local_ip = get_local_ip()
        
        print("\n🚀 Server starting...")
        print("📷 Camera streaming with libcamera-vid")
        print("🌐 Web interface available at:")
        print("   - Local: http://localhost:5000")
        print(f"   - Network: http://{local_ip}:5000")
        print("\n🎮 Controls:")
        print("   - Use WASD keys or web buttons")
        print("   - Each movement lasts 0.5 seconds")
        print("   - Red button for emergency stop")
        print("   - Status updates appear in log below video")
        print("\n" + "="*50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()