import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup camera
    cap = setup_camera()
    fps_calculator = FPSCalculator()
    
    print("Full Body Pose Detection Started")
    print("Press 'q' to quit")
    print("Pose landmarks will be printed to console...")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip image horizontally for selfie-view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = pose.process(image_rgb)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Print some landmark coordinates
            landmarks = results.pose_landmarks.landmark
            print(f"Left wrist (landmark 15): x={landmarks[15].x:.3f}, y={landmarks[15].y:.3f}")
            print(f"Right wrist (landmark 16): x={landmarks[16].x:.3f}, y={landmarks[16].y:.3f}")
            print(f"Left ankle (landmark 27): x={landmarks[27].x:.3f}, y={landmarks[27].y:.3f}")
            print(f"Right ankle (landmark 28): x={landmarks[28].x:.3f}, y={landmarks[28].y:.3f}")
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Full Body Pose Detection', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()