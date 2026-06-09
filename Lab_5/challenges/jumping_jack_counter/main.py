import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps, draw_counter, draw_status
from detection_logic import detect_jumping_jack

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
    # cap = setup_camera(raspberry_pi_url="http://cvpiXX.local:5000/video")     # uncomment if you're using the Raspberry Pi as your Camera    
    fps_calculator = FPSCalculator()
    
    jumping_jack_count = 0
    
    print("Jumping Jack Counter Challenge")
    print("Press 'q' to quit")
    print("Press 'r' to reset counter")
    print("Implement detect_jumping_jack() in detection_logic.py")
    print("Try doing jumping jacks to test your implementation")
    
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
        
        jumping_jack_detected = False
        keypoints_used = []
        
        # Detect jumping jack if pose is found
        if results.pose_landmarks:
            # Call student's detection function
            detection_result = detect_jumping_jack(results.pose_landmarks.landmark)
            
            # Handle both old format (bool) and new format (tuple)
            if isinstance(detection_result, tuple):
                jumping_jack_detected, keypoints_used = detection_result
            else:
                jumping_jack_detected = detection_result
                keypoints_used = []
            
            # Draw pose landmarks with highlighting
            h, w, _ = image.shape
            
            # Draw pose connections (white lines) first
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=None,  # We'll draw landmarks manually
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=1)
            )
            
            # Draw all pose landmarks as light gray circles
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Light gray for unused keypoints
                cv2.circle(image, (x, y), 4, (128, 128, 128), -1)
            
            # Highlight the keypoints being used by the student's function
            for keypoint_idx in keypoints_used:
                if 0 <= keypoint_idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[keypoint_idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Draw highlighted keypoint (red)
                    cv2.circle(image, (x, y), 6, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(image, (x, y), 8, (255, 255, 255), 2)  # White border
                    # Draw keypoint number
                    cv2.putText(image, str(keypoint_idx), (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Increment counter if jumping jack detected
            if jumping_jack_detected:
                jumping_jack_count += 1
        
        # Display counter and status
        draw_counter(image, jumping_jack_count, "Jumping Jacks")
        draw_status(image, jumping_jack_detected)
        
        # Instructions
        cv2.putText(image, "Press 'r' to reset", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "Arms up + legs apart", (10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display keypoints being used
        if keypoints_used:
            keypoints_str = f"Using keypoints: {keypoints_used}"
            cv2.putText(image, keypoints_str, (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Jumping Jack Counter Challenge', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            jumping_jack_count = 0
            print("Counter reset!")
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()