import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps, draw_text
from detection_logic import count_fingers

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup camera
    cap = setup_camera()
    # cap = setup_camera(raspberry_pi_url="http://cvpiXX.local:5000/video")     # uncomment if you're using the Raspberry Pi as your Camera

    
    fps_calculator = FPSCalculator()
    
    print("Number Recognition Challenge")
    print("Press 'q' to quit")
    print("Implement count_fingers() in detection_logic.py")
    print("Hold up 0-5 fingers per hand to test your implementation")
    print("Shows separate counts for left hand, right hand, and total")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip image horizontally for selfie-view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(image_rgb)
        
        # Default values
        counts = {"left": 0, "right": 0, "total": 0}
        keypoints_used = []
        
        # Count fingers if hands are detected
        if results.multi_hand_landmarks and results.multi_handedness:
            # Prepare hand data with classification
            hand_data_list = []
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification (Left or Right)
                hand_classification = results.multi_handedness[i].classification[0].label
                hand_data_list.append((hand_landmarks.landmark, hand_classification))
            
            # Call student's detection function
            detection_result = count_fingers(hand_data_list)
            
            # Handle both old format (int) and new format (tuple)
            if isinstance(detection_result, tuple):
                counts, keypoints_used = detection_result
                # Ensure counts is a dictionary
                if not isinstance(counts, dict):
                    counts = {"left": 0, "right": 0, "total": counts}
            else:
                counts = {"left": 0, "right": 0, "total": detection_result}
                keypoints_used = []
            
            # Draw hand landmarks with highlighting
            h, w, _ = image.shape
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_classification = results.multi_handedness[i].classification[0].label
                
                # Draw hand connections (white lines) first
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=None,  # We'll draw landmarks manually
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=1)
                )
                
                # Draw all hand landmarks as light gray circles
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    # Light gray for unused keypoints
                    cv2.circle(image, (x, y), 4, (128, 128, 128), -1)
                
                # Highlight the keypoints being used by the student's function
                for keypoint_idx in keypoints_used:
                    if 0 <= keypoint_idx < len(hand_landmarks.landmark):
                        landmark = hand_landmarks.landmark[keypoint_idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        # Draw highlighted keypoint (red)
                        cv2.circle(image, (x, y), 6, (0, 0, 255), -1)  # Red filled circle
                        cv2.circle(image, (x, y), 8, (255, 255, 255), 2)  # White border
                        # Draw keypoint number
                        cv2.putText(image, str(keypoint_idx), (x + 10, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Label each hand with L/R
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                hand_label = "L" if hand_classification == "Left" else "R"
                cv2.putText(image, hand_label, (wrist_x - 20, wrist_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw count displays (top-left, smaller)
        # Left hand count
        cv2.rectangle(image, (10, 50), (80, 80), (50, 50, 50), -1)
        draw_text(image, f"L: {counts['left']}", (15, 70), (0, 255, 255), 0.8, 2)
        
        # Right hand count  
        cv2.rectangle(image, (90, 50), (160, 80), (50, 50, 50), -1)
        draw_text(image, f"R: {counts['right']}", (95, 70), (0, 255, 255), 0.8, 2)
        
        # Total count
        cv2.rectangle(image, (170, 50), (260, 80), (70, 70, 70), -1)
        draw_text(image, f"Total: {counts['total']}", (175, 70), (0, 255, 0), 0.8, 2)
        
        # Display instructions (moved down)
        draw_text(image, "Show fingers: L=Left, R=Right", (10, 110), (255, 255, 255), 0.5)
        
        # Display keypoints being used (moved down)
        if keypoints_used:
            keypoints_str = f"Using keypoints: {keypoints_used}"
            cv2.putText(image, keypoints_str, (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Number Recognition Challenge', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()