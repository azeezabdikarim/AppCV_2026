import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps

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
    fps_calculator = FPSCalculator()
    
    print("Hand Pose Detection Started")
    print("Press 'q' to quit")
    print("Hand landmarks will be printed to console...")
    
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
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Print some landmark coordinates
                print(f"Hand {idx + 1}:")
                print(f"  Thumb tip (landmark 4): x={hand_landmarks.landmark[4].x:.3f}, y={hand_landmarks.landmark[4].y:.3f}")
                print(f"  Index tip (landmark 8): x={hand_landmarks.landmark[8].x:.3f}, y={hand_landmarks.landmark[8].y:.3f}")
                print(f"  Wrist (landmark 0): x={hand_landmarks.landmark[0].x:.3f}, y={hand_landmarks.landmark[0].y:.3f}")
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Hand Pose Detection', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()