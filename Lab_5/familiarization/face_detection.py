import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup camera
    cap = setup_camera()
    fps_calculator = FPSCalculator()
    
    print("Face Detection Started")
    print("Press 'q' to quit")
    print("Face landmarks will be printed to console...")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip image horizontally for selfie-view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_mesh.process(image_rgb)
        
        # Draw face landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                # Print some landmark coordinates
                print(f"Nose tip (landmark 1): x={face_landmarks.landmark[1].x:.3f}, y={face_landmarks.landmark[1].y:.3f}")
                print(f"Left eye corner (landmark 33): x={face_landmarks.landmark[33].x:.3f}, y={face_landmarks.landmark[33].y:.3f}")
                print(f"Right eye corner (landmark 362): x={face_landmarks.landmark[362].x:.3f}, y={face_landmarks.landmark[362].y:.3f}")
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Face Detection', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()