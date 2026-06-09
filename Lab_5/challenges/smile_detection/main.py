import cv2
import mediapipe as mp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.camera_utils import setup_camera, cleanup_camera, FPSCalculator
from utils.display_utils import draw_fps, draw_text
from detection_logic import detect_emotion

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
    # cap = setup_camera(raspberry_pi_url="http://cvpiXX.local:5000/video")     # uncomment if you're using the Raspberry Pi as your Camera

    fps_calculator = FPSCalculator()
    
    print("Smile Detection Challenge (Simple)")
    print("Press 'q' to quit")
    print("Implement detect_emotion() in detection_logic.py")
    print("This version shows fewer keypoints and highlights the ones you use")
    
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
        
        emotion = "neutral"  # Default
        keypoints_used = []  # Default empty list
        
        # Detect emotion if face is found
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw light face mesh outline
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 100, 100), thickness=1, circle_radius=1)
                )
                
                # Call student's detection function
                detection_result = detect_emotion(face_landmarks.landmark)
                
                # Handle both old format (string) and new format (tuple)
                if isinstance(detection_result, tuple):
                    emotion, keypoints_used = detection_result
                else:
                    emotion = detection_result
                    keypoints_used = []
                
                # Highlight the keypoints being used by the student's function
                h, w, _ = image.shape
                for keypoint_idx in keypoints_used:
                    if 0 <= keypoint_idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[keypoint_idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        # Draw highlighted keypoint
                        cv2.circle(image, (x, y), 6, (0, 0, 255), -1)  # Red filled circle
                        cv2.circle(image, (x, y), 8, (255, 255, 255), 2)  # White border
                        # Draw keypoint number
                        cv2.putText(image, str(keypoint_idx), (x + 10, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display emotion with color coding
        if emotion == "smile":
            color = (0, 255, 0)  # Green
            bg_color = (0, 100, 0)
        elif emotion == "frown":
            color = (0, 0, 255)  # Red  
            bg_color = (100, 0, 0)
        else:
            color = (255, 255, 255)  # White
            bg_color = (50, 50, 50)
        
        # Draw colored background rectangle
        cv2.rectangle(image, (10, 80), (300, 130), bg_color, -1)
        
        # Display emotion text
        draw_text(image, f"Emotion: {emotion.upper()}", (15, 110), color, 0.8)
        
        # Display keypoints being used
        if keypoints_used:
            keypoints_str = f"Using keypoints: {keypoints_used}"
            cv2.putText(image, keypoints_str, (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Calculate and display FPS
        fps = fps_calculator.update()
        draw_fps(image, fps)
        
        # Show the image
        cv2.imshow('Smile Detection Challenge (Simple)', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cleanup_camera(cap)

if __name__ == "__main__":
    main()