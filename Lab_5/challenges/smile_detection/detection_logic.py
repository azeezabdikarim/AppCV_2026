def detect_emotion(face_landmarks):
    """
    Detect if the person is smiling, frowning, or neutral.
    
    Args:
        face_landmarks: MediaPipe face landmarks (468 points)
                       Each landmark has .x, .y, .z coordinates (normalized 0-1)
    
    Returns:
        tuple: (emotion_string, keypoints_list)
            - emotion_string: "smile", "frown", or "neutral"  
            - keypoints_list: List of landmark indices you're using for detection
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the face with
    red circles and their numbers displayed. This helps you visualize which
    points you're analyzing and verify they're in the right locations.
    
    Key facial landmarks for emotion detection:
    - Left mouth corner: 61
    - Right mouth corner: 291
    - Upper lip center: 13  
    - Lower lip center: 14
    - Left lip corner: 308
    - Right lip corner: 78
    - Nose tip: 1
    - Chin: 175
    
    Tip: Start by returning a few key mouth landmarks in the keypoints list
    to see where they are positioned on your face, then build your logic
    around their relative positions.
    
    Example return format:
    return ("smile", [61, 291, 13, 14])  # emotion + keypoints being analyzed
    """
    
    # TODO: Implement emotion detection logic here
    # Currently returns neutral with no keypoints highlighted
    
    # Example of keypoints you might want to examine:
    # mouth_corners = [61, 291]  # Left and right mouth corners
    # mouth_center = [13, 14]    # Upper and lower lip center
    
    # Return both the emotion and the keypoints you're using
    # This will help you debug by seeing exactly which points you're analyzing
    
    return ('neutral', [])