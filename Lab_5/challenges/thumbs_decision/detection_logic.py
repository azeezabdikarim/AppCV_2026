def detect_thumbs_decision(hand_landmarks_list):
    """
    Detect thumbs up, thumbs down, or no decision gesture.
    
    Args:
        hand_landmarks_list: List of hand landmarks for all detected hands
                            Each hand has 21 landmarks with .x, .y, .z coordinates
    
    Returns:
        tuple: (decision_string, keypoints_list)
            - decision_string: "thumbs_up", "thumbs_down", or "no_decision"
            - keypoints_list: List of landmark indices you're using for analysis
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the hands with
    red circles. All other hand landmarks will show as light gray circles.
    This helps you visualize which points you're analyzing and verify they're
    in the right locations for thumbs detection.
    
    Hint: Analyze thumb position relative to other fingers and hand orientation
    Key landmarks you might find useful:
    - Thumb tip: landmark 4
    - Thumb joints: landmarks 2, 3
    - Index finger tip: landmark 8
    - Middle finger tip: landmark 12
    - Wrist: landmark 0
    - Other fingertips: 16 (ring), 20 (pinky)
    
    Detection logic ideas:
    - Thumbs up: thumb extended upward, other fingers curled
    - Thumbs down: thumb extended downward, other fingers curled  
    - No decision: thumb not clearly extended or multiple fingers up
    
    Example usage:
    keypoints_used = [4, 8, 12, 16, 20]  # Thumb + other fingertips
    return (decision, keypoints_used)
    """
    
    # TODO: Implement thumbs decision detection logic here
    # Currently returns no_decision - you need to analyze hand landmarks
    # to determine thumb position and other finger states
    
    # Example of keypoints you might want to examine:
    # thumb_landmarks = [4, 3, 2]  # Thumb tip and joints
    # finger_tips = [8, 12, 16, 20]  # Other fingertips
    # reference_points = [0]  # Wrist for orientation
    
    # Return both the decision and the keypoints you're using
    # This will help you debug by seeing exactly which points you're analyzing
    
    return ("no_decision", [])