def detect_jumping_jack(pose_landmarks):
    """
    Detect if person is in jumping jack position.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks (33 points)
                       Each landmark has .x, .y, .z coordinates and .visibility
    
    Returns:
        tuple: (is_jumping_jack, keypoints_list)
            - is_jumping_jack: True if jumping jack position detected, False otherwise
            - keypoints_list: List of landmark indices you're using for analysis
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the body with
    red circles. All other pose landmarks will show as light gray circles.
    This helps you visualize which points you're analyzing and verify they're
    in the right locations for jumping jack detection.
    
    Hint: Check both arm and leg positions
    Key landmarks you might find useful:
    - Left wrist: landmark 15
    - Right wrist: landmark 16  
    - Left shoulder: landmark 11
    - Right shoulder: landmark 12
    - Left ankle: landmark 27
    - Right ankle: landmark 28
    - Left hip: landmark 23
    - Right hip: landmark 24
    - Nose: landmark 0 (for head reference)
    
    Jumping jack position requirements:
    1. Both arms should be raised above shoulder level
    2. Legs should be spread apart (wider than hip width)
    
    Example usage:
    keypoints_used = [15, 16, 11, 12, 27, 28, 23, 24]  # Arms and legs
    return (is_jumping_jack, keypoints_used)
    """
    
    # TODO: Implement jumping jack detection logic here
    # Currently returns False - you need to analyze pose landmarks
    # to determine if both arms are up AND legs are spread apart
    
    # Example of keypoints you might want to examine:
    # arms = [15, 16, 11, 12]  # Wrists and shoulders
    # legs = [27, 28, 23, 24]  # Ankles and hips
    
    # Return both the detection result and the keypoints you're using
    # This will help you debug by seeing exactly which points you're analyzing
    
    # Note: Like the clapping counter, you might notice this gets 
    # called many times per second. What happens when you hold 
    # a jumping jack pose?
    
    return (False, [])