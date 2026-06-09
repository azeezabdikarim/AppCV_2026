def detect_clap(hand_landmarks_list):
    """
    Detect if a clap motion is occurring.
    
    Args:
        hand_landmarks_list: List containing landmarks for both hands
                            Each hand has 21 landmarks with .x, .y, .z coordinates
    
    Returns:
        tuple: (is_clapping, keypoints_list)
            - is_clapping: True if clap is detected, False otherwise
            - keypoints_list: List of landmark indices you're using for analysis
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the hands with
    red circles. All other hand landmarks will show as light gray circles.
    This helps you visualize which points you're analyzing and verify they're
    in the right locations for clap detection.
    
    Hint: Calculate distance between palm centers of both hands
    Key landmarks you might find useful:
    - Palm center approximation: landmark 9 (middle finger base)
    - Or use landmark 0 (wrist) as reference point
    
    Example usage:
    keypoints_used = [0, 9]  # Wrists and palm centers from both hands
    return (is_clapping, keypoints_used)
    """
    
    # TODO: Implement clap detection logic here
    # Currently returns False - you need to analyze hand positions
    # to determine if hands are close enough together to be clapping
    
    # Example of keypoints you might want to examine:
    # palm_centers = [9]  # Middle finger base (approximate palm center)
    # wrists = [0]        # Wrist landmarks
    
    # Return both the detection result and the keypoints you're using
    # This will help you debug by seeing exactly which points you're analyzing
    
    # Note: You might notice this gets called many times per second
    # What happens when you clap once but the function returns True
    # for multiple consecutive frames? What happens when keypoint tracking is lost? 
    
    return (False, [])