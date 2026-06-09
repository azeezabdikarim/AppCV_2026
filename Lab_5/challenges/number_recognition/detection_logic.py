def count_fingers(hand_data_list):
    """
    Count the number of fingers being held up on each hand.
    
    Args:
        hand_data_list: List of tuples, each containing:
                       (hand_landmarks, hand_classification)
                       - hand_landmarks: 21 landmarks with .x, .y, .z coordinates
                       - hand_classification: MediaPipe classification ("Left" or "Right")
    
    Returns:
        tuple: (counts_dict, keypoints_list)
            - counts_dict: Dictionary with keys "left", "right", "total"
            - keypoints_list: List of landmark indices you're using for analysis
    
    DEBUGGING HELPER:
    The keypoints you return in the list will be highlighted on the hands with
    red circles. All other hand landmarks will show as light gray circles.
    This helps you visualize which points you're analyzing and verify they're
    in the right locations for finger detection.
    
    Hand Classification:
    MediaPipe provides hand classification information that tells you if each
    detected hand is "Left" or "Right". This is accessed through:
    results.multi_handedness[i].classification[0].label
    
    Hint: Compare fingertip positions to their respective joints
    Key landmarks for each finger:
    - Thumb: tip=4, joint=3
    - Index: tip=8, joint=6  
    - Middle: tip=12, joint=10
    - Ring: tip=16, joint=14
    - Pinky: tip=20, joint=18
    
    Example usage:
    keypoints_used = [4, 8, 12, 16, 20]  # All fingertips
    counts = {"left": 0, "right": 0, "total": 0}
    return (counts, keypoints_used)
    """
    
    # TODO: Implement finger counting logic here
    # Currently returns 0 for all counts with no keypoints highlighted
    
    # Example of keypoints you might want to examine:
    # fingertips = [4, 8, 12, 16, 20]  # All fingertip landmarks
    # joints = [3, 6, 10, 14, 18]      # Corresponding joint landmarks
    
    # Return both the counts and the keypoints you're using
    # This will help you debug by seeing exactly which points you're analyzing
    
    counts = {"left": 0, "right": 0, "total": 0}
    keypoints_used = []
    
    return (counts, keypoints_used)