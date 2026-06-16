def detect_steering_gesture(hand_landmarks):
    """
    Decide whether the hand gesture means left, right, straight, or unknown.

    Args:
        hand_landmarks: MediaPipe landmarks for one detected hand.
                        Each landmark has .x, .y, and .z coordinates.

    Returns:
        tuple: (direction, keypoints_used)
            direction: "left", "right", "straight", or "unknown"
            keypoints_used: list of landmark indices used by your logic

    Suggested first approach:
        Use the vector from the index-finger base (5) to the index-finger tip
        (8). The x coordinate increases from left to right in the camera image.

        dx = hand_landmarks[8].x - hand_landmarks[5].x

        If left and right are reversed on your car, swap the returned labels.
    """

    # TODO: Replace this placeholder with your steering decision logic.
    # Return the keypoints you use so the live debug feed can highlight them.
    return ("unknown", [5, 8])
