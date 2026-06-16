#!/usr/bin/env python3

"""
Feature control configuration for the shifted 2026 PiCar-X block.
Week 1 uses the separate first-drive gesture app.
"""

FEATURES_ENABLED = {
    "line_following": True,     # Week 2
    "sign_detection": False,    # Week 3
    "speed_estimation": False,  # Week 4
}

FEATURE_DESCRIPTIONS = {
    "line_following": "Computer vision + PID control for following lines",
    "sign_detection": "ONNX model integration for detecting stop signs",
    "speed_estimation": "Optical flow analysis for estimating robot speed",
}

REQUIRED_METHODS = {
    "line_following": ["compute_steering_angle"],
    "sign_detection": ["detect_signs", "should_stop"],
    "speed_estimation": ["estimate_speed"],
}


def is_feature_enabled(feature_name):
    return FEATURES_ENABLED.get(feature_name, False)


def get_enabled_features():
    return [name for name, enabled in FEATURES_ENABLED.items() if enabled]


def get_feature_description(feature_name):
    return FEATURE_DESCRIPTIONS.get(feature_name, "No description available")
