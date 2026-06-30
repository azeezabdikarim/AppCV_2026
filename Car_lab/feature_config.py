#!/usr/bin/env python3

"""
Feature control configuration for the shifted 2026 PiCar-X block.
Week 1 uses the separate first-drive gesture app.
"""

FEATURES_ENABLED = {
    "line_following": False,    # Week 2
    "speed_estimation": True,   # Week 3
    "sign_detection": False,    # Week 4
}

FEATURE_DESCRIPTIONS = {
    "line_following": "Computer vision + PID control for following lines",
    "speed_estimation": "Optical flow analysis for estimating robot speed",
    "sign_detection": "ONNX model integration for detecting stop signs",
}

REQUIRED_METHODS = {
    "line_following": ["compute_steering_angle"],
    "speed_estimation": ["estimate_speed"],
    "sign_detection": ["detect_signs", "should_stop"],
}


def is_feature_enabled(feature_name):
    return FEATURES_ENABLED.get(feature_name, False)


def get_enabled_features():
    return [name for name, enabled in FEATURES_ENABLED.items() if enabled]


def get_feature_description(feature_name):
    return FEATURE_DESCRIPTIONS.get(feature_name, "No description available")
