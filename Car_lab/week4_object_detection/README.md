# Car Lab Week 4: Custom Object Detection

This is the deployment exercise that follows course Lab 6.

Lab 6 creates `best.onnx`. Week 4 loads that model, filters its output and
uses bounding-box size as a simple stopping proxy. Monocular depth estimation
is deliberately postponed to `week5_depth_estimation`.

## Learning objectives

- inspect the output of an exported detection model;
- filter candidates using a confidence threshold;
- map numeric class IDs back to class names;
- calculate the fraction of the frame occupied by a bounding box;
- turn a perception result into a simple stop decision.

## Copy the model from Lab 6

From the laptop:

```bash
scp best.onnx \
  cvpcar#@cvpcar#.local:~/AppCV_2026/Car_lab/week4_object_detection/models/best.onnx
```

## Runtime and student files

```text
sign_detector.py        # working instructor/runtime implementation
sign_detector_empty.py  # student TODO version
sign_detector_base.py   # provided model and geometry helpers
```

For the student release, use `sign_detector_empty.py` as the editable exercise.
Its three TODO sections ask students to:

1. filter raw predictions by class confidence;
2. create detection dictionaries and apply non-maximum suppression;
3. return a stop decision for a sufficiently large target sign.

Model loading, letterbox preprocessing, coordinate conversion and ONNX Runtime
setup are provided in `sign_detector_base.py`.

## Static test first

Run this test with the motors inactive:

```bash
cd ~/AppCV_2026/Car_lab
conda activate app_cv

python -m week4_object_detection.test_detector \
  --model week4_object_detection/models/best.onnx \
  --image week4_object_detection/test_images/custom_sign.jpg
```

The image path above is an example; use one of your own validation images if a
course test image has not been supplied.

## Integrated car test

The repository is already primed for Week 4 in `Car_lab/feature_config.py`:

```python
FEATURES_ENABLED = {
    "line_following": False,
    "speed_estimation": False,
    "sign_detection": True,
}
```

Start the shared server:

```bash
cd ~/AppCV_2026/Car_lab/core
conda activate app_cv
python run.py
```

Open the Object Detection tab and test all signs with autonomous movement off.
The overlay displays class, confidence and bounding-box area percentage.

Only after the static and live-camera tests work should the car be placed on
the floor for a low-speed stopping test.
