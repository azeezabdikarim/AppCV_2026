# Lab 6: Fine-Tuning YOLOv8n for Custom Signs

Lab 6 produces the model used later in `Car_lab/week4_object_detection`.

The workflow is:

1. capture training and validation images with a PiCar-X camera;
2. annotate the images in CVAT;
3. export an **Ultralytics YOLO Detection 1.0** dataset;
4. fine-tune `yolov8n.pt` in Google Colab;
5. export and test `best.onnx`;
6. copy `best.onnx` into the Week 4 car exercise.

Training runs in Colab. The PiCar-X is used for image capture and, later, ONNX
inference. It does not train the model.

## Capture images

The course repository and `app_cv` conda environment should already exist on
the car.

```bash
ssh cvpcar#@cvpcar#.local
cd ~/AppCV_2026
git pull
conda activate app_cv
cd ~/AppCV_2026/Lab_6
```

Collect the training and validation sessions separately:

```bash
python scripts/capture_training_images.py \
  --session train --num-images 60 --delay 0.4

python scripts/capture_training_images.py \
  --session val --num-images 20 --delay 0.6
```

The images are saved under:

```text
captured_images/
├── train/
│   ├── train_000.jpg
│   └── ...
└── val/
    ├── val_000.jpg
    └── ...
```

Change the background, sign pose or lighting before collecting validation
images. Consecutive frames from one burst should not be divided between train
and validation.

Copy the images to the laptop:

```bash
scp -r cvpcar#@cvpcar#.local:~/AppCV_2026/Lab_6/captured_images .
```

## Optional standalone detector

`detection/app.py` is the 2026 version of the standalone browser detector from
the 2025 lab. It is useful for checking an exported ONNX model independently of
the car controller.

Copy a model into `models/`, then run:

```bash
conda activate app_cv
cd ~/AppCV_2026/Lab_6/detection
python app.py --model ../models/best.onnx --port 8000
```

Open `http://<car-address>:8000` in a laptop browser. This app only displays
detections; it never controls the motors.

## Handoff to Car Lab Week 4

From the laptop:

```bash
scp best.onnx \
  cvpcar#@cvpcar#.local:~/AppCV_2026/Car_lab/week4_object_detection/models/best.onnx
```

Then follow `Car_lab/week4_object_detection/README.md`.

