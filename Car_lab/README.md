# AppCV 2026 Car Lab

This directory contains the 2026 PiCar-X lab code.

## Week layout

- `week1_first_drive_gesture`: first-drive web UI and MediaPipe gesture steering.
- `week2_line_following`: course Lab 9, line following with classical CV and PID.
- `week3_speed_estimation`: course Lab 10, optical-flow speed estimation.
- `week4_object_detection`: deploy the custom ONNX detector produced in Lab 6 and trigger stopping from bounding-box area.
- `week5_depth_estimation`: downstream monocular-depth work, separated from the Week 4 area proxy.
- `core`: full multi-week web UI used after the Week 1 introduction.

## Core stream/control architecture

The multi-week `core` app uses one background vision/control worker for Weeks
2-4. That worker owns the loop:

```text
camera frame -> active feature processing -> debug overlay -> cached JPEG
```

## Week 1 network rule

- Pi 5 cars join `CV-PI-NET`.
- Laptop viewers join `CV-CAR-VIEW-5G`.
- Use one live video UI per car during normal work.

## Python environment setup

We use Miniforge, a lightweight conda installer, to create the same Python
environment on every car. Conda is a package and environment manager: it keeps
the course Python version and packages separate from the Raspberry Pi system
Python, which helps avoid version conflicts with packages such as MediaPipe.

Install Miniforge:

```bash
cd ~

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p "$HOME/miniforge3"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
```

Create and activate the course environment from the file in this directory:

```bash
cd ~/AppCV_2026/Car_lab

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env create -f app_cv_environment.yml
conda activate app_cv
```

Install the local PiCar-X hardware packages into the environment:

```bash
sudo chown -R "$USER:$USER" ~/robot-hat ~/picar-x

python -m pip install -e ~/robot-hat
python -m pip install -e ~/picar-x
```

The `chown` command makes sure the current user owns the local SunFounder
package folders, so `pip` can generate its package metadata during installation.

## Run Week 1

```bash
cd ~
git clone https://github.com/azeezabdikarim/AppCV_2026.git
cd ~/AppCV_2026/Car_lab/week1_first_drive_gesture
conda activate app_cv
python app.py
```

You should edit only:

```text
week1_first_drive_gesture/gesture_logic.py
```

## Run Week 2 and later

```bash
cd ~/AppCV_2026/Car_lab/core
conda activate app_cv
python run.py
```

## Lab 6 and Week 4 model handoff

Course Lab 6 captures and annotates sign images, fine-tunes YOLOv8n in Colab,
and exports `best.onnx`. The resulting model is copied to:

```text
Car_lab/week4_object_detection/models/best.onnx
```

Week 4 deliberately uses only detection confidence and bounding-box area.
Monocular depth estimation begins in Week 5.
