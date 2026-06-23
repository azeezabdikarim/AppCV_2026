# AppCV 2026 Car Lab

This directory contains the 2026 PiCar-X lab code.

## Week layout

- `week1_first_drive_gesture`: first-drive web UI and MediaPipe gesture steering.
- `week2_line_following`: shifted copy of the 2025 line-following lab.
- `week3_object_detection`: shifted copy of the 2025 object-detection/sign-policy lab.
- `week4_speed_estimation`: shifted copy of the 2025 optical-flow speed-estimation lab.
- `core`: full multi-week web UI used after the Week 1 introduction.

## Core stream/control architecture

The multi-week `core` app uses one background vision/control worker for Weeks
2-4. That worker owns the loop:

```text
camera frame -> active feature processing -> debug overlay -> cached JPEG
```

The Flask `/video_feed` route serves the cached JPEG. This means a second
browser viewer still adds network traffic, but it does not run a second
line-following/object-detection/speed-estimation pass and does not JPEG-encode
the same frame twice. New car-lab features should plug into
`core/robot_controller.py::process_autonomous_frame()` rather than adding
separate per-client video processing routes.

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

Students should edit only:

```text
week1_first_drive_gesture/gesture_logic.py
```

## Run Week 2 and later

```bash
cd ~/AppCV_2026/Car_lab/core
conda activate app_cv
python run.py
```
