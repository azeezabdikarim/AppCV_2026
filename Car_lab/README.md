# AppCV 2026 Car Lab

This directory contains the 2026 PiCar-X lab code.

## Week layout

- `week1_first_drive_gesture`: first-drive web UI and MediaPipe gesture steering.
- `week2_line_following`: shifted copy of the 2025 line-following lab.
- `week3_object_detection`: shifted copy of the 2025 object-detection/sign-policy lab.
- `week4_speed_estimation`: shifted copy of the 2025 optical-flow speed-estimation lab.
- `core`: full multi-week web UI used after the Week 1 introduction.

## Week 1 network rule

- Pi 5 cars join `CV-PI-NET`.
- Laptop viewers join `CV-CAR-VIEW-5G`.
- Use one live video UI per car during normal work.

## Run Week 1

```bash
cd ~
git clone https://github.com/azeezabdikarim/AppCV_2026.git
cd ~/AppCV_2026/Car_lab/week1_first_drive_gesture
python3 app.py
```

Students should edit only:

```text
week1_first_drive_gesture/gesture_logic.py
```
