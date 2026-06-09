# Pose Estimation Challenges

Welcome to the challenge phase! You'll implement different pose and gesture recognition systems. Each challenge is in its own directory with a skeleton function for you to complete.

## Using Raspberry Pi Camera (Optional)

If your laptop doesn't have a webcam or you want to use a Raspberry Pi camera, you can stream video from a Raspberry Pi to your laptop for processing.

### How It Works
The Raspberry Pi runs a video server that captures camera frames and streams them over the network. Your laptop receives these frames and processes them with MediaPipe - all your detection code remains exactly the same.

### Setup Instructions

**On the Raspberry Pi:**
1. Install required packages:
   ```bash
   pip install flask opencv-python
   ```

2. Run the video server:
   ```bash
   python3 raspberry_pi/video_server.py
   ```

3. Test the connection by opening `http://cvpiXX.local:5000` in your browser (replace XX with your RPi number). You should see a live video stream.

**On your laptop:**
1. Ensure your laptop and Raspberry Pi are on the same WiFi network

2. In any challenge's `main.py` file, modify the camera setup:
   ```python
   # Default (uses laptop webcam):
   cap = setup_camera()
   
   # To use Raspberry Pi camera, uncomment and update:
   # cap = setup_camera(raspberry_pi_url="http://cvpiXX.local:5000/video")
   ```

3. Replace `XX` with your assigned Raspberry Pi number (e.g., `cvpi33.local`)

### Important Notes
- **Automatic fallback**: If the Raspberry Pi connection fails, the code automatically falls back to your laptop webcam
- **Performance**: Expect slightly lower FPS and video quality due to network streaming
- **Network requirement**: Both devices must be connected to the same WiFi network

---

## Challenge Overview

### ⭐ Smile Detection
**Directory:** `smile_detection/`
**Goal:** Detect smiles and frowns in real-time

Your task is to implement the `detect_emotion()` function in `detection_logic.py`. The function should:
- Return a tuple: `(emotion_string, keypoints_list)`
- emotion_string: `"smile"`, `"frown"`, or `"neutral"`
- keypoints_list: List of landmark indices you're using for analysis

The GUI will highlight the keypoints you specify, helping you debug and understand which facial points you're analyzing.

---

### ⭐⭐ Number Recognition
**Directory:** `number_recognition/`
**Goal:** Count the number of fingers being held up

Your task is to implement the `count_fingers()` function in `detection_logic.py`. The function should:
- Return a tuple: `(finger_count, keypoints_list)`
- finger_count: Number of fingers extended (0-5, or 0-10 if counting both hands)
- keypoints_list: List of landmark indices you're using for analysis

The GUI will highlight your selected keypoints in red while showing all other hand landmarks in gray, helping you visualize your detection logic.

---

### ⭐⭐ Colosseum Decision
**Directory:** `thumbs_decision/`
**Goal:** Detect thumbs up, thumbs down, or no clear decision

Your task is to implement the `detect_thumbs_decision()` function in `detection_logic.py`. The function should:
- Return a tuple: `(decision_string, keypoints_list)`
- decision_string: `"thumbs_up"`, `"thumbs_down"`, or `"no_decision"`
- keypoints_list: List of landmark indices you're using for analysis

The GUI displays gladiator-style feedback with color coding for each decision type.

---

### ⭐⭐⭐ Clapping Counter
**Directory:** `clapping_counter/`
**Goal:** Count individual claps

Your task is to implement the `detect_clap()` function in `detection_logic.py`. The function should:
- Return a tuple: `(is_clapping, keypoints_list)`
- is_clapping: `True` when a clap is detected, `False` otherwise
- keypoints_list: List of landmark indices you're using for analysis
- Each clap should only be counted once

The GUI will display a running count of claps detected and highlight the keypoints you're analyzing.

---

### ⭐⭐⭐ Jumping Jack Counter
**Directory:** `jumping_jack_counter/`
**Goal:** Count jumping jack repetitions

Your task is to implement the `detect_jumping_jack()` function in `detection_logic.py`. The function should:
- Return a tuple: `(is_jumping_jack, keypoints_list)`
- is_jumping_jack: `True` when a jumping jack position is detected, `False` otherwise
- keypoints_list: List of landmark indices you're using for analysis
- Require both arms above head AND legs spread apart

The GUI will display a running count of jumping jacks detected and show which body keypoints you're analyzing.

## How to Work on Challenges

### Running a Challenge
1. Navigate to the challenge directory:
   ```bash
   cd smile_detection  # or whichever challenge you choose
   ```

2. Run the main application:
   ```bash
   python main.py
   ```

3. The GUI will open showing:
   - Your camera feed with keypoints
   - FPS counter
   - Challenge-specific feedback/counters
   - Current detection status
   - Highlighted keypoints you're analyzing (red circles)

### Implementing Your Solution
1. Open `detection_logic.py` in your chosen challenge
2. Find the function with the `#TODO` comment
3. Implement your detection logic using the provided keypoints
4. Return both your result and the keypoints you're using for analysis
5. Test your implementation by running `main.py`
6. Use the visual feedback to debug and improve your solution

### Understanding the Data
The detection functions receive keypoint data in this format:
- **Face keypoints**: List of 468 facial landmarks with x, y coordinates
- **Hand keypoints**: List of 21 hand landmarks per hand with x, y, z coordinates  
- **Pose keypoints**: List of 33 body landmarks with x, y, z coordinates and visibility

### Debugging Visualization
All challenges include visual debugging tools:
- **Gray circles**: Show all available keypoints
- **Red circles with white borders**: Highlight keypoints you're analyzing

## Tips for Success

### General Strategy
- Start with the easier challenges to understand the keypoint data
- Use the debugging visualization to see exactly which points you're analyzing
- Print keypoint values to understand coordinate systems
- Test with exaggerated motions first
- Consider edge cases (hands off-screen, poor lighting, etc.)

### Debugging
- Return keypoints in your detection functions to see them highlighted
- Use `print()` statements to understand keypoint values
- Check the console output for keypoint coordinates
- Test with different hand positions and facial expressions
- Ensure good lighting and clear camera view

### Common Challenges You May Encounter
- **High frame rates**: You might notice rapid repeated detections
- **Coordinate systems**: Understanding how MediaPipe coordinates work
- **Missing detections**: Handling cases where hands/face aren't detected
- **Sensitivity**: Balancing detection accuracy with false positives

Remember: The goal is learning, not completion. Focus on understanding the unique attributes of each challenge and the methods one must use to implement a robust solutions.