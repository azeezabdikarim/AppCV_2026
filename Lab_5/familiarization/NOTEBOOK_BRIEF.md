# Build Brief: "Explore the Landmark Data" Notebook

> **For the agent building this:** This is a spec, not the notebook. Read it fully
> before writing any cells. Your output is a single Jupyter notebook (see
> "Deliverable" below). Do not change the existing `.py` familiarization scripts or
> any challenge code — this notebook is an *addition*.

---

## 1. Why this notebook exists (read this first — it drives every decision)

Lab 5 teaches students to build gesture/pose recognizers on top of MediaPipe
(hands, face, body). The single biggest prerequisite — and the one the lab
currently under-serves — is **understanding the exact data structure MediaPipe
returns**. Without that, the challenge docstrings (e.g. "thumb tip is landmark 4",
"mouth corner is landmark 61") are just magic numbers.

Today the only "familiarization" with the data is `print()` statements buried
inside a 30fps live video loop (`familiarization/hand_pose.py` etc.). The output
scrolls past faster than anyone can read it, so students never actually *inspect*
anything. This notebook fixes that by doing the opposite:

- **Freeze the frame.** We run the models on three *static* reference images, not a
  live webcam. The data holds still so students can poke at it across many cells.
- **Inspect, don't skim.** Every inspection cell is followed by a markdown prompt
  that asks the student a concrete question about the output they just printed.
  The learning happens when they *read the output and answer*, not when code runs.
- **No camera plumbing.** Zero webcam code here on purpose. Camera flakiness
  (permissions, wrong index, Continuity Camera) is a distraction from the actual
  goal. Live capture stays in the `.py` scripts.

### Division of labour in Lab 5 (state this in the notebook intro)
- **This notebook** = *understand the data* (static, exploratory, read-and-answer).
- **`familiarization/*.py` scripts** = *see it run live* (real-time webcam).
- **`challenges/*`** = *use the data* to build something.

### What students must walk away able to answer
1. What object does each model return, and what does it hold when nothing is
   detected? (`None` vs empty list — they will hit this in the challenges.)
2. How many landmarks does each model produce? (hands 21, pose 33, face 468/478.)
3. What fields does a single landmark have? (`x`, `y`, `z`, and `visibility` for
   pose only.)
4. What coordinate system are `x`/`y` in? (**Normalized 0–1**, relative to image
   width/height — NOT pixels.) How do you convert to pixels? (`x*w`, `y*h`.)
5. For hands: how does MediaPipe tell you left vs right? (`multi_handedness`.)
6. What *exact shape* does each challenge's `detect_*()` function actually receive?
   (This differs per challenge — see §6. Connecting the notebook to this is the
   whole payoff.)

---

## 2. Inputs you have to work with

Reference images live in `AppCV_2026/Lab_5/ref_imgs/`:
- `hands.jpg` — two open hands on white. Clean, JPEG, 360×240. Good as-is.
- `body.jpg` — full-body standing figure, JPEG, 302×400. **Dark / dramatic
  lighting.** Run pose on it early; if detection is poor or partial, flag it and
  swap for a brighter, clearer full-body photo. Don't sink the notebook on a bad
  input image.
- `face.jpg` — **NOT actually a JPEG. It is an AVIF file with a `.jpg`
  extension.** `cv2.imread()` returns `None` on it. You MUST handle this:
  - Preferred: re-save it once as a true JPEG/PNG (e.g. via `pillow` +
    `pillow-heif`/`pillow-avif-plugin`, or `ffmpeg -i face.jpg face.png`), commit
    the fixed file, and load that. Keep the dependency footprint minimal.
  - Whatever you choose, the notebook itself must include an assertion with a
    clear error message if any image fails to load, so a student never sees a
    cryptic `NoneType` error 10 cells later.

---

## 3. Hard technical requirements

- **`static_image_mode=True`** on every MediaPipe model (these are single images,
  not video). Document why in a comment.
- **Display with matplotlib inline**, never `cv2.imshow` (no GUI in Jupyter).
  Provide one `show(img_bgr, title=...)` helper that converts BGR→RGB and shows it.
  Call out the BGR-vs-RGB gotcha in markdown the first time — it's a classic CV
  trap and reinforces "know your data format".
- **Runs top-to-bottom with no webcam and no network.** A student on a broken
  camera or locked-down WiFi must be able to complete this notebook.
- **Dependencies:** stick to what the lab already uses — `opencv-python`,
  `mediapipe`, `numpy` — plus `matplotlib`. If AVIF handling needs an extra
  package, isolate it to the one-time image-conversion step and note it in the
  README install line.
- **Don't pre-write the answers in code comments.** The pedagogy is read-the-output.
  Use markdown questions; keep answers out of the cells. (An optional instructor
  answer key at the very end, clearly fenced, is fine — see §5 final section.)

---

## 4. Notebook structure — cell by cell

Notation: **(M)** = markdown cell, **(C)** = code cell. Keep code cells short and
single-purpose so output is easy to read. Order of modalities is deliberate:
**hands (simple, 21 pts) → pose (33 pts, adds `visibility`) → face (468 pts, "just
more of the same").**

### Section 0 — Title & orientation
1. **(M)** Title: "Lab 5 — Exploring the Landmark Data". State the goal (understand
   what MediaPipe returns), why static images (freeze the data), and the
   division-of-labour blurb from §1. One line: "No webcam needed — run every cell
   top to bottom and *read the printed output*."

### Section 1 — Setup
2. **(C)** Imports: `cv2`, `mediapipe as mp`, `numpy as np`,
   `matplotlib.pyplot as plt`, `pathlib`. Set `REF_DIR = Path(...) / "ref_imgs"`.
3. **(C)** Define `show(img_bgr, title=None)` helper (BGR→RGB + `plt.imshow`,
   axis off).
4. **(M)** One short paragraph: OpenCV loads images as **BGR**, matplotlib expects
   **RGB** — hence the conversion. First taste of "the format matters."

### Section 1.5 — Load the three reference images
5. **(C)** Load all three with `cv2.imread`. Assert each is not `None` with a
   message naming the file (this is where the AVIF/`face.jpg` failure surfaces if
   unfixed). Print each image's `.shape` → introduce `(height, width, channels)`.
6. **(M)** Prompt: "Note the shape of each image: `(H, W, 3)`. The `3` is the BGR
   channels. Hold onto `H` and `W` — we'll need them to turn landmark coordinates
   into pixels." Show all three images (one `show()` call each or a small grid).

### Section 2 — HANDS (21 landmarks)
7. **(M)** What `mp.solutions.hands` is and what it returns at a high level.
8. **(C)** Init `Hands(static_image_mode=True, max_num_hands=2, ...)`. Process
   `hands.jpg` (convert to RGB first). Store `results`.
9. **(C)** Inspect the **top-level result**: `type(results.multi_hand_landmarks)`,
   and `len(results.multi_hand_landmarks)`. Print them.
10. **(M)** 🔍 Prompt: "How many hands were detected? What kind of object is
    `multi_hand_landmarks` — and what do you think it holds if **no** hand is
    found?" (Answer they should infer: a list, one entry per hand; `None` if none.)
11. **(C)** Drill into one hand: `hand = results.multi_hand_landmarks[0]`, then
    `len(hand.landmark)`. Print it.
12. **(C)** Print a **single landmark** object: `hand.landmark[8]` (index
    fingertip). Show its `.x`, `.y`, `.z` individually too.
13. **(M)** 🔍 Prompt: "How many points define a hand? What three fields does each
    landmark have?"
14. **(C)** Show the coordinate range: compute min/max of all `x` and all `y`
    across the 21 landmarks. Print them.
15. **(M)** 🔍 Prompt: "What range do `x` and `y` fall in? Are these pixels or
    something else?" → lead them to **normalized 0–1**. Brief note on `z` (relative
    depth, smaller = closer to camera; treat as approximate).
16. **(C)** Normalized → pixels, and draw it: for a few named landmarks (wrist 0,
    thumb tip 4, index tip 8), compute `px = int(x*W)`, `py = int(y*H)`, draw
    `cv2.circle` + label on a copy of the image, then `show()`.
17. **(M)** Tie-in: "This `x*W`, `y*H` conversion is exactly what the challenge
    GUIs do to draw the red 'keypoints you're using' circles. You now understand
    that debugging tool."
18. **(C)** Handedness: inspect `results.multi_handedness[0].classification[0]` —
    print `.label` and `.score`.
19. **(M)** 🔍 Prompt: "How does MediaPipe report left vs right, and how confident
    is it? Which challenge will need this?" (→ number_recognition.)
20. **(C)** *Your turn* mini-exercise: "Change the index in cell 16 from 8 to
    another fingertip (12, 16, 20) and re-run. Which point lights up?" Leave a clear
    editable spot.

### Section 3 — POSE / BODY (33 landmarks, adds `visibility`)
Mirror the hands flow, but on `body.jpg` and highlighting the differences:
21. **(M)** What `mp.solutions.pose` returns. Note: pose returns a **single**
    `pose_landmarks` (not a list of people).
22. **(C)** Init `Pose(static_image_mode=True, ...)`, process `body.jpg`. If
    `results.pose_landmarks is None`, print a clear "pose not detected on this
    image" message (this is where a too-dark `body.jpg` shows up — see §2).
23. **(C)** `len(results.pose_landmarks.landmark)` → 33. Print one landmark, e.g.
    `landmark[16]` (right wrist), and show it has **`.visibility`** in addition to
    x/y/z.
24. **(M)** 🔍 Prompt: "What new field does a pose landmark have that a hand
    landmark didn't? What might `visibility` near 0 mean?" (occluded / off-frame /
    low confidence.)
25. **(C)** Print the `visibility` of a likely-visible landmark vs a likely-hidden
    one (e.g. nose vs an ankle that may be cropped) to make the field concrete.
26. **(M)** Tie-in: "The jumping-jack challenge should check `visibility` before
    trusting a point — otherwise an off-screen ankle gives garbage coordinates."
27. **(C)** Draw a few named pose landmarks (wrists 15/16, ankles 27/28, shoulders
    11/12) on the image with the same `x*W, y*H` conversion. `show()`.

### Section 4 — FACE (468 landmarks)
28. **(M)** What `mp.solutions.face_mesh` returns; set expectation: "Don't panic —
    468 points, but it's *the same structure* you already understand. You'll only
    ever use a handful."
29. **(C)** Init `FaceMesh(static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, ...)`, process `face.jpg`. Print
    `len(results.multi_face_landmarks[0].landmark)` (478 with refine_landmarks).
30. **(M)** 🔍 Prompt: "How many points now? Is a single landmark any different in
    structure from a hand landmark?" (No — same x/y/z.)
31. **(C)** Draw the smile-relevant landmarks from the challenge docstring (mouth
    corners 61 & 291, lips 13 & 14) on the face, labeled. `show()`.
32. **(M)** Tie-in to smile_detection: "These are the exact indices the smile
    challenge suggests. Now you can *see* where they sit." Foreshadow the next
    teaching step: "Notice raw pixel distance between mouth corners changes when
    you move closer/farther from the camera — that's why robust detectors use
    **ratios**, which the landmark-reference material covers next."

### Section 5 — Recap & bridge to the challenges
33. **(M)** Summary table:

    | Model | Result attribute | # landmarks | Per-landmark fields | Multiple subjects? |
    |-------|------------------|-------------|---------------------|--------------------|
    | Hands | `multi_hand_landmarks` (+ `multi_handedness`) | 21 | x, y, z | list, one per hand |
    | Pose  | `pose_landmarks` | 33 | x, y, z, **visibility** | single |
    | Face  | `multi_face_landmarks` | 468 (478 refined) | x, y, z | list, one per face |

34. **(M)** "Three things that carry into every challenge": (1) landmark **indices**
    identify body parts; (2) coords are **normalized 0–1**, multiply by W/H for
    pixels; (3) you'll compare points by **ratios**, not raw distances, for
    camera-distance robustness (next material).
35. **(M)** §6 cross-reference (below) — the most important bridge. Then: "Next:
    run the live `.py` scripts to see this in motion, then open
    `challenges/README.md`."

---

## 5. (Optional) instructor answer key
At the very bottom, in a clearly fenced/marked section a student can ignore,
provide one-line answers to each 🔍 prompt. Mark it explicitly as instructor
reference so it doesn't short-circuit the inspection exercise.

---

## 6. CRITICAL: connect the notebook to what the challenges actually receive

The notebook teaches the *raw* MediaPipe structure, but each challenge's
`detect_*()` function receives a **pre-unpacked** slice of it, and the shape
**differs per challenge**. Add a markdown cell (in §5) that lays this out
explicitly, because this mismatch is a top source of student confusion:

- **smile_detection** → `detect_emotion(face_landmarks)` gets the **flat
  `.landmark` list** (the 468/478 points directly), already indexed per face.
- **jumping_jack_counter** → `detect_jumping_jack(pose_landmarks)` gets the **flat
  `.landmark` list** of 33 pose points.
- **clapping_counter** → `detect_clap(hand_landmarks_list)` gets a **list of
  `.landmark` lists** (one per hand), **no handedness**.
- **number_recognition** → `count_fingers(hand_data_list)` gets a **list of
  `(landmark_list, "Left"/"Right")` tuples** — the only challenge that passes
  handedness alongside the points.
- **thumbs_decision** → `detect_thumbs_decision(hand_landmarks_list)` gets a list
  of hand landmark lists.

Point students at the relevant `main.py` (the few lines that build the argument
before calling `detect_*`) so they can match "raw result" → "what my function
sees". Reference files:
`challenges/number_recognition/main.py` (the tuple-building loop) and
`challenges/clapping_counter/main.py` (the list-building loop) are the clearest
two to cite.

---

## 7. Deliverable & integration
- Write the notebook to: `AppCV_2026/Lab_5/familiarization/explore_landmark_data.ipynb`
- Fix/commit the `face.jpg` issue (§2) so the notebook loads it cleanly.
- Add `matplotlib` (and any AVIF dep, if used) to the install line in
  `AppCV_2026/Lab_5/README.md`, and add one line to the README's "Phase 1:
  Familiarization" section pointing students to **run this notebook first**, then
  the live `.py` scripts.
- Verify the notebook executes top-to-bottom with no errors and no webcam before
  considering it done. If `body.jpg` pose detection is poor, note it and propose a
  replacement rather than shipping a section that shows nothing.
