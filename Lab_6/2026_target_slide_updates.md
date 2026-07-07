# Lab 6 (2026): target slide-by-slide updates

## Purpose and scope

This document defines the target 2026 revision of the 32-slide 2025 deck, **Lab 6: YOLOv8 Fine Tuning: Detecting Custom Signs**.

Lab 6 remains a model-development lab:

1. Use a PiCar-X camera to capture custom-sign images.
2. Annotate the images in CVAT.
3. Fine-tune YOLOv8n in Google Colab.
4. Use the supplied notebook to train and evaluate the model.
5. Export `best.onnx`.

Deployment is a separate, minimal car-lab week in:

```text
AppCV_2026/Car_lab/week4_object_detection/
```

That later exercise will load the ONNX model, filter detections by class and confidence, calculate how much of the camera frame the sign occupies, and issue a stop decision. Monocular depth estimation remains a later lab and is not part of this revision.

Proposed Lab 6 directory:

```text
AppCV_2026/Lab_6/
├── README.md
├── scripts/
│   └── capture_training_images.py
├── captured_images/             # created by students; ignored by Git
├── colab/
│   └── fine_tune_yolov8n.ipynb
└── test_images/                  # a few supplied smoke-test images
```

## CVAT format decision

### 2026 flow

Use one image collection and one CVAT annotation task. Export the annotations as
`annotations.xml`, matching the format expected by the supplied Colab notebook.
Students do not create separate training and validation capture folders.

### How this changes the 2025 notebook

The 2025 flow was:

```text
original JPG files + annotations.xml
    -> custom XML parser
    -> custom coordinate conversion
    -> YOLO TXT files
    -> data.yaml
    -> training
```

The 2026 flow remains:

```text
captured_images/*.jpg + annotations.xml
    -> notebook XML parser
    -> YOLO TXT files
    -> data.yaml
    -> training
```

The notebook converts each CVAT bounding box into the normalized YOLO format:

```text
class_id center_x center_y width height
```

### Required Step 3 change

Step 3 should retain **“Upload Your Data: images and `annotations.xml`”**.
Students upload every `img_XXX.jpg` file from `captured_images/` into the
notebook's images directory and upload the single CVAT `annotations.xml` file
into its annotations directory.

---

## Slide-by-slide replacement text

### Slide 1 — Replace

**Title**

> Lab 6: Fine-Tuning YOLOv8n  
> Detecting Custom Signs

**Subtitle**

> Applied Computer Vision — 2026  
> From camera images to a deployable ONNX model

---

### Slide 2 — Replace

**Title**

> Understanding Images: The Computer Vision Challenge

**Body**

> How do we teach a computer to interpret visual content?
>
> **Classification — What is in this image?**  
> Output: one label for the complete image, such as car, cat or bicycle.
>
> **Object detection — What objects are present, and where are they?**  
> Output: a class label, confidence score and bounding box for each detected object.
>
> **Segmentation — Which pixels belong to each object?**  
> Output: a pixel-level mask.
>
> **Today’s focus:** fine-tuning an object detector for custom signs.

---

### Slide 3 — Replace

**Title**

> Supervised and Unsupervised Learning

**Body**

> **Supervised learning — used in this lab**
>
> - Training examples include the correct answers.
> - Each sign is marked with a class label and bounding box.
> - The model learns to predict those labels and boxes in new images.
>
> **Unsupervised learning**
>
> - Training examples do not include target labels.
> - The algorithm searches for structure or patterns in the data.
>
> **Our task:** supervised object detection using annotated camera images.

---

### Slide 4 — Replace title and body

**Title**

> From Image Classification to Real-Time Object Detection

**Body**

> Important milestones:
>
> - Large labelled datasets made data-driven visual recognition possible.
> - Convolutional neural networks greatly improved image classification.
> - Two-stage detectors first propose candidate regions and then classify them.
> - Single-stage detectors predict boxes and classes in one model pass.
> - The YOLO family made fast, practical object detection widely accessible.
>
> **Why this matters:** our car needs both the identity and location of a sign.

---

### Slide 5 — Replace

**Title**

> Two-Stage and Single-Stage Detection

**Body**

> **Two-stage detector**
>
> 1. Propose possible object regions.
> 2. Classify and refine each region.
>
> Examples include Faster R-CNN and Mask R-CNN.
>
> **Single-stage detector**
>
> - Predicts object locations and classes in one model pass.
> - Usually provides a useful speed–accuracy trade-off for real-time systems.
>
> **Our choice:** a small YOLO model suitable for later deployment on the PiCar-X.

---

### Slide 6 — Retain the comparison visual; replace its caption

**Caption**

> Two-stage methods separate region proposal from classification. YOLO predicts candidate boxes and class scores in one pass, followed by confidence filtering and non-maximum suppression.

---

### Slide 7 — Replace

**Title**

> What the Pre-Trained Model Already Knows

**Body**

> YOLOv8n starts with weights learned from the COCO dataset.
>
> It already contains useful visual features for:
>
> - edges, corners and textures;
> - common shapes and object parts;
> - spatial patterns found in everyday scenes;
> - common classes such as people, cars and traffic lights.
>
> Our custom labels — `Stop_Sign`, `TU_Logo`, `Stahp` and `Falling_Cows` — are not the four classes we need from the original model.
>
> We therefore adapt the pre-trained model using our own annotated examples.

---

### Slide 8 — Replace

**Title**

> Why We Freeze the Course Model

**Body**

> The YOLO family continues to change, but a teaching lab needs a stable model and output contract.
>
> For the 2026 lab we will use:
>
> - the `yolov8n.pt` starting checkpoint;
> - a tested and pinned Ultralytics package version in Colab;
> - four custom output classes;
> - a fixed-size ONNX export for the Raspberry Pi.
>
> This keeps every group’s notebook and deployment code compatible.

---

### Slide 9 — Replace

**Title**

> The Complete Lab Workflow

**Body**

> PiCar-X image capture  
> → CVAT annotation  
> → Ultralytics YOLO dataset export  
> → Colab training  
> → validation on unseen images  
> → ONNX export  
> → later deployment in Car Lab Week 4
>
> Each stage has a clear input and output. Diagnose problems at the stage where they occur.

---

### Slide 10 — Replace

**Title**

> YOLOv8 Nano: A Small Deployment Model

**Body**

> `YOLOv8n` is the smallest standard YOLOv8 detection model.
>
> We use it because it offers:
>
> - a small model artifact;
> - faster CPU inference than larger variants;
> - transfer learning from a general object-detection model;
> - export to ONNX for a lightweight car runtime.
>
> **Engineering trade-off:** smaller models are faster, but may miss small, partly hidden or unusual signs.
>
> We will measure performance on the target Pi rather than claiming a desktop frame rate.

---

### Slide 11 — Replace

**Title**

> What Is Image Annotation?

**Body**

> Image annotation creates the target answers used for supervised learning.
>
> For every visible training object, we provide:
>
> - a class label; and
> - a tight rectangular bounding box.
>
> The annotations tell the model both **what** it should detect and **where** the object appears.
>
> Applications include autonomous systems, medical imaging, manufacturing inspection and inventory monitoring.

---

### Slide 12 — Retain the example images; replace the caption

**Caption**

> Different tasks require different annotation types. This lab uses axis-aligned bounding boxes because our output is object detection, not classification or segmentation.

---

### Slide 13 — Replace

**Title**

> Annotation Tools and Our Choice of CVAT

**Body**

> Professional annotation tools include CVAT, Roboflow, Supervisely and Label Studio.
>
> We use CVAT because it provides:
>
> - browser-based rectangle annotation;
> - shared class definitions across a project;
> - review and correction of annotations;
> - direct export in Ultralytics YOLO detection format.
>
> Use Chrome or Firefox for the lab. Save your work regularly and verify the export before leaving CVAT.

---

### Slide 14 — Replace

**Title**

> 2026 Lab Setup

**Body**

> The course repository and `app_cv` conda environment are already installed on the car from the earlier car labs.

**Code block**

```bash
ssh cvpcar#@cvpcar#.local

cd ~/AppCV_2026
git pull
conda activate app_cv

cd ~/AppCV_2026/Lab_6
```

**Repository structure**

```text
Lab_6/
├── scripts/
│   └── capture_training_images.py
├── captured_images/       # created during the lab
├── colab/
│   └── fine_tune_yolov8n.ipynb
└── test_images/
```

**Footer**

> Training happens in Google Colab. The car is used here to collect the camera images.

---

### Slide 15 — Replace

**Title**

> Capture the Training Images

**Body**

> Capture one varied image collection. Every image is saved directly in
> `Lab_6/captured_images/` and will be uploaded to one CVAT annotation task.

**Code block**

```bash
cd ~/AppCV_2026/Lab_6
conda activate app_cv

python scripts/capture_training_images.py \
  --num-images 80 --delay 0.5
```

**Capture checklist**

> - Vary distance, angle, position and scale.
> - Use multiple backgrounds and lighting conditions.
> - Include some frames with more than one sign.
> - Include background images containing no target sign.
> - Move the signs and change the scene throughout the capture.
>
> Running the command again continues the numbering in the same folder, so
> existing images are not overwritten.

**Transfer to the laptop**

```bash
scp -r cvpcar#@cvpcar#.local:~/AppCV_2026/Lab_6/captured_images .
```

---

### Slide 16 — Replace

**Title**

> Coordinate Systems: Pixels and Normalized YOLO Boxes

**Body**

> Images use pixel coordinates with `(0, 0)` at the top-left.
>
> A rectangle can be represented by its corners:
>
> ```text
> x_left, y_top, x_right, y_bottom
> ```
>
> Ultralytics YOLO stores each box as:
>
> ```text
> class_id center_x center_y width height
> ```
>
> The four box values are divided by the image width or height, so they normally lie between 0 and 1.
>
> **Why normalize?** The annotation describes the same relative location even when an image is displayed at a different resolution.

---

### Slide 17 — Replace

**Title**

> Transfer Learning and Fine-Tuning

**Body**

> **Training from scratch**
>
> - starts with random weights;
> - needs a very large dataset;
> - requires more computation and training time.
>
> **Fine-tuning — our approach**
>
> - starts from `yolov8n.pt`;
> - reuses general visual features learned from a large dataset;
> - adapts the model to our four custom classes;
> - can work with a much smaller, carefully collected dataset.
>
> Fine-tuning does not remove the need for representative data and honest evaluation.

---

### Slide 18 — Replace completely

**Title**

> One Dataset, One Annotation Task

**Body**

> All captured images belong to one dataset:
>
> ```text
> Lab_6/captured_images/
> ├── img_000.jpg
> ├── img_001.jpg
> └── ...
> ```
>
> Upload the complete folder to one CVAT task and export one
> `annotations.xml` file. Do not manually create `train/` and `val/` capture
> folders; the supplied Colab notebook controls the dataset used by YOLOv8.

---

### Slide 19 — Replace

**Title**

> Single-Object, Multi-Object and Background Images

**Body**

> Our dataset should include:
>
> - images containing one sign;
> - images containing multiple signs;
> - signs at different positions and apparent sizes;
> - background images containing none of the four classes.
>
> Every visible target object should be annotated. Background images receive no bounding boxes.
>
> A detector must learn both when to produce a detection and when not to produce one.

---

### Slide 20 — Replace

**Title**

> Create the CVAT Project and Task

**Body**

> 1. Open CVAT and create a project named `Custom Signs Detection`.
> 2. Add these labels in this exact order:
>
> ```text
> Stop_Sign
> TU_Logo
> Stahp
> Falling_Cows
> ```
>
> 3. Create one image-annotation task inside the project.
> 4. Upload every `img_XXX.jpg` file from `Lab_6/captured_images/`.
>
> Exact spelling and class order matter because the exported numeric class IDs follow the project label mapping.

---

### Slide 21 — Replace

**Title**

> Annotation Workflow

**Body**

> For each image:
>
> 1. Select the correct class.
> 2. Draw a tight rectangle around the complete visible sign.
> 3. Check whether any additional target signs are present.
> 4. Leave true background images empty.
> 5. Save your work and continue to the next image.
>
> Use the shortcuts displayed by the current CVAT interface; shortcuts can change between CVAT versions.
>
> Before export, review examples from every class and across the complete dataset.

---

### Slide 22 — Replace

**Title**

> Annotation Quality Controls Model Quality

**Body**

> **Good annotations**
>
> - use the correct class every time;
> - tightly cover the visible object;
> - include every relevant object in the image;
> - use a consistent policy for partial or occluded signs.
>
> **Common problems**
>
> - loose boxes containing large amounts of background;
> - missing signs;
> - inconsistent class names;
> - duplicated near-identical views;
> - inconsistent policies for partially visible signs.
>
> Inspect the exported labels visually in Colab before training.

---

### Slide 23 — Replace completely

**Title**

> Export the CVAT Annotations and Upload to Colab

**Body**

> **Export from CVAT**
>
> 1. Open the task’s **Actions** menu.
> 2. Choose **Export task dataset**.
> 3. Select **CVAT for images 1.1**.
> 4. Export and locate the generated `annotations.xml` file.
>
> **Upload to the supplied Colab notebook**
>
> - Upload all `img_XXX.jpg` files to `/content/stop_sign_dataset/images`.
> - Upload `annotations.xml` to `/content/stop_sign_dataset/annotations`.
>
> **The notebook will**
>
> - read the CVAT XML annotations;
> - convert each bounding box to YOLO format;
> - create one `.txt` label file for each image;
> - create the final `data.yaml` used for training.

---

### Slide 24 — Replace

**Title**

> Understanding Model Performance

**Body**

> **Intersection over Union — IoU**
>
> IoU measures overlap between a predicted box and the annotated ground-truth box:
>
> ```text
> IoU = intersection area / union area
> ```
>
> **Precision** asks: when the model reports a sign, how often is it correct?
>
> **Recall** asks: of the signs that are present, how many did the model find?
>
> **mAP50** summarizes precision and recall using an IoU threshold of 0.50.
>
> **mAP50–95** averages across stricter IoU thresholds.
>
> These values are only meaningful when validation images are independent of training images. Also inspect false positives, missed signs and predictions on unseen scenes.

---

### Slide 25 — Replace

**Title**

> The Output of Lab 6

**Body**

> After training, keep these artifacts:
>
> ```text
> best.pt              # best training checkpoint
> best.onnx            # deployment model
> model_metadata.json  # class names, input size and export details
> results.csv          # training history
> confusion_matrix.png
> validation_examples/
> ```
>
> The required submission artifact for the later car exercise is:
>
> ```text
> best.onnx
> ```
>
> Do not deploy `last.pt`. Use the checkpoint selected from validation performance and exported by the supplied notebook.

---

### Slide 26 — Replace

**Title**

> What the ONNX Model Produces

**Body**

> The deployment pipeline will be:
>
> ```text
> camera frame
>     -> model input tensor
>     -> ONNX inference
>     -> candidate boxes and class scores
>     -> confidence filtering
>     -> non-maximum suppression
>     -> final detections
> ```
>
> Each final detection should provide:
>
> ```text
> class name
> confidence score
> bounding box
> ```
>
> Lab 6 creates and validates the model. Car Lab Week 4 will turn those outputs into a simple driving decision.

---

### Slide 27 — Replace completely

**Title**

> Two Execution Environments, Two Jobs

**Body**

> **Google Colab — training and export**
>
> - Ultralytics and PyTorch
> - GPU training
> - evaluation plots
> - export from `best.pt` to `best.onnx`
>
> **PiCar-X `app_cv` conda environment — capture and inference**
>
> - camera capture
> - OpenCV and NumPy
> - ONNX Runtime
> - car controls and debug interface
>
> The 2026 car does not create a new `venv` for this lab and does not train the model locally.

**Car environment check**

```bash
cd ~/AppCV_2026
git pull
conda activate app_cv

python -c "import cv2, numpy, onnxruntime; print('app_cv ready')"
```

---

### Slide 28 — Replace

**Title**

> Verify the Exported ONNX Model in Colab

**Body**

> Before downloading the model:
>
> 1. Load `best.onnx` using the notebook’s ONNX test cell.
> 2. Run it on several validation images.
> 3. Display the final boxes, class names and confidence scores.
> 4. Compare the ONNX result with the `best.pt` result.
> 5. Record the model input shape and output shape.
>
> Do not wait until the car exercise to discover that the exported model or class mapping is incorrect.

---

### Slide 29 — Replace

**Title**

> Download and Preserve the Model Artifact

**Body**

> Download the following files from Colab:
>
> ```text
> best.onnx
> model_metadata.json
> validation_predictions.zip
> ```
>
> Rename the folder using your group number:
>
> ```text
> group_XX_model/
> ├── best.onnx
> ├── model_metadata.json
> └── validation_predictions/
> ```
>
> Keep a local copy. Colab runtimes are temporary and may be reset.

---

### Slide 30 — Replace

**Title**

> Next Step: Car Lab Week 4

**Body**

> In the separate car exercise you will:
>
> 1. Copy `best.onnx` into `Car_lab/week4_object_detection/models/`.
> 2. Load the model using ONNX Runtime.
> 3. Run inference on frames from the shared car camera.
> 4. Keep detections above a confidence threshold.
> 5. Select the target sign class.
> 6. Calculate the bounding-box area as a fraction of the image.
> 7. Request a stop when the sign is sufficiently large and consistently detected.
>
> The car integration will be intentionally small. The model-development work belongs to Lab 6.

---

### Slide 31 — Replace

**Title**

> Apparent Size Is a Useful Proxy, but It Is Not Depth

**Body**

> In the next car exercise we will use:
>
> ```text
> box_area_ratio = (box_width × box_height) /
>                  (frame_width × frame_height)
> ```
>
> A larger image fraction often means the same sign is closer, but the relationship also depends on:
>
> - the sign’s physical size;
> - camera angle and field of view;
> - partial occlusion;
> - the accuracy of the predicted box;
> - the model’s inference delay and the car’s stopping distance.
>
> This limitation motivates the later monocular depth-estimation lab, where the system will reason about scene depth rather than use box size alone.

---

### Slide 32 — Replace

**Title**

> Lab 6 Completion Checklist

**Body**

> By the end of this lab, your group should have:
>
> - one varied image collection in `captured_images/`;
> - consistent CVAT bounding-box annotations;
> - one CVAT `annotations.xml` export;
> - visual verification of several exported labels;
> - a fine-tuned YOLOv8n model;
> - training and evaluation results from the supplied notebook;
> - ONNX predictions checked in Colab;
> - a downloaded and backed-up `best.onnx` file.
>
> **Handoff:** bring `best.onnx` to Car Lab Week 4.

---

## Implementation status before publishing the slides

Completed in the repository:

1. `AppCV_2026/Lab_6/scripts/capture_training_images.py` supports `--num-images` and `--delay`, and saves every image directly in `captured_images/`.
2. The capture code checks for the current `rpicam-still` command and the older `libcamera-still` fallback.
3. The Week 4 handout, model path and detection-dictionary contract now exist.
4. Week 4 contains separate student and instructor detector implementations.
5. Depth-specific scaffolding has moved to `Car_lab/week5_depth_estimation`.
6. `onnxruntime` is declared in the `app_cv` environment.

Still required:

1. Add the supplied executable Colab notebook to the repository.
2. Pin and test the chosen Ultralytics version and fixed ONNX input size.
3. Run the capture, environment and ONNX smoke tests on a physical Pi 5.

Until those checks pass, the command blocks in this document are target text rather than released student instructions.
