# Colab notebook status

The workspace contains a text extraction of the 2025 notebook at:

```text
Fine_Tuning_YoloV8nano-2_extract.txt
```

It does not contain the original executable `.ipynb` file. The 2026 notebook
therefore still needs to be reconstructed or recovered before the lab is
released.

The 2026 notebook must differ from the extracted 2025 version in two important
ways:

1. accept CVAT's **Ultralytics YOLO Detection 1.0** ZIP directly instead of
   parsing `annotations.xml`;
2. create genuinely separate training and validation datasets instead of
   pointing both entries in `data.yaml` at the same image directory.

The target notebook path is:

```text
Lab_6/colab/fine_tune_yolov8n.ipynb
```

