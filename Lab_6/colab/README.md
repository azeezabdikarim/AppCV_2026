# Colab notebook status

The workspace contains a text extraction of the 2025 notebook at:

```text
Fine_Tuning_YoloV8nano-2_extract.txt
```

It does not contain the original executable `.ipynb` file. The 2026 notebook
therefore still needs to be reconstructed or recovered before the lab is
released.

The intended 2026 workflow uses one image collection and one CVAT
`annotations.xml` export. The notebook must:

1. accept the captured images and CVAT `annotations.xml`;
2. convert the XML boxes to YOLO label files;
3. split matching image/label pairs into physical train and validation
   directories;
4. point `data.yaml` at those two directories.

The text extraction currently says the notebook will split the dataset, but
its shown `data.yaml` points both `train` and `val` to the same images. That
cell must be replaced in the recovered notebook.

The target notebook path is:

```text
Lab_6/colab/fine_tune_yolov8n.ipynb
```
