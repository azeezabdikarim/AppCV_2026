# Lab 2 AR extension

This directory holds the optional Lab 2 extension on image-aligned augmentation.

Expected inputs:

- intrinsics at `../captured_points/<board>/intrinsics.yml`
- a short video clip at `../data/<board>/board_clip.mp4`

Main notebook:

- `calibration_ar_notebook.ipynb`

Outputs written by the notebook:

- `outputs/axes_preview.mp4`
- `outputs/cuboid_overlay.mp4`
- `outputs/cuboid_overlay_perturbed.mp4`

The extension is designed around a short Pi-captured clip copied to the laptop. The notebook then:

1. samples frames,
2. detects the board,
3. estimates pose,
4. draws axes,
5. draws a wireframe cuboid,
6. re-renders the result with perturbed camera parameters.
