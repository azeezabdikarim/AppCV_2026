# Notes for 2027

## 1. Board geometry config needs to be split by board type

The current `src/board_config.py` uses one shared `SQUARE_SIZE_M` for both the checkerboard and the ChArUco board, plus one `CHARUCO_MARKER_SIZE_M` for the marker size. That is not correct for the real boards used in class if the checkerboard square pitch and the ChArUco outer-square pitch differ physically.

This matters most for:

- `src/measure_object.py`, because world distances will be scaled incorrectly if the wrong square size is used.
- `ar_extension/ar_helpers.py`, because board extents and pose scale are derived from the board geometry.
- `boards/generate_board.py`, because regenerated printable assets should reflect the real intended dimensions.

For 2027, the config should use explicit board-specific names instead:

- `CHECKERBOARD_SQUARE_SIZE_M`
- `CHARUCO_SQUARE_SIZE_M`
- `CHARUCO_INNER_MARKER_SIZE_M`

The rest of Lab 2 should then import the appropriate constant instead of relying on a generic "default square size".

## 2. The calibration notebook currently asks students for board dimensions, but those values do not drive calibration

This is a pedagogical bug in:

- `src/calibration_notebook.ipynb`
- `src/calibration_notebook_solutions.ipynb`
- `src/calibrate.ipynb` if that older notebook copy is still kept around or reused

Section 2 asks students to enter:

- checkerboard inner-corner counts
- ChArUco square counts
- square size
- marker size

Those values are printed back to the user, but they do not actually affect the generated intrinsics.

### What currently happens

In the notebook, the section-2 values are local variables that are only displayed. The later calibration path still uses geometry from `board_config.py` and from the precomputed `corners.npz` file:

- In checkerboard mode, the notebook loads `objpoints` directly from `captured_points/checker/corners.npz`.
- Those `objpoints` were already created earlier by `src/detect_corners.py`, which calls `checkerboard_object_points()` from `src/board_config.py`.
- So by the time the notebook runs, the checkerboard world geometry is already baked into the saved file.

- In ChArUco mode, the notebook creates `board = create_charuco_board()`.
- That board is built from the constants in `src/board_config.py`, not from the values the student typed in section 2.

As a result, a student can change the numbers in section 2 and still get the same intrinsics, which is confusing because the markdown text explicitly says those measurements matter for the calibration scale.

## 3. Why this is confusing for students

The notebook text says:

- if the square size or marker size is wrong, the calibration scale and later measurement step will also be wrong

That statement is true in general, but in the current notebook flow the section-2 edits do not propagate into the actual calibration geometry. So the notebook teaches the right concept while the code behaves as if those inputs are advisory only.

## 4. Recommended fix direction

For 2027, section 2 should become the source of truth for the calibration geometry used later in the notebook.

### Checkerboard

Instead of relying on checkerboard `objpoints` that were already saved in `corners.npz`, the notebook should derive the checkerboard object points from the student's section-2 inputs before calling `cv2.calibrateCamera`.

That likely means changing the data flow so that checkerboard detection saves image points and image metadata, but not pre-scaled world points. Then the notebook can rebuild the world points from:

- checkerboard inner-corner counts
- checkerboard square size

### ChArUco

The notebook should build the ChArUco board used for calibration from the student's section-2 values rather than from `create_charuco_board()` in `board_config.py`.

That means the calibration board should be parameterized by:

- ChArUco square counts
- ChArUco square size
- ChArUco inner marker size

### General rule

The same geometry values should drive all of the following:

- section-2 reference/validation prints
- expected point-count checks
- the board object used for calibration
- any later planar measurement step

If we keep a central `board_config.py`, it should either:

- match the exact printed boards used in class and be clearly presented as the source of truth, or
- be used only for defaults, while the notebook rebuilds the calibration geometry from student-entered values and makes that dependency explicit.

Right now it is halfway between those two models, which is what causes the confusion.
