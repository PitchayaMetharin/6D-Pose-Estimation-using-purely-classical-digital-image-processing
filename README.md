# Classical 6D Pose Estimation

A calibrated, explainable OpenCV system for estimating the pose of an unmarked cube and the position of colour-identified spheres from a single camera. It uses classical digital image processing and projective geometry only—no machine learning or deep learning model.

## What it tracks

- An unmarked 30 mm cube: 3D centre and symmetry-relative rotation
- A 20.0 mm-radius small ball: 3D centre and range
- A 37.5 mm-radius large ball: 3D centre and range
- Optional 40 mm ArUco markers as a marked-object reference

Ball tracking is deliberately orientation-free because a featureless sphere has no observable body-fixed rotation from one image.

## Documentation

The complete theory, equations, implementation record, verification results, screenshots, presentation outline, speaker notes, and likely assessment questions are in:

- [Mini-project report](6D_Pose_Estimation_Mini_Project_Report.md)
- [Implementation plan and defect analysis](6d_plan.md)

## Run

The verified environment uses Python 3.10, OpenCV with the `aruco` module, NumPy, Linux/V4L2, and a 1280 × 720 camera stream.

```bash
cd 6D_Pose
../vision_env/bin/python 6d.py
```

Calibrate the camera when the lens, focus, zoom, or capture resolution changes:

```bash
../vision_env/bin/python 6d.py --calibrate
```

The repository contains the calibration used for the documented demonstration. A different camera requires its own calibration.

## Runtime controls

| Control | Action |
|---|---|
| `1` | Select the 20.0 mm-radius `small_ball` profile |
| `2` | Select the 37.5 mm-radius `large_ball` profile |
| Left click | Calibrate the selected ball colour from a local HSV patch |
| `S` | Save both colour profiles |
| `R` | Reset the selected profile and its tracker |
| `H` | Toggle handheld partial cube tracking |
| `Q` | Quit |

## Verification

```bash
python3 -m pytest -q
python3 -m unittest discover -s tests -q
python3 -m py_compile 6d.py pose6d/*.py tests/*.py
```

Current result: **14 tests pass**. See the report for the distinction between automated regression tests, noise-free synthetic diagnostics, and live observations.

## Design principle

A numerical pose is not accepted merely because PnP can fit it. Full and partial cube paths require independent silhouette or physical-edge support, positive depth, geometry agreement, symmetry-aware motion checks, and transactional tracker updates. Unsupported observations become `HELD/STALE` briefly and then `LOST` instead of producing a floating pose.
