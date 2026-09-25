# 6D Pose Estimation using Classical Image Processing

Marker-based and markerless 6D pose estimation built with OpenCV and NumPy.
The markerless tracker handles the unmarked dark cube across face-on,
transition, side, top, back, and isometric views, with symmetry-relative
orientation and short occlusion holding.

## Run

The application expects the calibrated camera at `/dev/video4` by default.

```bash
python3 6d.py
```

Press `q` to quit. To run the calibration workflow instead:

```bash
python3 6d.py --calibrate
```

Camera selection, resolution, and cube settings are in
`pose6d/settings.py`. The bundled `camera_calibration.npz` is the calibrated
1280×720 camera file used by the application.

## Verify

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile 6d.py pose6d/*.py tests/test_cube_all_views.py
```

The synthetic tests cover face-on, four-/five-/six-corner transitions,
symmetry-relative tracking, and the held/stale timeout policy.
