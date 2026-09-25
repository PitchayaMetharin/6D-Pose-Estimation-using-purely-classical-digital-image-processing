# Luna/Max Implementation Plan — Stabilize `6d.py`

## Summary

Repair the markerless 30 mm cube pipeline in `6d.py`. The implementation must replace the incorrect six-corner geometry, prevent false detections from corrupting tracking, use valid rotation filtering, survive temporary camera-frame failures, and preserve the existing calibration and ArUco workflows.

Only `6d.py` may be modified. Use OpenCV and NumPy only.

## Implementation Changes

### 1. Replace the cube pose model and solver

- Define all eight cube vertices around the cube center at `±15 mm`; make pose `tvec` represent the center directly.
- Generate valid six-vertex silhouette cycles from the cube’s edge graph instead of using the current hard-coded correspondence pattern.
- For six detected corners, enumerate valid cyclic/reversed correspondence candidates and solve each with `SOLVEPNP_ITERATIVE`.
- For four corners, enumerate all six cube faces and both solutions returned by `solvePnPGeneric(..., SOLVEPNP_IPPE)`.
- Reject candidates with non-finite values, any vertex behind the camera, or fit error above `max(3 px, 3% of the observed silhouette diagonal)`.
- Score candidates using paired reprojection RMS plus bidirectional observed/projected hull-edge error. Never accept a pose merely because it is the lowest-scoring candidate.
- Refine accepted contour corners to subpixel precision before pose estimation.

### 2. Handle cube symmetry and stabilize orientation

- Generate the 24 proper rotational symmetries of a cube.
- Express every new rotation using the symmetry-equivalent orientation nearest to the previous accepted orientation. For the first frame, choose the equivalent orientation with the smallest rotation-vector magnitude.
- Replace linear Rodrigues-vector averaging with interpolation on SO(3): compute the relative rotation, scale its Rodrigues vector by the smoothing alpha, and compose it with the previous rotation.
- Make smoothing time-based with a `0.12 s` time constant instead of frame-dependent.
- Keep X and Y signed, require Z to be positive, and add an always-positive `Range` value.
- Label Euler telemetry `RPY relative` because an unmarked cube has no uniquely identifiable face orientation.

### 3. Make detection and tracking transactional

- Evaluate candidate contour, pose, geometric fit, and temporal continuity before changing any tracking state.
- Store separate `last_raw_pose` and `filtered_pose`; use raw poses for continuity checks and filtered poses for drawing.
- Apply center, scale and pose-continuity checks during both four-to-six and six-to-four transitions. Remove the current point-count-change bypass.
- Use time-based limits: center velocity ≤ `1.0 m/s`, Z velocity ≤ `0.75 m/s`, and angular velocity ≤ `360°/s`.
- While tracking, rank candidates by geometric fit first and temporal proximity second. A rejected candidate must not alter tracked points or pose.
- Hold the last accepted pose for at most `0.75 s` during occlusion, visibly mark it as stale, then reset to global acquisition.
- Preserve automatic acquisition, but assume the cube is the dominant dark cube-like object; a plain dark square is fundamentally indistinguishable from a frontal unmarked cube.

### 4. Recover from camera failures

- Centralize camera creation and configuration in `open_configured_camera()`.
- On a failed read, retry five times with short delays. If still failing, release and reopen `/dev/video4` up to three times.
- Resume processing without resetting a valid track after a successful reconnect.
- Exit with a clear error only after all reopen attempts fail.
- Use the same recovery path in normal and calibration modes.

### 5. Strengthen calibration without breaking the existing file

- Continue loading the current `camera_calibration.npz` format.
- Validate positive focal lengths, matrix normalization, plausible principal point, finite distortion values and optional RMS metadata.
- Reject near-duplicate chessboard captures using board-center, apparent-area and orientation differences.
- Use `calibrateCameraExtended`, save per-view reprojection errors, and refuse results with overall RMS above `1.0 px` or any view above `2.0 px`.
- Do not overwrite the existing calibration automatically; only replace it during an explicit `--calibrate` run.

## Verification and Acceptance

- Both `python3 -m py_compile 6d.py` and `vision_env/bin/python -m py_compile 6d.py` must pass.
- Synthetic noise-free poses across multiple distances and roll/pitch/yaw combinations must recover:
  - projected hull error ≤ `0.5 px`;
  - center error ≤ `1 mm`;
  - rotation error modulo cube symmetry ≤ `1°`.
- With Gaussian corner noise of `1 px`, 95% of cases must stay below `3 px` hull error and `10 mm` depth error.
- The `+179°`/`−179°` regression must interpolate along the 2° shortest path, never produce the previous 161° jump.
- Rejected contours and temporary hand occlusions must leave tracking state unchanged and reacquire the original cube.
- Simulated temporary frame failures must recover; persistent failures must terminate only after the configured retries.
- Load and use the existing 1280×720 calibration successfully.
- Live-test `/dev/video4` for at least 60 seconds:
  - no termination from an isolated dropped frame;
  - no pose switch when a hand briefly crosses the cube;
  - wireframe remains aligned at tilted camera angles;
  - stationary Z jitter stays under `10 mm` and orientation jitter under `3°`.

## Assumptions

- The target remains an unmarked, dark, 30 mm cube.
- ArUco support and generic 2D shape detection remain functional and otherwise unchanged.
- Ball 6D orientation is deferred. A featureless sphere can provide center and radius, but not observable orientation or meaningful axes without texture or a marker.
- No new dependencies or tracked files are introduced.
