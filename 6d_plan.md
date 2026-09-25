# Concurrent Cube and Dual-Ball Tracking Plan

## Outcome

Extend the existing OpenCV application so it can track the unmarked cube and
two genuine spherical balls concurrently:

- `small_ball`: configurable radius, default `20.0 mm`.
- `large_ball`: configurable radius, default `37.5 mm`; treated as a plain
  sphere.
- Both balls report their 3D center (`X`, `Y`, `Z`) and straight-line range.
  Sphere rotation is explicitly unavailable.
- The cube continues to report symmetry-relative rotation.
- The reliable working distance is `250–750 mm`.

Existing ArUco detection, camera recovery, calibration, cube tracking, and
debug windows remain active.

## Implementation Changes

### Ball detection and position

1. Add two persistent HSV profiles identified by `small_ball` and
   `large_ball`.
2. Convert each frame to HSV once, repair small segmentation gaps (including a
   zipper line), and extract dense outer contours for each enabled profile.
3. Reject border-touching, undersized, non-convex, poorly circular, and
   ambiguous overlapping-color candidates.
4. Undistort contour points, fit a robust ellipse/conic, and recover the sphere
   center from the calibrated tangent cone and configured physical radius.
5. Reproject the predicted sphere limb and validate edge error, IoU, area
   ratio, finite coordinates, positive depth, and the configured range.
6. Keep radii configurable because radius error produces proportional depth
   error.

### Tracking and handheld mode

- Keep an independent transactional tracker for each color identity. Color,
  not apparent image size, selects the radius.
- Smooth accepted 3D centers and apply velocity/depth gates. Hold the last
  trusted pose for `0.75 s`; a rejected candidate must never mutate trusted
  state.
- Add `HANDHELD_TRACKING_ENABLED = True` and an `H` runtime toggle.
- Always attempt normal full-silhouette detection first.
- Partial ball fallback is allowed only from an existing trusted track. It
  accepts a predicted-sphere-constrained visible boundary arc with at least
  `120°` support and low residual; weaker coverage holds the previous pose.
- Partial cube fallback remains local to the previous trusted pose and never
  globally acquires from a partial outline.

### Calibration, API, and UI

- On startup load `ball_color_profiles.npz`; a missing or invalid profile
  disables only that ball tracker.
- `1`/`2` selects the small/large profile. Clicking a solid-colored ball area
  samples a local HSV patch. `S` saves profiles and `R` resets the selected
  profile. Hue wraparound and neutral colors are handled explicitly.
- Ball APIs return `profile_id`, `radius_m`, `center_tvec`, projected contour,
  full/partial mode, fit metrics, and `orientation_observable=False`.
- Draw each ball's projected outline and center with separate status/radius/
  XYZ/range telemetry. Never draw axes or RPY for balls.
- Show `PARTIAL/HANDHELD`, `HELD/STALE`, and `LOST` clearly. Suppress generic
  `CIRCLE` overlays for contours claimed by a ball tracker.
- Share grayscale and HSV preprocessing where possible.

## Concrete defaults

The following conservative defaults are exposed in `pose6d/settings.py` so
they can be tuned without changing algorithms:

- radius: `0.020 m` / `0.0375 m`;
- range: `0.250 m` to `0.750 m`;
- stale hold: `0.75 s`;
- minimum partial arc: `120°`;
- full-silhouette edge error: `8 px` maximum;
- center speed: `1.0 m/s`; depth speed: `0.75 m/s`;
- sphere center smoothing time constant: `0.12 s`.

Iterative PnP for partial cube observations is always seeded from the previous
pose; three matched vertices are therefore a local refinement minimum, never a
fresh global acquisition.

## Verification

- Add deterministic synthetic calibrated projections for both radii at
  `250`, `500`, and `750 mm`, including off-axis spheres and lens distortion.
- Require noise-free center error within `1 mm`; with `1 px` boundary noise,
  require at least `95%` acceptance and accepted depth error within `10 mm`.
- Test simultaneous balls, profile persistence, hue wraparound, crossing
  paths, partial masks, stale holding, and identity-preserving reacquisition.
- Verify partial-ball updates with at least `120°` support and partial-cube
  updates with at least three matched vertices; weaker observations hold
  without changing trusted state.
- Preserve all existing cube, calibration, ArUco, compilation, and timeout
  tests.

## Assumptions

Both round objects are genuine spheres with distinct calibratable colors. The
zipper is a segmentation gap only. Featureless-sphere orientation and spin are
not observable. A configured-color flat disk is assumed not to be present as a
distractor.

## Implementation status

The plan is now represented by the modular `pose6d` implementation:

- `ball_detection.py` contains profile persistence/sampling, shared HSV masks,
  tangent-cone sphere recovery, reprojection gates, and partial-arc fallback.
- `tracking.py` contains independent transactional `BallTracker` instances;
  `cube_detection.py`/`cube_pose.py` contain trusted-track-only partial cube
  refinement.
- `app.py` integrates both balls, the cube, ArUco, shared preprocessing,
  profile controls, handheld toggle, telemetry, and debug masks.
- `tests/test_ball_tracking.py` and the existing cube suite cover geometry,
  persistence, hue wrapping, transaction/timeout behavior, and partial paths.

## Existing cube implementation

The existing cube model, solver, symmetry handling, transactional tracking,
camera recovery, and calibration remain in place. The details below are the
already-delivered cube contract and are retained for regression coverage.

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
- No new third-party dependencies are introduced; the ball modules and tests
  are intentional tracked additions.

---

## Recorded plan: cube false-tracking and ball calibration fix

This follow-up scope preserves the configured ball radii, 250--750 mm working
range, color identity, 0.75 s stale hold, ArUco behavior, and default-enabled
handheld mode.

### Ball calibration and diagnostics

- Keep circular hue estimation for chromatic profiles, but use one-sided
  illumination bounds: saturation fifth percentile minus 48 and value fifth
  percentile minus 64, with upper bounds fixed at 255. Neutral profiles keep
  bounded saturation/value ranges and ignore hue.
- Reset only the selected ball tracker after successful recalibration.
- Add a detailed per-profile result (candidate or `None`, best rejection reason,
  fit metrics, range reached by geometry, mask, and full/partial mode) while
  retaining existing wrappers and tuple-returning APIs.
- Report `TOO CLOSE`, `TOO FAR`, `OUTLINE REJECTED`, or `COLOR NOT FOUND`;
  display the selected profile and physical radius prominently. The Ball
  Segmentation window shows only the selected profile mask and its reason.

### Cube exclusion, physical edges, and partial tracking

- Pass the union of all enabled ball masks to cube detection as an optional
  exclusion mask, including rejected ball poses. Dilate it by three pixels and
  remove it before and after cube morphology.
- Require Canny support on full-candidate polygon edges (60% overall and 35%
  on every retained edge), and return a filtered debug mask with rejected dark
  pixels dim gray and surviving preliminary contours white.
- Keep normal `detect_cube` full-silhouette-only; the application may invoke
  the trusted local partial path once afterward.
- Replace partial nearest-corner matching with predicted-hull line matching:
  12° angle, 6 px perpendicular distance, 25% overlap, at least three matched
  edges and 35% perimeter coverage, with adjacent-edge intersections within
  10 px of predicted vertices. Seeded iterative PnP remains local, but matched
  line samples independently require at most 3 px RMS normal residual and the
  same orientation limit.
- Rejected partial observations never mutate trusted pose or timers.

### Verification targets

- Textured no-cube clutter produces zero partial accepts over distributed
  previous poses; corner-rich clutter without aligned edges expires from
  `HELD/STALE` to `LOST` within 0.75 s without trusted-state mutation.
- Valid partially occluded cubes refine only while aligned physical edges remain
  visible; removing edge support rejects the partial observation.
- Pink HSV spheres with highlight/shade variation recover their mask without
  admitting a differently hued soft shadow. A 20 mm ball at 160 mm reports
  `TOO CLOSE`, produces no trusted pose, and displays the valid range.
- Preserve hue-wrap, neutral-profile persistence, both radii at 250/500/750 mm,
  recalibration isolation, cube/ball/ArUco/calibration/compilation/timeout
  tests, and live acceptance behavior.

### Follow-up implementation status

Implemented in `pose6d/ball_detection.py`, `pose6d/cube_detection.py`,
`pose6d/cube_pose.py`, `pose6d/tracking.py`, `pose6d/app.py`, and
`pose6d/settings.py`; focused regression coverage is in the two test modules.
