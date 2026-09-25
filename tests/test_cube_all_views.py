"""Deterministic synthetic checks for the unmarked all-view cube tracker."""

import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pose6d.calibration import load_camera_calibration  # noqa: E402
from pose6d.cube_detection import detect_cube  # noqa: E402
from pose6d.cube_geometry import cube_object_points  # noqa: E402
from pose6d.cube_pose import estimate_cube_pose_from_detection  # noqa: E402
from pose6d.tracking import CubeTracker  # noqa: E402


class CubeAllViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera_matrix, cls.dist_coeffs, _ = load_camera_calibration(1280, 720)

    def render(self, rvec, depth=0.35):
        tvec = np.asarray([[0.0], [0.0], [depth]], dtype=np.float64)
        projected, _ = cv.projectPoints(
            cube_object_points(), np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec, self.camera_matrix, self.dist_coeffs,
        )
        hull = cv.convexHull(projected.astype(np.float32)).reshape(-1, 2)
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv.fillConvexPoly(frame, np.rint(hull).astype(np.int32), (20, 20, 20))
        return frame

    def test_face_transition_and_multi_face_views(self):
        for degrees in (0, 5, 10, 20, 45, 90):
            detection, _ = detect_cube(
                self.render([np.deg2rad(degrees), 0, 0])
            )
            self.assertIsNotNone(detection)
            self.assertTrue({len(item["points"]) for item in detection["point_hypotheses"]} & {4, 5, 6})
            pose, diagnostics = estimate_cube_pose_from_detection(
                detection, self.camera_matrix, self.dist_coeffs
            )
            self.assertIsNotNone(pose, diagnostics)
            self.assertGreater(float(pose["center_tvec"][2, 0]), 0.0)

    def test_tracker_keeps_face_on_orientation_ambiguous(self):
        tracker = CubeTracker()
        for index, degrees in enumerate((0, 2, 4, 6, 8, 10)):
            detection, _ = detect_cube(
                self.render([0, 0, np.deg2rad(degrees)]),
                tracker.reference_pose,
            )
            result = tracker.update(
                detection, self.camera_matrix, self.dist_coeffs,
                100.0 + index / 30.0,
            )
            self.assertTrue(result["accepted"], result)
            self.assertTrue(result["pose"]["orientation_ambiguous"])

    def test_timeout_resets_global_reference(self):
        tracker = CubeTracker(hold_seconds=0.75)
        detection, _ = detect_cube(self.render([0, 0, 0]))
        first = tracker.update(detection, self.camera_matrix, self.dist_coeffs, 10.0)
        self.assertTrue(first["accepted"])
        held = tracker.update(None, self.camera_matrix, self.dist_coeffs, 10.2)
        self.assertTrue(held["held"])
        expired = tracker.update(None, self.camera_matrix, self.dist_coeffs, 11.0)
        self.assertIsNone(expired["pose"])
        self.assertIsNone(tracker.raw_pose)


if __name__ == "__main__":
    unittest.main()
