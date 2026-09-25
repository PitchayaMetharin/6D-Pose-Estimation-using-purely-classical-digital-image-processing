"""Deterministic checks for dual spherical-ball geometry and transactions."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pose6d.ball_detection import (  # noqa: E402
    BallColorProfile,
    build_ball_mask,
    default_ball_color_profiles,
    detect_partial_ball,
    detect_ball_detailed,
    estimate_ball_pose_from_contour,
    load_ball_color_profiles,
    project_sphere_limb,
    sample_ball_color_profile,
    save_ball_color_profiles,
)
from pose6d.tracking import BallTracker  # noqa: E402


class BallTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera_matrix = np.array([
            [900.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0],
        ])
        cls.dist_coeffs = np.array([0.08, -0.02, 0.001, 0.001, 0.0])

    def test_noise_free_centres_for_both_radii(self):
        for radius in (0.020, 0.0375):
            for depth in (0.250, 0.500, 0.750):
                expected = np.array([[0.035], [-0.025], [depth]])
                contour = project_sphere_limb(
                    expected, radius, self.camera_matrix, self.dist_coeffs, 512,
                )
                pose, diagnostics = estimate_ball_pose_from_contour(
                    contour,
                    self.camera_matrix,
                    self.dist_coeffs,
                    radius,
                    (720, 1280),
                    "small_ball",
                )
                self.assertIsNotNone(pose, diagnostics)
                self.assertLess(
                    float(np.linalg.norm(pose["center_tvec"] - expected)),
                    0.001,
                )
                self.assertFalse(pose["orientation_observable"])

    def test_profiles_persist_and_hue_wraps(self):
        hsv = np.zeros((30, 30, 3), dtype=np.uint8)
        hsv[:] = (178, 220, 220)
        hsv[10:20, 10:20] = (2, 220, 220)
        profile = sample_ball_color_profile(hsv, (15, 15), "small_ball")
        mask = build_ball_mask(hsv, profile)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertTrue(profile.enabled)

        profiles = default_ball_color_profiles()
        profiles["small_ball"] = profile
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profiles.npz")
            save_ball_color_profiles(profiles, path)
            loaded = load_ball_color_profiles(path)
        self.assertTrue(loaded["small_ball"].enabled)
        self.assertAlmostEqual(loaded["small_ball"].radius_m, 0.020)
        self.assertFalse(loaded["large_ball"].enabled)

    def test_chromatic_calibration_keeps_illumination_limb(self):
        hsv = np.zeros((30, 30, 3), dtype=np.uint8)
        values = np.linspace(90, 240, 30, dtype=np.uint8)
        for column, value in enumerate(values):
            hsv[:, column] = (12, 170, value)
        profile = sample_ball_color_profile(hsv, (15, 15), "small_ball")
        self.assertEqual(profile.saturation_max, 255.0)
        self.assertEqual(profile.value_max, 255.0)
        self.assertAlmostEqual(profile.saturation_min, 122.0, delta=1.0)
        self.assertAlmostEqual(profile.value_min, 72.0, delta=1.0)

    def test_detailed_report_preserves_range_reason(self):
        centre = np.array([[0.0], [0.0], [0.16]])
        limb = project_sphere_limb(
            centre, 0.020, self.camera_matrix, self.dist_coeffs, 512,
        )
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv.fillConvexPoly(
            frame,
            cv.convexHull(np.rint(limb).astype(np.int32)),
            (0, 0, 255),
        )
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        profile = sample_ball_color_profile(hsv, (640, 360), "small_ball")
        report = detect_ball_detailed(
            frame, profile, self.camera_matrix, self.dist_coeffs, hsv=hsv,
        )
        self.assertIsNone(report["candidate"])
        self.assertEqual(report["status"], "TOO CLOSE")
        self.assertIsNotNone(report["estimated_range_m"])

    def test_rejected_pose_does_not_mutate_and_then_expires(self):
        expected = np.array([[0.0], [0.0], [0.5]])
        contour = project_sphere_limb(
            expected, 0.020, self.camera_matrix, self.dist_coeffs, 256,
        )
        pose, diagnostics = estimate_ball_pose_from_contour(
            contour, self.camera_matrix, self.dist_coeffs, 0.020,
            (720, 1280), "small_ball",
        )
        self.assertIsNotNone(pose, diagnostics)
        tracker = BallTracker("small_ball", 0.020)
        first = tracker.update(pose, self.camera_matrix, self.dist_coeffs, 1.0)
        self.assertTrue(first["accepted"])
        trusted = tracker.raw_pose["center_tvec"].copy()

        rejected = dict(pose)
        rejected["center_tvec"] = np.array([[0.0], [0.0], [0.1]])
        rejected_result = tracker.update(
            rejected, self.camera_matrix, self.dist_coeffs, 1.1,
        )
        self.assertTrue(rejected_result["held"])
        np.testing.assert_array_equal(tracker.raw_pose["center_tvec"], trusted)

        expired = tracker.update(None, self.camera_matrix, self.dist_coeffs, 1.9)
        self.assertEqual(expired["status"], "LOST")
        self.assertIsNone(tracker.raw_pose)

    def test_partial_observation_requires_existing_track(self):
        profile = BallColorProfile("small_ball", 0.020, enabled=True)
        tracker = BallTracker("small_ball", 0.020)
        pose = {
            "profile_id": "small_ball",
            "radius_m": 0.020,
            "center_tvec": np.array([[0.0], [0.0], [0.5]]),
            "projected_contour": np.zeros((8, 1, 2), dtype=np.float32),
            "full": False,
            "partial": True,
            "arc_coverage_deg": 150.0,
            "partial_residual_pixels": 2.0,
            "orientation_observable": False,
        }
        result = tracker.update(pose, self.camera_matrix, self.dist_coeffs, 1.0)
        self.assertEqual(result["status"], "LOST")
        self.assertIsNone(tracker.raw_pose)

    def test_partial_ball_arc_uses_existing_prediction(self):
        expected = np.array([[0.0], [0.0], [0.5]])
        limb = project_sphere_limb(
            expected, 0.020, self.camera_matrix, self.dist_coeffs, 512,
        )
        full = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv.fillConvexPoly(
            full,
            cv.convexHull(np.rint(limb).astype(np.int32)),
            (0, 0, 255),
        )
        hsv = cv.cvtColor(full, cv.COLOR_BGR2HSV)
        profile = sample_ball_color_profile(hsv, (640, 360), "small_ball")
        pose, diagnostics = estimate_ball_pose_from_contour(
            limb, self.camera_matrix, self.dist_coeffs, 0.020,
            (720, 1280), "small_ball",
        )
        self.assertIsNotNone(pose, diagnostics)
        occluded = full.copy()
        occluded[:, :640] = 220
        partial = detect_partial_ball(
            cv.cvtColor(occluded, cv.COLOR_BGR2HSV),
            profile,
            pose,
            self.camera_matrix,
            self.dist_coeffs,
            (720, 1280),
        )
        self.assertIsNotNone(partial)
        self.assertGreaterEqual(partial["arc_coverage_deg"], 120.0)


if __name__ == "__main__":
    unittest.main()
