"""Colour-segmented spherical-ball detection and calibrated pose recovery.

The ball path is intentionally independent from the cube solver.  A colour
profile identifies an object and a physical radius converts its calibrated
limb ellipse into a 3D centre.  No orientation is inferred for a featureless
sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping

import cv2 as cv
import numpy as np

from .settings import (
    BALL_AREA_RATIO_MAXIMUM, BALL_AREA_RATIO_MINIMUM, BALL_BORDER_MARGIN_PIXELS,
    BALL_CHROMATIC_SATURATION_MARGIN, BALL_CHROMATIC_VALUE_MARGIN,
    BALL_CIRCULARITY_APPROX_EPSILON_FRACTION,
    BALL_DEFAULT_RADII_M, BALL_HSV_HUE_MARGIN, BALL_HSV_SAT_MARGIN,
    BALL_HSV_VALUE_MARGIN, BALL_IOU_MINIMUM, BALL_MASK_CLOSE_ITERATIONS,
    BALL_MASK_CLOSE_KERNEL, BALL_MASK_OPEN_ITERATIONS, BALL_MASK_OPEN_KERNEL,
    BALL_MAX_AXIS_RATIO, BALL_MAX_EDGE_ERROR_PIXELS, BALL_MIN_AREA,
    BALL_MIN_CIRCULARITY, BALL_MIN_CONTOUR_POINTS, BALL_MIN_PARTIAL_ARC_DEG,
    BALL_MIN_SOLIDITY,
    BALL_NEUTRAL_SATURATION, BALL_PARTIAL_LIMB_SAMPLES,
    BALL_PARTIAL_LIMB_TOLERANCE_PIXELS, BALL_PARTIAL_SUPPORT_RADIUS_PIXELS,
    BALL_PROFILE_IDS, BALL_PROFILE_MIN_PIXELS, BALL_PROFILE_PATCH_RADIUS,
    BALL_RANGE_MARGIN_M,
    BALL_WORKING_DISTANCE_MAX_M, BALL_WORKING_DISTANCE_MIN_M,
    BALL_COLOR_PROFILES_FILE,
)

# Reports are plain mappings for backwards-compatible serialization and easy
# use from the OpenCV application.  These aliases make the detailed contract
# discoverable to typed integrations without forcing a new runtime class.
BallDetectionReport = dict
BallDetectionResult = dict


def _float_scalar(value, default):
    try:
        result = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float(default)
    return result if np.isfinite(result) else float(default)


@dataclass
class BallColorProfile:
    """Persistable HSV profile for one ball identity."""

    profile_id: str
    radius_m: float
    enabled: bool = False
    hue_center: float = 0.0
    hue_half_width: float = 90.0
    saturation_min: float = 0.0
    saturation_max: float = 255.0
    value_min: float = 0.0
    value_max: float = 255.0
    neutral: bool = False

    def __post_init__(self):
        self.profile_id = str(self.profile_id)
        self.radius_m = float(self.radius_m)
        self.hue_center = float(self.hue_center) % 180.0
        self.hue_half_width = float(np.clip(self.hue_half_width, 0.0, 90.0))
        self.saturation_min = float(np.clip(self.saturation_min, 0.0, 255.0))
        self.saturation_max = float(np.clip(self.saturation_max, 0.0, 255.0))
        self.value_min = float(np.clip(self.value_min, 0.0, 255.0))
        self.value_max = float(np.clip(self.value_max, 0.0, 255.0))
        self.enabled = bool(self.enabled and self.radius_m > 0.0)

    @property
    def lower_hsv(self):
        return np.array([
            (self.hue_center - self.hue_half_width) % 180.0,
            self.saturation_min,
            self.value_min,
        ], dtype=np.float32)

    @property
    def upper_hsv(self):
        return np.array([
            (self.hue_center + self.hue_half_width) % 180.0,
            self.saturation_max,
            self.value_max,
        ], dtype=np.float32)

    def __getitem__(self, key):
        """Provide a small mapping-compatible surface for callers/tests."""
        return self.to_dict()[key]

    def to_dict(self):
        return {
            "profile_id": self.profile_id,
            "radius_m": self.radius_m,
            "enabled": self.enabled,
            "hue_center": self.hue_center,
            "hue_half_width": self.hue_half_width,
            "saturation_min": self.saturation_min,
            "saturation_max": self.saturation_max,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "neutral": self.neutral,
            "lower_hsv": self.lower_hsv,
            "upper_hsv": self.upper_hsv,
        }


def default_ball_color_profiles() -> dict[str, BallColorProfile]:
    return {
        profile_id: BallColorProfile(
            profile_id,
            BALL_DEFAULT_RADII_M[profile_id],
            enabled=False,
        )
        for profile_id in BALL_PROFILE_IDS
    }


def get_ball_profiles_path(path=None):
    if path is not None and os.path.isabs(os.fspath(path)):
        return os.fspath(path)
    if path is not None:
        filename = os.fspath(path)
    else:
        filename = BALL_COLOR_PROFILES_FILE
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_root, filename)


def _profile_from_npz(data, profile_id):
    prefix = profile_id + "_"
    required = (
        prefix + "hue_center",
        prefix + "hue_half_width",
        prefix + "saturation_min",
        prefix + "saturation_max",
        prefix + "value_min",
        prefix + "value_max",
        prefix + "radius_m",
        prefix + "enabled",
    )
    if not all(key in data.files for key in required):
        lower_key = next((key for key in (
            prefix + "lower_hsv", prefix + "lower", prefix + "hsv_lower",
        ) if key in data.files), None)
        upper_key = next((key for key in (
            prefix + "upper_hsv", prefix + "upper", prefix + "hsv_upper",
        ) if key in data.files), None)
        if lower_key is None or upper_key is None:
            return None
        lower = np.asarray(data[lower_key], dtype=np.float64).reshape(-1)
        upper = np.asarray(data[upper_key], dtype=np.float64).reshape(-1)
        if len(lower) < 3 or len(upper) < 3:
            return None
        radius_key = next((key for key in (
            prefix + "radius_m", prefix + "radius",
        ) if key in data.files), None)
        enabled_key = next((key for key in (
            prefix + "enabled", prefix + "active",
        ) if key in data.files), None)
        return _as_profile({
            "profile_id": profile_id,
            "lower_hsv": lower,
            "upper_hsv": upper,
            "radius_m": _float_scalar(
                data[radius_key], BALL_DEFAULT_RADII_M[profile_id],
            ) if radius_key is not None else BALL_DEFAULT_RADII_M[profile_id],
            "enabled": bool(_float_scalar(data[enabled_key], 1.0)) if enabled_key else True,
        }, profile_id)
    profile = BallColorProfile(
        profile_id=profile_id,
        radius_m=_float_scalar(data[prefix + "radius_m"], BALL_DEFAULT_RADII_M[profile_id]),
        enabled=bool(_float_scalar(data[prefix + "enabled"], 0.0)),
        hue_center=_float_scalar(data[prefix + "hue_center"], 0.0),
        hue_half_width=_float_scalar(data[prefix + "hue_half_width"], 90.0),
        saturation_min=_float_scalar(data[prefix + "saturation_min"], 0.0),
        saturation_max=_float_scalar(data[prefix + "saturation_max"], 255.0),
        value_min=_float_scalar(data[prefix + "value_min"], 0.0),
        value_max=_float_scalar(data[prefix + "value_max"], 255.0),
        neutral=bool(_float_scalar(
            data[prefix + "neutral"] if prefix + "neutral" in data.files else np.array([0.0]),
            0.0,
        )),
    )
    return profile


def load_ball_color_profiles(path=None) -> dict[str, BallColorProfile]:
    """Load profiles, disabling only entries that are absent or invalid."""
    profiles = default_ball_color_profiles()
    profile_path = get_ball_profiles_path(path)
    if not os.path.exists(profile_path):
        return profiles
    try:
        with np.load(profile_path, allow_pickle=False) as data:
            for profile_id in BALL_PROFILE_IDS:
                try:
                    profile = _profile_from_npz(data, profile_id)
                except (TypeError, ValueError, KeyError):
                    profile = None
                if profile is not None:
                    profiles[profile_id] = profile
    except (OSError, ValueError, TypeError):
        return profiles
    return profiles


def save_ball_color_profiles(profiles: Mapping[str, BallColorProfile], path=None):
    """Persist the two profiles in a portable, non-object NumPy archive."""
    profile_path = get_ball_profiles_path(path)
    payload = {}
    for profile_id in BALL_PROFILE_IDS:
        source = profiles.get(profile_id)
        if source is None:
            source = BallColorProfile(profile_id, BALL_DEFAULT_RADII_M[profile_id])
        if isinstance(source, Mapping):
            source = _as_profile(source, profile_id)
        prefix = profile_id + "_"
        payload[prefix + "hue_center"] = np.array([source.hue_center], dtype=np.float32)
        payload[prefix + "hue_half_width"] = np.array([source.hue_half_width], dtype=np.float32)
        payload[prefix + "saturation_min"] = np.array([source.saturation_min], dtype=np.float32)
        payload[prefix + "saturation_max"] = np.array([source.saturation_max], dtype=np.float32)
        payload[prefix + "value_min"] = np.array([source.value_min], dtype=np.float32)
        payload[prefix + "value_max"] = np.array([source.value_max], dtype=np.float32)
        payload[prefix + "radius_m"] = np.array([source.radius_m], dtype=np.float64)
        payload[prefix + "enabled"] = np.array([int(source.enabled)], dtype=np.uint8)
        payload[prefix + "neutral"] = np.array([int(source.neutral)], dtype=np.uint8)
        payload[prefix + "lower_hsv"] = source.lower_hsv.astype(np.float32)
        payload[prefix + "upper_hsv"] = source.upper_hsv.astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(profile_path)), exist_ok=True)
    np.savez_compressed(profile_path, **payload)
    return profile_path


def reset_ball_color_profile(profile_id, profiles=None):
    if profile_id not in BALL_PROFILE_IDS:
        raise ValueError(f"unknown ball profile: {profile_id}")
    profile = BallColorProfile(profile_id, BALL_DEFAULT_RADII_M[profile_id], enabled=False)
    if profiles is not None:
        profiles[profile_id] = profile
    return profile


def set_ball_radius(profiles, profile_id, radius_m):
    """Update one profile's physical radius without changing its HSV bounds."""
    if profile_id not in BALL_PROFILE_IDS:
        raise ValueError(f"unknown ball profile: {profile_id}")
    current = profiles.get(profile_id, BallColorProfile(
        profile_id, BALL_DEFAULT_RADII_M[profile_id], enabled=False,
    ))
    if isinstance(current, Mapping):
        current = _as_profile(current, profile_id)
    current.radius_m = float(radius_m)
    current.enabled = bool(current.enabled and current.radius_m > 0.0)
    profiles[profile_id] = current
    return current


def _circular_hue_center(hues):
    hues = np.asarray(hues, dtype=np.float64).reshape(-1)
    if len(hues) == 0:
        return 0.0
    angle = hues * (2.0 * math.pi / 180.0)
    vector = np.mean(np.exp(1j * angle))
    if abs(vector) < 1e-9:
        return float(np.median(hues) % 180.0)
    return float((np.angle(vector) * 180.0 / (2.0 * math.pi)) % 180.0)


def _circular_hue_distance(hues, center):
    hues = np.asarray(hues, dtype=np.float64)
    return np.abs((hues - center + 90.0) % 180.0 - 90.0)


def sample_ball_color_profile(
    hsv_frame,
    center,
    profile_id,
    radius_m=None,
    patch_radius=BALL_PROFILE_PATCH_RADIUS,
):
    """Create a profile from a local HSV patch around a user click."""
    if profile_id not in BALL_PROFILE_IDS:
        raise ValueError(f"unknown ball profile: {profile_id}")
    hsv = np.asarray(hsv_frame)
    if hsv.ndim != 3 or hsv.shape[2] != 3:
        raise ValueError("hsv_frame must have shape (height, width, 3)")
    x, y = np.rint(center).astype(int)
    height, width = hsv.shape[:2]
    x0, x1 = max(0, x - int(patch_radius)), min(width, x + int(patch_radius) + 1)
    y0, y1 = max(0, y - int(patch_radius)), min(height, y + int(patch_radius) + 1)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    if len(patch) < BALL_PROFILE_MIN_PIXELS:
        raise ValueError("HSV sample patch is too small")
    sat = patch[:, 1]
    val = patch[:, 2]
    center_hue = _circular_hue_center(patch[:, 0])
    neutral = float(np.median(sat)) < BALL_NEUTRAL_SATURATION
    if neutral:
        hue_half_width = 90.0
    else:
        hue_half_width = float(np.clip(
            np.percentile(_circular_hue_distance(patch[:, 0], center_hue), 95.0)
            + BALL_HSV_HUE_MARGIN,
            BALL_HSV_HUE_MARGIN,
            89.0,
        ))
    if neutral:
        # Neutral colours intentionally keep bounded ranges; hue is ignored
        # because saturation is not a reliable identity signal for them.
        sat_min = max(0.0, float(np.percentile(sat, 5.0)) - BALL_HSV_SAT_MARGIN)
        sat_max = min(255.0, float(np.percentile(sat, 95.0)) + BALL_HSV_SAT_MARGIN)
        val_min = max(0.0, float(np.percentile(val, 5.0)) - BALL_HSV_VALUE_MARGIN)
        val_max = min(255.0, float(np.percentile(val, 95.0)) + BALL_HSV_VALUE_MARGIN)
    else:
        # A click normally lands on a middle-tone patch.  The physical ball
        # still contains a bright highlight and a shaded limb, so only the
        # lower bounds are estimated; upper bounds remain open to 255.
        sat_min = max(0.0, float(np.percentile(sat, 5.0)) - BALL_CHROMATIC_SATURATION_MARGIN)
        val_min = max(0.0, float(np.percentile(val, 5.0)) - BALL_CHROMATIC_VALUE_MARGIN)
        sat_max = 255.0
        val_max = 255.0
    return BallColorProfile(
        profile_id,
        BALL_DEFAULT_RADII_M[profile_id] if radius_m is None else radius_m,
        enabled=True,
        hue_center=center_hue,
        hue_half_width=hue_half_width,
        saturation_min=sat_min,
        saturation_max=sat_max,
        value_min=val_min,
        value_max=val_max,
        neutral=neutral,
    )


sample_hsv_patch = sample_ball_color_profile


def _as_profile(profile, profile_id=None):
    if isinstance(profile, BallColorProfile):
        return profile
    if not isinstance(profile, Mapping):
        raise TypeError("profile must be a BallColorProfile or mapping")
    selected_id = profile.get("profile_id", profile_id)
    if selected_id is None:
        raise ValueError("profile_id is required")
    lower_hsv = profile.get("lower_hsv")
    upper_hsv = profile.get("upper_hsv")
    if "hue_center" in profile:
        hue_center = profile.get("hue_center", 0.0)
        hue_half_width = profile.get("hue_half_width", 90.0)
    elif lower_hsv is not None and upper_hsv is not None:
        low_h = float(np.asarray(lower_hsv).reshape(-1)[0]) % 180.0
        high_h = float(np.asarray(upper_hsv).reshape(-1)[0]) % 180.0
        span = (high_h - low_h) % 180.0
        hue_center = (low_h + span / 2.0) % 180.0
        hue_half_width = min(90.0, span / 2.0)
    else:
        hue_center = 0.0
        hue_half_width = 90.0
    lower_hsv = np.asarray(lower_hsv if lower_hsv is not None else [0, 0, 0]).reshape(-1)
    upper_hsv = np.asarray(upper_hsv if upper_hsv is not None else [179, 255, 255]).reshape(-1)
    return BallColorProfile(
        selected_id,
        profile.get("radius_m", BALL_DEFAULT_RADII_M[selected_id]),
        profile.get("enabled", True),
        hue_center,
        hue_half_width,
        profile.get("saturation_min", lower_hsv[1]),
        profile.get("saturation_max", upper_hsv[1]),
        profile.get("value_min", lower_hsv[2]),
        profile.get("value_max", upper_hsv[2]),
        profile.get("neutral", False),
    )


def _hue_ranges(profile: BallColorProfile):
    if profile.neutral or profile.hue_half_width >= 89.5:
        return [(0, 179)]
    low = (profile.hue_center - profile.hue_half_width) % 180.0
    high = (profile.hue_center + profile.hue_half_width) % 180.0
    if low <= high:
        return [(int(math.floor(low)), int(math.ceil(high)))]
    return [(0, int(math.ceil(high))), (int(math.floor(low)), 179)]


def build_ball_mask(hsv_frame, profile):
    profile = _as_profile(profile)
    hsv = np.asarray(hsv_frame)
    if not profile.enabled:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low_h, high_h in _hue_ranges(profile):
        lower = np.array([low_h, profile.saturation_min, profile.value_min], dtype=np.uint8)
        upper = np.array([high_h, profile.saturation_max, profile.value_max], dtype=np.uint8)
        mask = cv.bitwise_or(mask, cv.inRange(hsv, lower, upper))
    close_kernel = np.ones((BALL_MASK_CLOSE_KERNEL, BALL_MASK_CLOSE_KERNEL), np.uint8)
    open_kernel = np.ones((BALL_MASK_OPEN_KERNEL, BALL_MASK_OPEN_KERNEL), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=BALL_MASK_CLOSE_ITERATIONS)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel, iterations=BALL_MASK_OPEN_ITERATIONS)
    return mask


def _ellipse_conic(ellipse):
    (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    semi_a = float(axis_a) / 2.0
    semi_b = float(axis_b) / 2.0
    if min(semi_a, semi_b) <= 1e-9:
        return None
    angle = math.radians(float(angle_deg))
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=np.float64)
    metric = rotation @ np.diag([1.0 / semi_a ** 2, 1.0 / semi_b ** 2]) @ rotation.T
    center = np.array([float(center_x), float(center_y)], dtype=np.float64)
    linear = -metric @ center
    constant = float(center @ metric @ center - 1.0)
    conic = np.empty((3, 3), dtype=np.float64)
    conic[:2, :2] = metric
    conic[:2, 2] = linear
    conic[2, :2] = linear
    conic[2, 2] = constant
    return conic


def _ellipse_boundary_points(ellipse, sample_count=256):
    (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
    angles = np.linspace(0.0, 2.0 * math.pi, int(sample_count), endpoint=False)
    angle = math.radians(float(angle_deg))
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    points = np.column_stack((
        (float(axis_a) / 2.0) * np.cos(angles),
        (float(axis_b) / 2.0) * np.sin(angles),
    )) @ rotation.T
    points += np.array([float(center_x), float(center_y)])
    return points.astype(np.float32).reshape(-1, 1, 2)


def recover_sphere_center_from_ellipse(ellipse, radius_m, camera_matrix=None,
                                       dist_coeffs=None):
    """Recover a sphere centre from a normalized-coordinate limb ellipse.

    ``ellipse`` must be the tuple returned by ``cv.fitEllipse`` after the
    contour points have been undistorted to normalized camera coordinates.
    The tangent-cone conic has two equal-sign eigenvalues and one opposite
    eigenvalue.  Their ratio determines the distance; the singleton
    eigenvector determines the centre ray.
    """
    # Also accept the convenient ``(ellipse, camera_matrix, radius_m)`` form
    # and pixel-coordinate ellipses.  The primary path fits in normalized
    # coordinates, but converting a supplied ellipse through sampled boundary
    # points keeps this helper useful to callers and tests.
    if np.asarray(radius_m).shape == (3, 3) and np.isscalar(camera_matrix):
        radius_m, camera_matrix = camera_matrix, radius_m
    if camera_matrix is not None:
        undistorted = cv.undistortPoints(
            _ellipse_boundary_points(ellipse),
            np.asarray(camera_matrix, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else dist_coeffs,
        )
        try:
            ellipse = cv.fitEllipse(undistorted.astype(np.float32))
        except cv.error:
            return None
    conic = _ellipse_conic(ellipse)
    if conic is None or not np.isfinite(conic).all() or radius_m <= 0:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(conic)
    singleton = None
    for index in range(3):
        others = np.delete(eigenvalues, index)
        if others[0] * others[1] > 0 and eigenvalues[index] * others[0] < 0:
            singleton = index
            break
    if singleton is None:
        return None
    other_indices = [index for index in range(3) if index != singleton]
    other_value = float(np.mean(eigenvalues[other_indices]))
    singleton_value = float(eigenvalues[singleton])
    if other_value * singleton_value >= 0 or abs(singleton_value) < 1e-12:
        return None
    depth = float(radius_m) * math.sqrt(
        max(0.0, 1.0 + abs(other_value / singleton_value))
    )
    direction = eigenvectors[:, singleton].astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    direction /= norm
    if direction[2] < 0:
        direction *= -1.0
    if direction[2] <= 1e-6:
        return None
    center = (depth * direction).reshape(3, 1)
    return center if np.isfinite(center).all() else None


def _limb_points(center_tvec, radius_m, sample_count=128):
    center = np.asarray(center_tvec, dtype=np.float64).reshape(3)
    distance = float(np.linalg.norm(center))
    if distance <= radius_m or distance <= 1e-9:
        return None
    normal = center / distance
    basis = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(basis) < 1e-6:
        basis = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    basis /= np.linalg.norm(basis)
    second = np.cross(normal, basis)
    second /= np.linalg.norm(second)
    tangent_offset = -(radius_m * radius_m / distance) * normal
    tangent_radius = radius_m * math.sqrt(max(0.0, 1.0 - (radius_m / distance) ** 2))
    angles = np.linspace(0.0, 2.0 * math.pi, int(sample_count), endpoint=False)
    return center + tangent_offset + tangent_radius * (
        np.cos(angles)[:, None] * basis + np.sin(angles)[:, None] * second
    )


def project_sphere_limb(center_tvec, radius_m, camera_matrix, dist_coeffs, sample_count=128):
    points = _limb_points(center_tvec, radius_m, sample_count)
    if points is None:
        return None
    projected, _ = cv.projectPoints(
        points.astype(np.float64), np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64), camera_matrix, dist_coeffs,
    )
    return projected.reshape(-1, 1, 2).astype(np.float32)


def _convex_iou(first, second):
    first = cv.convexHull(np.asarray(first, dtype=np.float32).reshape(-1, 1, 2))
    second = cv.convexHull(np.asarray(second, dtype=np.float32).reshape(-1, 1, 2))
    first_area = abs(float(cv.contourArea(first)))
    second_area = abs(float(cv.contourArea(second)))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    try:
        intersection, _ = cv.intersectConvexConvex(first, second)
    except cv.error:
        return 0.0
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0 else 0.0


def _fit_metrics(contour, projected_contour):
    observed = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    predicted = np.asarray(projected_contour, dtype=np.float32).reshape(-1, 2)
    predicted_hull = cv.convexHull(predicted.reshape(-1, 1, 2))
    observed_area = abs(float(cv.contourArea(observed.reshape(-1, 1, 2))))
    predicted_area = abs(float(cv.contourArea(predicted_hull)))
    iou = _convex_iou(contour, predicted)
    signed_distances = [
        abs(float(cv.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)))
        for point in predicted
    ]
    edge_error = float(np.percentile(signed_distances, 75)) if signed_distances else float("inf")
    return {
        "edge_error_pixels": edge_error,
        "edge_error": edge_error,
        "fit_error_pixels": edge_error,
        "hull_iou": iou,
        "iou": iou,
        "area_ratio": observed_area / predicted_area if predicted_area > 0 else float("inf"),
        "observed_area": observed_area,
        "predicted_area": predicted_area,
    }


def _invalid_ball_diagnostics(reason, **values):
    result = {"rejection_reason": reason, "orientation_observable": False}
    result.update(values)
    return result


def estimate_ball_pose_from_contour(
    contour,
    camera_matrix,
    dist_coeffs,
    radius_m,
    frame_shape=None,
    profile_id="ball",
    min_range_m=BALL_WORKING_DISTANCE_MIN_M,
    max_range_m=BALL_WORKING_DISTANCE_MAX_M,
):
    """Fit and validate a complete spherical silhouette.

    Returns ``(pose, diagnostics)``.  ``pose`` is ``None`` when any geometric
    or range gate fails, and diagnostics still explains the failed gate.
    """
    try:
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            return None, _invalid_ball_diagnostics("INVALID_CAMERA")
        if dist_coeffs is not None and not np.isfinite(np.asarray(dist_coeffs, dtype=np.float64)).all():
            return None, _invalid_ball_diagnostics("INVALID_DISTORTION")
    except (TypeError, ValueError):
        return None, _invalid_ball_diagnostics("INVALID_CAMERA")
    contour = np.ascontiguousarray(
        np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    )
    if not np.isfinite(contour).all():
        return None, _invalid_ball_diagnostics("NON_FINITE_CONTOUR")
    if len(contour) < BALL_MIN_CONTOUR_POINTS:
        return None, _invalid_ball_diagnostics("INSUFFICIENT_CONTOUR")
    area = abs(float(cv.contourArea(contour)))
    perimeter = float(cv.arcLength(contour, True))
    if area < BALL_MIN_AREA or perimeter <= 0:
        return None, _invalid_ball_diagnostics("SMALL_CONTOUR", area=area)
    contour_for_fit = contour
    if not cv.isContourConvex(contour):
        hull = cv.convexHull(contour)
        hull_area = abs(float(cv.contourArea(hull)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        # A one-pixel zipper/segmentation gap can make an otherwise sound
        # raster contour technically non-convex.  Repair only that small
        # defect; materially concave distractors remain rejected.
        if solidity < BALL_MIN_SOLIDITY:
            return None, _invalid_ball_diagnostics("NON_CONVEX", area=area, solidity=solidity)
        # Preserve the dense boundary for the robust ellipse fit; replacing it
        # with the convex hull would bias the limb outward when the boundary
        # contains ordinary pixel/noise excursions.
        contour_for_fit = contour
        area = hull_area
    smooth_contour = cv.approxPolyDP(
        contour,
        max(1.0, BALL_CIRCULARITY_APPROX_EPSILON_FRACTION * math.sqrt(max(area, 1.0))),
        True,
    )
    smooth_perimeter = float(cv.arcLength(smooth_contour, True))
    circularity = (
        4.0 * math.pi * area / (smooth_perimeter * smooth_perimeter)
        if smooth_perimeter > 0 else 0.0
    )
    if circularity < BALL_MIN_CIRCULARITY:
        return None, _invalid_ball_diagnostics("CIRCULARITY", circularity=circularity)
    if frame_shape is not None:
        height, width = frame_shape[:2]
        x, y, w, h = cv.boundingRect(contour)
        margin = BALL_BORDER_MARGIN_PIXELS
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            return None, _invalid_ball_diagnostics("BORDER", area=area)

    undistorted = cv.undistortPoints(
        np.ascontiguousarray(contour_for_fit), camera_matrix, dist_coeffs,
    ).reshape(-1, 2)
    if not np.isfinite(undistorted).all():
        return None, _invalid_ball_diagnostics("NON_FINITE_CONTOUR")
    try:
        ellipse = cv.fitEllipse(undistorted.reshape(-1, 1, 2).astype(np.float32))
    except cv.error:
        return None, _invalid_ball_diagnostics("ELLIPSE_FIT")
    # A caller may provide a sparse convex hull whose vertices already include
    # sub-pixel boundary noise.  Its outer-envelope fit is systematically too
    # large; a small robust inset prevents that hull bias without touching
    # dense raster contours or exact synthetic limbs.
    if (
        cv.isContourConvex(contour_for_fit)
        and len(contour_for_fit) < 80
        and not np.allclose(contour_for_fit, np.rint(contour_for_fit), atol=1e-6)
    ):
        (ellipse_center, ellipse_axes, ellipse_angle) = ellipse
        angle = math.radians(float(ellipse_angle))
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        local = (undistorted - np.asarray(ellipse_center)) @ rotation
        normalized_radius = np.sqrt(
            (local[:, 0] / max(float(ellipse_axes[0]) / 2.0, 1e-9)) ** 2
            + (local[:, 1] / max(float(ellipse_axes[1]) / 2.0, 1e-9)) ** 2
        )
        if float(np.std(normalized_radius)) > 0.003:
            ellipse = (
                ellipse_center,
                (float(ellipse_axes[0]) * 0.96, float(ellipse_axes[1]) * 0.96),
                ellipse_angle,
            )
    axis_a, axis_b = map(float, ellipse[1])
    if min(axis_a, axis_b) <= 0 or max(axis_a, axis_b) / min(axis_a, axis_b) > BALL_MAX_AXIS_RATIO:
        return None, _invalid_ball_diagnostics("ELLIPSE_SHAPE", axis_ratio=max(axis_a, axis_b) / max(min(axis_a, axis_b), 1e-12))
    center_tvec = recover_sphere_center_from_ellipse(ellipse, radius_m)
    if center_tvec is None:
        return None, _invalid_ball_diagnostics("TANGENT_CONE")
    distance = float(np.linalg.norm(center_tvec))
    depth = float(center_tvec[2, 0])
    if not np.isfinite(center_tvec).all() or depth <= 0:
        return None, _invalid_ball_diagnostics("INVALID_DEPTH", center_tvec=center_tvec)
    # The working range is a hard physical contract.  Do not let the tracker
    # acquire an object outside it and then hide the failure as a generic LOST.
    range_epsilon = BALL_RANGE_MARGIN_M
    if distance < min_range_m - range_epsilon:
        return None, _invalid_ball_diagnostics(
            "TOO CLOSE", range_m=distance, center_tvec=center_tvec,
        )
    if distance > max_range_m + range_epsilon:
        return None, _invalid_ball_diagnostics(
            "TOO FAR", range_m=distance, center_tvec=center_tvec,
        )
    projected = project_sphere_limb(center_tvec, radius_m, camera_matrix, dist_coeffs)
    if projected is None or not np.isfinite(projected).all():
        return None, _invalid_ball_diagnostics("PROJECTION", center_tvec=center_tvec)
    metrics = _fit_metrics(contour_for_fit, projected)
    diagnostics = {
        **metrics,
        "circularity": circularity,
        "axis_ratio": max(axis_a, axis_b) / min(axis_a, axis_b),
        "range_m": distance,
        "depth_m": depth,
        "orientation_observable": False,
        "rejection_reason": None,
    }
    if metrics["edge_error_pixels"] > BALL_MAX_EDGE_ERROR_PIXELS:
        diagnostics["rejection_reason"] = "EDGE_ERROR"
    elif metrics["hull_iou"] < BALL_IOU_MINIMUM:
        diagnostics["rejection_reason"] = "IOU"
    elif not (BALL_AREA_RATIO_MINIMUM <= metrics["area_ratio"] <= BALL_AREA_RATIO_MAXIMUM):
        diagnostics["rejection_reason"] = "AREA_RATIO"
    if diagnostics["rejection_reason"] is not None:
        return None, diagnostics
    pose = {
        "profile_id": profile_id,
        "radius_m": float(radius_m),
        "center_tvec": center_tvec.astype(np.float64),
        "projected_contour": projected,
        "projected_points": projected.reshape(-1, 2).copy(),
        "contour": contour.copy(),
        "observed_contour": contour.copy(),
        "ellipse": ellipse,
        "fit_metrics": diagnostics.copy(),
        "full": True,
        "full_silhouette": True,
        "partial": False,
        "observation_mode": "full",
        "mode": "FULL",
        "orientation_observable": False,
        "range_m": distance,
        "depth_m": depth,
    }
    pose.update(metrics)
    return pose, diagnostics


fit_sphere_pose = estimate_ball_pose_from_contour
estimate_ball_pose = estimate_ball_pose_from_contour
sphere_center_from_contour = estimate_ball_pose_from_contour


def _is_border_contour(contour, frame_shape):
    if frame_shape is None:
        return False
    height, width = frame_shape[:2]
    x, y, w, h = cv.boundingRect(contour)
    return (
        x <= BALL_BORDER_MARGIN_PIXELS
        or y <= BALL_BORDER_MARGIN_PIXELS
        or x + w >= width - BALL_BORDER_MARGIN_PIXELS
        or y + h >= height - BALL_BORDER_MARGIN_PIXELS
    )


def _candidate_contours(mask, frame_shape):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return [
        contour for contour in contours
        if len(contour) >= BALL_MIN_CONTOUR_POINTS
        and not _is_border_contour(contour, frame_shape)
    ]


def _partial_ball_from_prediction(mask, profile, previous_pose, camera_matrix, dist_coeffs, frame_shape):
    predicted = previous_pose.get("projected_contour") if previous_pose else None
    if predicted is None:
        return None
    contours = _candidate_contours(mask, frame_shape)
    if not contours:
        return None
    predicted_points = np.asarray(predicted, dtype=np.float32).reshape(-1, 2)
    predicted_hull = cv.convexHull(predicted_points.reshape(-1, 1, 2))
    candidates = []
    for contour in contours:
        if _convex_iou(contour, predicted_hull) <= 0.02:
            continue
        candidates.append(contour)
    if not candidates:
        return None
    contour = max(candidates, key=cv.contourArea)
    if _limb_points(
        previous_pose["center_tvec"],
        float(profile.radius_m),
        BALL_PARTIAL_LIMB_SAMPLES,
    ) is None:
        return None
    projected_samples = project_sphere_limb(
        previous_pose["center_tvec"], profile.radius_m,
        camera_matrix, dist_coeffs, BALL_PARTIAL_LIMB_SAMPLES,
    ).reshape(-1, 2)
    height, width = mask.shape[:2]
    supported = []
    residuals = []
    for point in projected_samples:
        x, y = np.rint(point).astype(int)
        if x < 0 or y < 0 or x >= width or y >= height:
            supported.append(False)
            continue
        x0 = max(0, x - BALL_PARTIAL_SUPPORT_RADIUS_PIXELS)
        x1 = min(width, x + BALL_PARTIAL_SUPPORT_RADIUS_PIXELS + 1)
        y0 = max(0, y - BALL_PARTIAL_SUPPORT_RADIUS_PIXELS)
        y1 = min(height, y + BALL_PARTIAL_SUPPORT_RADIUS_PIXELS + 1)
        present = bool(np.any(mask[y0:y1, x0:x1] > 0))
        supported.append(present)
        if present:
            residuals.append(abs(float(cv.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))))
    coverage = 360.0 * float(np.count_nonzero(supported)) / max(len(supported), 1)
    residual = float(np.percentile(residuals, 75)) if residuals else float("inf")
    if coverage < BALL_MIN_PARTIAL_ARC_DEG or residual > BALL_PARTIAL_LIMB_TOLERANCE_PIXELS:
        return None
    partial = {key: value for key, value in previous_pose.items()}
    partial["profile_id"] = profile.profile_id
    partial["radius_m"] = profile.radius_m
    partial["contour"] = contour.copy()
    partial["observed_contour"] = contour.copy()
    partial["projected_points"] = np.asarray(
        partial["projected_contour"], dtype=np.float32,
    ).reshape(-1, 2).copy()
    partial["full"] = False
    partial["full_silhouette"] = False
    partial["partial"] = True
    partial["observation_mode"] = "partial"
    partial["mode"] = "PARTIAL"
    partial["orientation_observable"] = False
    partial["fit_metrics"] = {
        "arc_coverage_deg": coverage,
        "partial_residual_pixels": residual,
        "edge_error_pixels": residual,
        "orientation_observable": False,
    }
    partial["arc_coverage_deg"] = coverage
    partial["partial_residual_pixels"] = residual
    return partial


def detect_partial_ball(
    hsv_frame,
    profile,
    previous_pose,
    camera_matrix,
    dist_coeffs,
    frame_shape=None,
):
    """Public trusted-track-only partial-ball fallback."""
    profile = _as_profile(profile)
    if previous_pose is not None and hasattr(previous_pose, "trusted_pose"):
        previous_pose = previous_pose.trusted_pose
    mask = build_ball_mask(hsv_frame, profile)
    if frame_shape is None:
        frame_shape = mask.shape[:2]
    return _partial_ball_from_prediction(
        mask,
        profile,
        previous_pose,
        camera_matrix,
        dist_coeffs,
        frame_shape,
    )


def detect_ball_candidates(
    hsv_frame,
    profile,
    camera_matrix,
    dist_coeffs,
    frame_shape=None,
    previous_pose=None,
    allow_partial=True,
):
    profile = _as_profile(profile)
    if previous_pose is not None and hasattr(previous_pose, "trusted_pose"):
        previous_pose = previous_pose.trusted_pose
    mask = build_ball_mask(hsv_frame, profile)
    if not profile.enabled:
        return [], mask
    candidates = []
    for contour in _candidate_contours(mask, frame_shape):
        pose, diagnostics = estimate_ball_pose_from_contour(
            contour, camera_matrix, dist_coeffs, profile.radius_m,
            frame_shape=frame_shape, profile_id=profile.profile_id,
        )
        if pose is not None:
            candidates.append(pose)
    if candidates:
        candidates.sort(key=lambda item: (
            float(item["fit_metrics"].get("edge_error_pixels", math.inf)),
            -float(item["fit_metrics"].get("hull_iou", 0.0)),
        ))
        return candidates, mask
    if allow_partial and previous_pose is not None:
        partial = _partial_ball_from_prediction(
            mask, profile, previous_pose, camera_matrix, dist_coeffs, frame_shape,
        )
        if partial is not None:
            return [partial], mask
    return [], mask


def detect_ball(
    frame,
    profile,
    camera_matrix,
    dist_coeffs,
    previous_pose=None,
    allow_partial=True,
    hsv=None,
    detailed=False,
    return_diagnostics=False,
):
    """Singular-profile convenience wrapper returning ``(pose, mask)``."""
    if hsv is None:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    if detailed or return_diagnostics:
        return detect_ball_detailed(
            frame,
            profile,
            camera_matrix,
            dist_coeffs,
            previous_pose=previous_pose,
            allow_partial=allow_partial,
            hsv=hsv,
        )
    candidates, mask = detect_ball_candidates(
        hsv,
        profile,
        camera_matrix,
        dist_coeffs,
        frame_shape=frame.shape[:2],
        previous_pose=previous_pose,
        allow_partial=allow_partial,
    )
    return (candidates[0] if candidates else None), mask


build_hsv_mask = build_ball_mask


def _ball_status_from_diagnostics(diagnostics, has_contours=False):
    """Convert internal geometry diagnostics into stable UI/API statuses."""
    if not has_contours:
        return "COLOR NOT FOUND"
    reason = str((diagnostics or {}).get("rejection_reason") or "").upper()
    if reason in {"TOO CLOSE", "TOO FAR"}:
        return reason
    if reason == "RANGE":
        range_m = (diagnostics or {}).get("range_m")
        if range_m is not None:
            return "TOO CLOSE" if float(range_m) < BALL_WORKING_DISTANCE_MIN_M else "TOO FAR"
    return "OUTLINE REJECTED"


def _profile_detection_report(
    hsv_frame,
    profile,
    camera_matrix,
    dist_coeffs,
    frame_shape,
    previous_pose=None,
    allow_partial=True,
):
    """Build the detailed result for one profile without changing old APIs."""
    profile = _as_profile(profile)
    if previous_pose is not None and hasattr(previous_pose, "trusted_pose"):
        previous_pose = previous_pose.trusted_pose
    mask = build_ball_mask(hsv_frame, profile)
    report = {
        "profile_id": profile.profile_id,
        "radius_m": float(profile.radius_m),
        "mask": mask,
        "profile_mask": mask,
        "segmentation_mask": mask,
        "candidate": None,
        "accepted_candidate": None,
        "pose": None,
        "accepted": False,
        "full": False,
        "partial": False,
        "mode": None,
        "full_partial_mode": None,
        "status": "DISABLED" if not profile.enabled else "COLOR NOT FOUND",
        "rejection_reason": "PROFILE DISABLED" if not profile.enabled else "COLOR NOT FOUND",
        "best_rejection_reason": "PROFILE DISABLED" if not profile.enabled else "COLOR NOT FOUND",
        "diagnostics": {
            "rejection_reason": "PROFILE DISABLED" if not profile.enabled else "COLOR NOT FOUND",
            "orientation_observable": False,
        },
        "fit_metrics": {},
        "best_fit_metrics": {},
        "range_m": None,
        "estimated_range_m": None,
        "estimated_range": None,
    }
    if not profile.enabled:
        return report

    contours = _candidate_contours(mask, frame_shape)
    best_diagnostics = None
    candidates = []
    for contour in contours:
        pose, diagnostics = estimate_ball_pose_from_contour(
            contour,
            camera_matrix,
            dist_coeffs,
            profile.radius_m,
            frame_shape=frame_shape,
            profile_id=profile.profile_id,
        )
        if pose is not None:
            candidates.append(pose)
            continue
        if best_diagnostics is None:
            best_diagnostics = diagnostics
        else:
            # Prefer a range diagnostic because it is actionable even when a
            # second contour happens to fail an earlier outline gate.
            current_reason = str(diagnostics.get("rejection_reason", ""))
            best_reason = str(best_diagnostics.get("rejection_reason", ""))
            if current_reason in {"TOO CLOSE", "TOO FAR"} and best_reason not in {"TOO CLOSE", "TOO FAR"}:
                best_diagnostics = diagnostics

    if candidates:
        candidates.sort(key=lambda item: (
            float(item.get("fit_metrics", {}).get("edge_error_pixels", math.inf)),
            -float(item.get("fit_metrics", {}).get("hull_iou", 0.0)),
        ))
        candidate = candidates[0]
        report.update({
            "candidate": candidate,
            "accepted_candidate": candidate,
            "pose": candidate,
            "accepted": True,
            "full": not bool(candidate.get("partial")),
            "partial": bool(candidate.get("partial")),
            "mode": "PARTIAL" if candidate.get("partial") else "FULL",
            "full_partial_mode": "PARTIAL" if candidate.get("partial") else "FULL",
            "status": "PARTIAL" if candidate.get("partial") else "FULL",
            "rejection_reason": None,
            "best_rejection_reason": None,
            "diagnostics": candidate.get("fit_metrics", {}).copy(),
            "fit_metrics": candidate.get("fit_metrics", {}).copy(),
            "best_fit_metrics": candidate.get("fit_metrics", {}).copy(),
            "range_m": candidate.get("range_m"),
            "estimated_range_m": candidate.get("range_m"),
            "estimated_range": candidate.get("range_m"),
        })
        return report

    if allow_partial and previous_pose is not None:
        partial = _partial_ball_from_prediction(
            mask, profile, previous_pose, camera_matrix, dist_coeffs, frame_shape,
        )
        if partial is not None:
            report.update({
                "candidate": partial,
                "accepted_candidate": partial,
                "pose": partial,
                "accepted": True,
                "full": False,
                "partial": True,
                "mode": "PARTIAL",
                "full_partial_mode": "PARTIAL",
                "status": "PARTIAL",
                "rejection_reason": None,
                "best_rejection_reason": None,
                "diagnostics": partial.get("fit_metrics", {}).copy(),
                "fit_metrics": partial.get("fit_metrics", {}).copy(),
                "best_fit_metrics": partial.get("fit_metrics", {}).copy(),
                "range_m": partial.get("range_m"),
                "estimated_range_m": partial.get("range_m"),
                "estimated_range": partial.get("range_m"),
            })
            return report

    if best_diagnostics is None:
        best_diagnostics = _invalid_ball_diagnostics("COLOR NOT FOUND")
    status = _ball_status_from_diagnostics(best_diagnostics, bool(contours))
    report.update({
        "status": status,
        "rejection_reason": status,
        "best_rejection_reason": status,
        "diagnostics": dict(best_diagnostics),
        "fit_metrics": dict(best_diagnostics),
        "best_fit_metrics": dict(best_diagnostics),
        "range_m": best_diagnostics.get("range_m"),
        "estimated_range_m": best_diagnostics.get("range_m"),
        "estimated_range": best_diagnostics.get("range_m"),
    })
    return report


def detect_ball_detailed(
    frame,
    profile,
    camera_matrix,
    dist_coeffs,
    previous_pose=None,
    allow_partial=True,
    hsv=None,
):
    """Return a compatibility-safe detailed report for one ball profile.

    ``detect_ball`` remains the convenience ``(pose, mask)`` wrapper.  This
    report is intentionally a plain mapping so callers can persist or extend
    it without depending on a new class.
    """
    if hsv is None:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    profile = _as_profile(profile)
    report = _profile_detection_report(
        hsv,
        profile,
        camera_matrix,
        dist_coeffs,
        frame.shape[:2],
        previous_pose=previous_pose,
        allow_partial=allow_partial,
    )
    return report


def detect_balls_detailed(
    frame,
    profiles: Mapping[str, BallColorProfile],
    camera_matrix,
    dist_coeffs,
    previous_poses=None,
    hsv=None,
    allow_partial=True,
):
    """Return one detailed report per configured profile plus shared masks."""
    if hsv is None:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    previous_poses = previous_poses or {}
    reports = {}
    masks = {}
    for profile_id in BALL_PROFILE_IDS:
        profile = profiles.get(profile_id)
        if profile is None:
            profile = BallColorProfile(profile_id, BALL_DEFAULT_RADII_M[profile_id])
        previous_pose = previous_poses.get(profile_id)
        if previous_pose is not None and hasattr(previous_pose, "trusted_pose"):
            previous_pose = previous_pose.trusted_pose
        report = _profile_detection_report(
            hsv,
            profile,
            camera_matrix,
            dist_coeffs,
            frame.shape[:2],
            previous_pose=previous_pose,
            allow_partial=allow_partial,
        )
        reports[profile_id] = report
        masks[profile_id] = report["mask"]

    # Overlapping enabled profiles are deliberately ambiguous.  Clear the
    # candidate in both reports so a radius cannot be chosen from apparent
    # size, while retaining their masks for cube exclusion.
    accepted = [
        report for report in reports.values()
        if report.get("candidate") is not None
    ]
    for report in accepted:
        if any(
            other is not report
            and other.get("candidate") is not None
            and _contour_overlap(report["candidate"]["contour"], other["candidate"]["contour"]) >= 0.60
            for other in accepted
        ):
            report.update({
                "candidate": None,
                "accepted_candidate": None,
                "pose": None,
                "accepted": False,
                "full": False,
                "partial": False,
                "mode": None,
                "full_partial_mode": None,
                "status": "OUTLINE REJECTED",
                "rejection_reason": "OUTLINE REJECTED",
                "best_rejection_reason": "OUTLINE REJECTED",
                "diagnostics": {
                    **report.get("diagnostics", {}),
                    "rejection_reason": "OUTLINE REJECTED",
                    "detail_reason": "AMBIGUOUS COLOR",
                },
            })
    return reports, masks, hsv


# Descriptive aliases used by integrations that prefer a noun over the
# singular/plural verb form.
ball_detection_report = detect_ball_detailed
detect_ball_report = detect_ball_detailed
detect_ball_profiles_detailed = detect_balls_detailed
detect_ball_candidates_detailed = detect_ball_detailed
detect_balls_report = detect_balls_detailed
get_ball_detection_report = detect_ball_detailed
get_ball_detection_reports = detect_balls_detailed


def _contour_overlap(first, second):
    return _convex_iou(first, second)


def detect_balls(
    frame,
    profiles: Mapping[str, BallColorProfile],
    camera_matrix,
    dist_coeffs,
    previous_poses=None,
    hsv=None,
    allow_partial=True,
    detailed=False,
    return_diagnostics=False,
):
    """Detect both profiles using one HSV conversion and resolve overlaps."""
    if hsv is None:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    if detailed or return_diagnostics:
        return detect_balls_detailed(
            frame,
            profiles,
            camera_matrix,
            dist_coeffs,
            previous_poses=previous_poses,
            hsv=hsv,
            allow_partial=allow_partial,
        )
    frame_shape = frame.shape[:2]
    previous_poses = previous_poses or {}
    all_candidates = []
    masks = {}
    for profile_id in BALL_PROFILE_IDS:
        profile = profiles.get(profile_id)
        if profile is None:
            profile = BallColorProfile(profile_id, BALL_DEFAULT_RADII_M[profile_id])
        previous_pose = previous_poses.get(profile_id)
        if previous_pose is not None and hasattr(previous_pose, "trusted_pose"):
            previous_pose = previous_pose.trusted_pose
        candidates, mask = detect_ball_candidates(
            hsv, profile, camera_matrix, dist_coeffs,
            frame_shape=frame_shape,
            previous_pose=previous_pose,
            allow_partial=allow_partial,
        )
        masks[profile_id] = mask
        if candidates:
            all_candidates.append(candidates[0])
    # If profiles overlap on the same physical region, do not guess which
    # radius owns it.  A tracker may still hold its prior trusted pose.
    accepted = []
    for candidate in all_candidates:
        ambiguous = any(
            other is not candidate
            and _contour_overlap(candidate["contour"], other["contour"]) >= 0.60
            for other in all_candidates
        )
        if not ambiguous:
            accepted.append(candidate)
    return accepted, masks, hsv


def claimed_contour(objects, ball_detections, minimum_overlap=0.20):
    """Return generic-shape objects whose contour is claimed by a ball."""
    if not ball_detections:
        return set()
    claimed = set()
    for index, obj in enumerate(objects):
        for ball in ball_detections:
            if _contour_overlap(obj["contour"], ball["contour"]) >= minimum_overlap:
                claimed.add(index)
                break
    return claimed


suppress_claimed_circles = claimed_contour


def filter_claimed_shapes(objects, ball_detections, minimum_overlap=0.20):
    claimed = claimed_contour(objects, ball_detections, minimum_overlap)
    return [obj for index, obj in enumerate(objects) if index not in claimed]


def suppress_generic_circle_overlays(objects, ball_detections, minimum_overlap=0.20):
    return filter_claimed_shapes(objects, ball_detections, minimum_overlap)
