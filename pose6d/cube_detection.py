"""Dark-cube segmentation, multi-resolution corner hypotheses, and tracking."""

import math

import cv2 as cv
import numpy as np

from .settings import (
    CUBE_APPROX_EPSILON_FRACTIONS, CUBE_DARK_THRESHOLD,
    CUBE_MAX_TRACK_ERROR_PIXELS, CUBE_MIN_AREA,
)


def canonicalize_cube_points(points):
    """Return a cyclic point list with a stable top-most starting vertex."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points.copy()
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = min(range(len(ordered)), key=lambda index: (ordered[index][1], ordered[index][0]))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def align_cube_points(points, reference_points):
    """Align cyclic lists when their point counts match."""

    points = canonicalize_cube_points(points)
    reference_points = canonicalize_cube_points(reference_points)
    if len(points) != len(reference_points) or len(points) == 0:
        return points.astype(np.float32), float("inf")

    best_points = points.copy()
    best_error = float("inf")
    for ordered_points in (points, points[::-1]):
        for shift in range(len(points)):
            candidate = np.roll(ordered_points, shift, axis=0)
            error = float(np.mean(np.linalg.norm(candidate - reference_points, axis=1)))
            if error < best_error:
                best_points = candidate.copy()
                best_error = error
    return best_points.astype(np.float32), best_error


def refine_cube_points(gray, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.01)
    try:
        refined = cv.cornerSubPix(gray, points, (5, 5), (-1, -1), criteria)
    except cv.error:
        return points.reshape(-1, 2).astype(np.float32)
    return refined.reshape(-1, 2).astype(np.float32)


def fit_cube_edge_line(contour_points, start, end, diagonal):
    """Fit one polygon edge from dense contour samples with a Huber loss."""

    edge = end - start
    edge_length_squared = float(np.dot(edge, edge))
    if edge_length_squared <= 0:
        return None
    offsets = contour_points - start
    fractions = offsets @ edge / edge_length_squared
    distances = np.abs(edge[0] * offsets[:, 1] - edge[1] * offsets[:, 0]) / math.sqrt(edge_length_squared)
    distance_limit = max(2.0, 0.015 * diagonal)
    samples = contour_points[
        (fractions >= 0.10) & (fractions <= 0.90) & (distances <= distance_limit)
    ]
    if len(samples) < 6:
        return None
    try:
        line = cv.fitLine(samples.reshape(-1, 1, 2), cv.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
    except cv.error:
        return None
    direction = np.asarray(line[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0 or not np.isfinite(line).all():
        return None
    return direction / norm, np.asarray(line[2:], dtype=np.float64)


def line_fitted_cube_points(contour, polygon_points):
    """Refine vertices by intersecting robust adjacent edge fits."""

    contour_points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    polygon_points = np.asarray(polygon_points, dtype=np.float32).reshape(-1, 2)
    if len(polygon_points) < 4:
        return None
    diagonal = float(np.linalg.norm(np.ptp(polygon_points, axis=0)))
    if diagonal <= 0:
        return None

    lines = []
    for index, start in enumerate(polygon_points):
        line = fit_cube_edge_line(
            contour_points,
            start,
            polygon_points[(index + 1) % len(polygon_points)],
            diagonal,
        )
        if line is None:
            return None
        lines.append(line)

    refined = []
    displacement_limit = max(8.0, 0.06 * diagonal)
    for index, original_vertex in enumerate(polygon_points):
        previous_direction, previous_origin = lines[index - 1]
        next_direction, next_origin = lines[index]
        determinant = float(previous_direction[0] * next_direction[1] - previous_direction[1] * next_direction[0])
        if abs(determinant) < 0.10:
            return None
        coefficients = np.column_stack((previous_direction, -next_direction))
        try:
            parameters = np.linalg.solve(coefficients, next_origin - previous_origin)
        except np.linalg.LinAlgError:
            return None
        intersection = previous_origin + parameters[0] * previous_direction
        if not np.isfinite(intersection).all() or np.linalg.norm(intersection - original_vertex) > displacement_limit:
            return None
        refined.append(intersection)
    return np.asarray(refined, dtype=np.float32)


def _deduplicate_hypotheses(hypotheses):
    result = []
    for hypothesis in hypotheses:
        points = np.asarray(hypothesis["points"], dtype=np.float32).reshape(-1, 2)
        if len(points) not in (4, 5, 6) or not np.isfinite(points).all():
            continue
        points = canonicalize_cube_points(points)
        duplicate = False
        for existing in result:
            if len(existing["points"]) != len(points):
                continue
            aligned, error = align_cube_points(points, existing["points"])
            if error < 0.5:
                duplicate = True
                break
        if not duplicate:
            result.append({"method": hypothesis.get("method", "raw"), "points": points})
    return result


def cube_corner_hypotheses(gray, contour, approx):
    """Return raw, sub-pixel, and Huber line-fit hypotheses for an approximation."""

    raw_points = canonicalize_cube_points(np.asarray(approx, dtype=np.float32).reshape(-1, 2))
    candidates = [
        ("raw", raw_points),
        ("corner_subpix", refine_cube_points(gray, raw_points)),
        ("line_fit", line_fitted_cube_points(contour, raw_points)),
    ]
    return _deduplicate_hypotheses([
        {"method": method, "points": points}
        for method, points in candidates
        if points is not None
    ])


def _convex_hull(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None
    hull = cv.convexHull(points)
    return hull.reshape(-1, 2) if len(hull) >= 3 else None


def _hull_overlap(first, second):
    first = _convex_hull(first)
    second = _convex_hull(second)
    if first is None or second is None:
        return 0.0
    first_area = abs(float(cv.contourArea(first)))
    second_area = abs(float(cv.contourArea(second)))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    try:
        intersection, _ = cv.intersectConvexConvex(first.astype(np.float32), second.astype(np.float32))
    except cv.error:
        return 0.0
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0 else 0.0


def _reference_geometry(previous_pose, previous_points):
    if isinstance(previous_pose, dict):
        projected = previous_pose.get("projected_points")
        if projected is not None:
            return _convex_hull(projected), np.asarray(projected, dtype=np.float32).reshape(-1, 2)
    if previous_points is not None:
        points = np.asarray(previous_points, dtype=np.float32).reshape(-1, 2)
        return _convex_hull(points), points
    return None, None


def _select_default_points(hypotheses, reference_points=None):
    if not hypotheses:
        return None
    if reference_points is not None:
        count = len(reference_points)
        same_count = [item for item in hypotheses if len(item["points"]) == count]
        if same_count:
            return same_count[0]["points"]
    # Prefer the richest silhouette on acquisition; face-on views naturally
    # have only four hypotheses.
    return max(hypotheses, key=lambda item: len(item["points"]))["points"]


def detect_cube(frame, previous_pose=None, previous_points=None):
    """Detect a dark cube and retain four-, five-, and six-corner hypotheses.

    ``previous_pose`` is the preferred association reference.  The legacy
    ``previous_points`` keyword/second positional argument remains accepted so
    callers written against the original point tracker continue to work.
    """

    if previous_points is None and previous_pose is not None and not isinstance(previous_pose, dict):
        previous_points = previous_pose
        previous_pose = None

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(gray, 0, CUBE_DARK_THRESHOLD)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    frame_height, frame_width = gray.shape[:2]
    candidates = []
    for contour in contours:
        area = float(cv.contourArea(contour))
        if area < CUBE_MIN_AREA:
            continue
        x, y, width, height = cv.boundingRect(contour)
        if x <= 1 or y <= 1 or x + width >= frame_width - 1 or y + height >= frame_height - 1:
            continue
        if width <= 0 or height <= 0:
            continue
        aspect_ratio = max(width, height) / float(min(width, height))
        if aspect_ratio > 2.0:
            continue
        perimeter = cv.arcLength(contour, True)
        if perimeter <= 0:
            continue
        fill_ratio = area / float(width * height)
        if fill_ratio < 0.45:
            continue

        contour_hypotheses = []
        for epsilon_fraction in CUBE_APPROX_EPSILON_FRACTIONS:
            approx = cv.approxPolyDP(contour, epsilon_fraction * perimeter, True)
            if len(approx) not in (4, 5, 6) or not cv.isContourConvex(approx):
                continue
            contour_hypotheses.extend(cube_corner_hypotheses(gray, contour, approx))
        hypotheses = _deduplicate_hypotheses(contour_hypotheses)
        if not hypotheses:
            continue
        dense_hull = _convex_hull(contour)
        candidates.append({
            "area": area,
            "contour": contour,
            "dense_contour": contour,
            "observed_contour": contour,
            "observed_hull": dense_hull,
            "point_hypotheses": hypotheses,
            "points": _select_default_points(hypotheses, previous_points),
        })

    if not candidates:
        return None, mask

    reference_hull, reference_points = _reference_geometry(previous_pose, previous_points)
    if reference_hull is None:
        selected = max(candidates, key=lambda item: item["area"])
    else:
        reference_center = np.mean(reference_hull, axis=0)
        reference_area = max(abs(float(cv.contourArea(reference_hull))), 1.0)
        reference_diagonal = float(np.linalg.norm(np.ptp(reference_hull, axis=0)))
        tracked = []
        for candidate in candidates:
            current_hull = candidate["observed_hull"]
            if current_hull is None:
                continue
            current_center = np.mean(current_hull, axis=0)
            current_area = max(abs(float(cv.contourArea(current_hull))), 1.0)
            center_error = float(np.linalg.norm(current_center - reference_center))
            scale_error = abs(math.log(current_area / reference_area))
            overlap = _hull_overlap(current_hull, reference_hull)
            max_center_error = max(100.0, 0.70 * reference_diagonal)
            if center_error > max_center_error or scale_error > math.log(2.0):
                continue
            # Require overlap for a competing object unless it is very close
            # to the predicted centre (a one-frame silhouette jump can have a
            # small IoU at a sharp edge transition).
            if overlap <= 0.01 and center_error > max(24.0, 0.20 * reference_diagonal):
                continue
            candidate["association"] = {
                "hull_iou": overlap,
                "centroid_displacement": center_error,
                "scale_change": scale_error,
            }
            candidate["track_score"] = 100.0 * (1.0 - overlap) + center_error + 25.0 * scale_error
            tracked.append(candidate)
        if not tracked:
            return None, mask
        selected = min(tracked, key=lambda item: item["track_score"])

    selected["points"] = _select_default_points(selected["point_hypotheses"], reference_points)
    return selected, mask

