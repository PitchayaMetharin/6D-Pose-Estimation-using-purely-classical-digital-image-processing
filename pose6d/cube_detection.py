"""Dark-cube segmentation, multi-resolution corner hypotheses, and tracking."""

import math

import cv2 as cv
import numpy as np

from .settings import (
    CUBE_APPROX_EPSILON_FRACTIONS, CUBE_DARK_THRESHOLD,
    CUBE_EDGE_CANNY_HIGH, CUBE_EDGE_CANNY_LOW,
    CUBE_EDGE_MIN_PER_EDGE_SUPPORT, CUBE_EDGE_MIN_TOTAL_SUPPORT,
    CUBE_EDGE_SUPPORT_BAND_PIXELS, CUBE_EXCLUSION_DILATION_PIXELS,
    CUBE_MAX_TRACK_ERROR_PIXELS, CUBE_MIN_AREA,
    CUBE_PARTIAL_MAX_CORNER_ERROR_PIXELS, CUBE_PARTIAL_MIN_VERTICES,
    CUBE_PARTIAL_MAX_ANGLE_ERROR_DEG, CUBE_PARTIAL_MAX_EDGE_RESIDUAL_PIXELS,
    CUBE_PARTIAL_MAX_PERP_DISTANCE_PIXELS, CUBE_PARTIAL_MAX_VERTEX_ERROR_PIXELS,
    CUBE_PARTIAL_MIN_EDGE_OVERLAP, CUBE_PARTIAL_MIN_PERIMETER_COVERAGE,
    CUBE_PARTIAL_ROI_MARGIN_PIXELS, CUBE_PARTIAL_DEBUG_DIM_VALUE,
)
from .cube_geometry import cube_object_points


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


def _edge_support_metrics(gray, polygon, edge_map=None):
    """Measure hard physical support along a polygon's retained edges."""
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return [], 0.0
    if edge_map is None:
        edge_map = cv.Canny(
            cv.GaussianBlur(gray, (3, 3), 0),
            CUBE_EDGE_CANNY_LOW,
            CUBE_EDGE_CANNY_HIGH,
        )
    support = []
    weighted = []
    band = int(CUBE_EDGE_SUPPORT_BAND_PIXELS)
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            support.append(0.0)
            weighted.append((0.0, 0.0))
            continue
        count = max(8, int(math.ceil(length)))
        samples = start + np.linspace(0.0, 1.0, count)[:, None] * vector
        coordinates = np.rint(samples).astype(int)
        present = []
        height, width = edge_map.shape[:2]
        for x, y in coordinates:
            x0 = max(0, x - band)
            x1 = min(width, x + band + 1)
            y0 = max(0, y - band)
            y1 = min(height, y + band + 1)
            present.append(bool(np.any(edge_map[y0:y1, x0:x1] > 0)))
        value = float(np.mean(present)) if present else 0.0
        support.append(value)
        weighted.append((length, value))
    total_length = sum(length for length, _value in weighted)
    total_support = (
        sum(length * value for length, value in weighted) / total_length
        if total_length > 0 else 0.0
    )
    return support, float(total_support)


def _polygon_has_physical_edges(gray, polygon, edge_map=None):
    per_edge, total = _edge_support_metrics(gray, polygon, edge_map=edge_map)
    return (
        bool(per_edge)
        and total >= CUBE_EDGE_MIN_TOTAL_SUPPORT
        and min(per_edge) >= CUBE_EDGE_MIN_PER_EDGE_SUPPORT,
        per_edge,
        total,
    )


def _predicted_silhouette_cycle(projected):
    projected = np.asarray(projected, dtype=np.float32).reshape(-1, 2)
    if len(projected) < 3:
        return None
    hull_indices = cv.convexHull(
        projected.reshape(-1, 1, 2), returnPoints=False,
    ).reshape(-1).astype(int)
    if len(hull_indices) < 3:
        return None
    return hull_indices


def _line_angle_error(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return float("inf")
    cosine = abs(float(np.dot(first, second))) / (first_norm * second_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _line_distance_to_infinite_edge(points, start, end):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9 or len(points) == 0:
        return np.full(len(points), float("inf"), dtype=np.float64)
    return np.abs(vector[0] * (points[:, 1] - start[1]) - vector[1] * (points[:, 0] - start[0])) / length


def _segment_overlap_fraction(segment, start, end):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return 0.0
    unit = vector / length
    points = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    coordinates = (points - start) @ unit
    overlap = max(0.0, min(length, float(np.max(coordinates))) - max(0.0, float(np.min(coordinates))))
    return overlap / length


def _line_contrast_metrics(gray, segment):
    """Return support for a consistent dark/light physical boundary."""
    segment = np.asarray(segment, dtype=np.float64).reshape(2, 2)
    vector = segment[1] - segment[0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return 0.0, 0.0, 0.0
    tangent = vector / length
    normal = np.array([-tangent[1], tangent[0]])
    count = max(8, int(math.ceil(length)))
    points = segment[0] + np.linspace(0.10, 0.90, count)[:, None] * vector
    height, width = gray.shape[:2]
    differences = []
    signs = []
    for point in points:
        plus = np.rint(point + 3.0 * normal).astype(int)
        minus = np.rint(point - 3.0 * normal).astype(int)
        if (
            plus[0] < 0 or plus[1] < 0 or plus[0] >= width or plus[1] >= height
            or minus[0] < 0 or minus[1] < 0 or minus[0] >= width or minus[1] >= height
        ):
            continue
        difference = float(gray[plus[1], plus[0]]) - float(gray[minus[1], minus[0]])
        differences.append(abs(difference))
        signs.append(1 if difference >= 0 else -1)
    if not differences:
        return 0.0, 0.0, 0.0
    differences = np.asarray(differences, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.int8)
    strong = differences >= 25.0
    support = float(np.mean(strong))
    polarity = (
        float(max(np.count_nonzero(signs[strong] > 0), np.count_nonzero(signs[strong] < 0)))
        / max(int(np.count_nonzero(strong)), 1)
    )
    mean_contrast = float(np.mean(differences[strong])) if np.any(strong) else 0.0
    return support, polarity, mean_contrast


def _partial_cube_line_matches(gray, previous_pose, exclusion_mask=None):
    """Match Hough segments one-to-one to the predicted silhouette cycle."""
    projected = previous_pose.get("projected_points") if isinstance(previous_pose, dict) else None
    cycle = _predicted_silhouette_cycle(projected) if projected is not None else None
    if cycle is None:
        return None
    projected = np.asarray(projected, dtype=np.float32).reshape(8, 2)
    height, width = gray.shape[:2]
    hull_points = projected[cycle]
    min_xy = np.maximum(
        np.floor(np.min(hull_points, axis=0) - CUBE_PARTIAL_ROI_MARGIN_PIXELS), 0,
    ).astype(int)
    max_xy = np.minimum(
        np.ceil(np.max(hull_points, axis=0) + CUBE_PARTIAL_ROI_MARGIN_PIXELS),
        [width - 1, height - 1],
    ).astype(int)
    if np.any(max_xy <= min_xy):
        return None
    roi = gray[min_xy[1]:max_xy[1] + 1, min_xy[0]:max_xy[0] + 1]
    edges = cv.Canny(
        cv.GaussianBlur(roi, (3, 3), 0),
        CUBE_EDGE_CANNY_LOW,
        CUBE_EDGE_CANNY_HIGH,
    )
    exclusion_roi = None
    if exclusion_mask is not None:
        exclusion = np.asarray(exclusion_mask, dtype=np.uint8)
        if exclusion.shape != gray.shape:
            raise ValueError("exclusion_mask must match the grayscale frame shape")
        exclusion_roi = exclusion[min_xy[1]:max_xy[1] + 1, min_xy[0]:max_xy[0] + 1] > 0
        edges[exclusion_roi] = 0
    # Hough is restricted to this local ROI; segments from the rest of the
    # frame cannot become a partial acquisition.
    min_line_length = max(6, int(0.06 * min(roi.shape)))
    lines = cv.HoughLinesP(
        edges,
        1.0,
        np.pi / 180.0,
        threshold=max(5, int(0.04 * min(roi.shape))),
        minLineLength=min_line_length,
        maxLineGap=10,
    )
    if lines is None:
        return None
    local_lines = lines.reshape(-1, 4).astype(np.float32)
    local_lines[:, [0, 2]] += min_xy[0]
    local_lines[:, [1, 3]] += min_xy[1]
    matches = []
    used_segments = set()
    for edge_id, (start_index, end_index) in enumerate(zip(cycle, np.roll(cycle, -1))):
        start = projected[start_index]
        end = projected[end_index]
        predicted_vector = end - start
        edge_length = float(np.linalg.norm(predicted_vector))
        if edge_length <= 1e-6:
            continue
        edge_candidates = []
        for segment_index, line in enumerate(local_lines):
            if segment_index in used_segments:
                continue
            segment = line.reshape(2, 2)
            if exclusion_roi is not None:
                local_segment = np.rint(
                    segment - min_xy,
                ).astype(int)
                local_samples = np.rint(
                    segment[0]
                    + np.linspace(0.0, 1.0, 8)[:, None]
                    * (segment[1] - segment[0])
                    - min_xy
                ).astype(int)
                if any(
                    0 <= point[0] < exclusion_roi.shape[1]
                    and 0 <= point[1] < exclusion_roi.shape[0]
                    and exclusion_roi[point[1], point[0]]
                    for point in np.vstack((local_segment, local_samples))
                ):
                    continue
            segment_vector = segment[1] - segment[0]
            angle_error = _line_angle_error(predicted_vector, segment_vector)
            if angle_error > CUBE_PARTIAL_MAX_ANGLE_ERROR_DEG:
                continue
            distances = _line_distance_to_infinite_edge(segment, start, end)
            perpendicular = float(np.max(distances))
            if perpendicular > CUBE_PARTIAL_MAX_PERP_DISTANCE_PIXELS:
                continue
            overlap = _segment_overlap_fraction(segment, start, end)
            if overlap < CUBE_PARTIAL_MIN_EDGE_OVERLAP:
                continue
            contrast_support, contrast_polarity, mean_contrast = _line_contrast_metrics(gray, segment)
            if contrast_support < 0.60 or contrast_polarity < 0.70 or mean_contrast < 25.0:
                continue
            # Sample a line at approximately one-pixel spacing.  These are
            # retained for independent residual validation after PnP.
            sample_count = max(4, int(math.ceil(float(np.linalg.norm(segment)))))
            samples = segment[0] + np.linspace(0.0, 1.0, sample_count)[:, None] * segment_vector
            edge_candidates.append({
                "segment_index": int(segment_index),
                "edge_id": edge_id,
                "vertex_indices": (int(start_index), int(end_index)),
                "segment": segment.copy(),
                "line_samples": samples.astype(np.float32),
                "angle_error_deg": float(angle_error),
                "perpendicular_distance_pixels": perpendicular,
                "overlap": float(overlap),
                "contrast_support": float(contrast_support),
                "contrast_polarity": float(contrast_polarity),
                "mean_contrast": float(mean_contrast),
            })
        if edge_candidates:
            selected = min(
                edge_candidates,
                key=lambda item: (
                    item["angle_error_deg"],
                    item["perpendicular_distance_pixels"],
                    -item["overlap"],
                ),
            )
            used_segments.add(int(selected["segment_index"]))
            matches.append(selected)
    # Hough fragments are allowed to compete for an edge, but each predicted
    # edge may be claimed at most once by construction.  Sort by cycle order so
    # adjacent intersections are deterministic.
    matches.sort(key=lambda item: item["edge_id"])
    perimeter = sum(float(np.linalg.norm(projected[end] - projected[start])) for start, end in zip(cycle, np.roll(cycle, -1)))
    covered = sum(
        float(np.linalg.norm(projected[item["vertex_indices"][1]] - projected[item["vertex_indices"][0]])) * item["overlap"]
        for item in matches
    )
    coverage = covered / perimeter if perimeter > 0 else 0.0
    return {
        "cycle": cycle,
        "matches": matches,
        "edge_coverage": float(coverage),
        "roi": (min_xy, max_xy),
    }


def _partial_cube_vertices_from_edges(line_match_result, projected):
    if line_match_result is None:
        return []
    cycle = np.asarray(line_match_result["cycle"], dtype=int)
    by_edge = {int(item["edge_id"]): item for item in line_match_result["matches"]}
    vertices = []
    for edge_position, vertex_index in enumerate(cycle):
        previous_edge = (edge_position - 1) % len(cycle)
        current_edge = edge_position
        first = by_edge.get(previous_edge)
        second = by_edge.get(current_edge)
        if first is None or second is None:
            continue
        first_segment = first["segment"]
        second_segment = second["segment"]
        first_direction = first_segment[1] - first_segment[0]
        second_direction = second_segment[1] - second_segment[0]
        determinant = float(first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0])
        if abs(determinant) < 1e-6:
            continue
        matrix = np.column_stack((first_direction, -second_direction))
        try:
            parameters = np.linalg.solve(matrix, second_segment[0] - first_segment[0])
        except np.linalg.LinAlgError:
            continue
        intersection = first_segment[0] + parameters[0] * first_direction
        if not np.isfinite(intersection).all():
            continue
        error = float(np.linalg.norm(intersection - projected[vertex_index]))
        if error > CUBE_PARTIAL_MAX_VERTEX_ERROR_PIXELS:
            continue
        vertices.append({
            "vertex_index": int(vertex_index),
            "point": intersection.astype(np.float32),
            "prediction_error": error,
            "edge_ids": (int(previous_edge), int(current_edge)),
        })
    return vertices


def detect_cube_partial(frame, previous_pose, gray=None, detailed=False,
                        exclusion_mask=None, return_diagnostics=False):
    """Return a trusted-track-only partial cube observation, if possible.

    The fallback is line evidence constrained to the previous projected
    silhouette.  It deliberately does not inspect arbitrary corners: three
    unrelated corners can always be made to fit a seeded PnP solve.
    """
    detailed = bool(detailed or return_diagnostics)
    diagnostics = {
        "partial": True,
        "solve_mode": "partial_local",
        "status": "PARTIAL EDGE EVIDENCE",
        "rejection_reason": None,
        "matched_edge_ids": [],
        "edge_coverage": 0.0,
        "angle_error_deg": None,
        "independent_edge_residual_pixels": None,
        "independent_edge_residual": None,
    }
    if previous_pose is None:
        diagnostics["rejection_reason"] = "NO TRUSTED CUBE POSE"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None
    if gray is None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    projected = previous_pose.get("projected_points") if isinstance(previous_pose, dict) else None
    if projected is None:
        diagnostics["rejection_reason"] = "NO PREDICTED SILHOUETTE"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None
    projected = np.asarray(projected, dtype=np.float32).reshape(8, 2)
    line_result = _partial_cube_line_matches(
        gray, previous_pose, exclusion_mask=exclusion_mask,
    )
    if line_result is None:
        diagnostics["rejection_reason"] = "PARTIAL EDGE SUPPORT"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None
    matches = line_result["matches"]
    diagnostics.update({
        "matched_edge_ids": [int(item["edge_id"]) for item in matches],
        "matched_edge_vertex_ids": [
            tuple(int(value) for value in item["vertex_indices"]) for item in matches
        ],
        "edge_coverage": float(line_result["edge_coverage"]),
        "perimeter_coverage": float(line_result["edge_coverage"]),
        "angle_error_deg": max(
            (float(item["angle_error_deg"]) for item in matches),
            default=None,
        ),
    })
    if len(matches) < CUBE_PARTIAL_MIN_VERTICES:
        diagnostics["rejection_reason"] = "INSUFFICIENT PARTIAL EDGES"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None
    if line_result["edge_coverage"] < CUBE_PARTIAL_MIN_PERIMETER_COVERAGE:
        diagnostics["rejection_reason"] = "PARTIAL PERIMETER COVERAGE"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None

    vertices = _partial_cube_vertices_from_edges(line_result, projected)
    if len(vertices) < CUBE_PARTIAL_MIN_VERTICES:
        diagnostics["rejection_reason"] = "PARTIAL VERTEX INTERSECTIONS"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None

    # Prefer a geometrically non-collinear set of reliable intersections for
    # local PnP.  The object indices are direct cube vertex ids.
    object_points = cube_object_points()
    selected = []
    for vertex in sorted(vertices, key=lambda item: item["prediction_error"]):
        if len(selected) < 3:
            selected.append(vertex)
            continue
        selected_indices = [item["vertex_index"] for item in selected]
        trial_indices = selected_indices + [vertex["vertex_index"]]
        matrix = object_points[trial_indices] - np.mean(object_points[trial_indices], axis=0)
        if np.linalg.matrix_rank(matrix, tol=1e-8) >= 2:
            selected.append(vertex)
        if len(selected) >= 6:
            break
    if len(selected) < CUBE_PARTIAL_MIN_VERTICES:
        diagnostics["rejection_reason"] = "INSUFFICIENT PARTIAL VERTICES"
        diagnostics["status"] = diagnostics["rejection_reason"]
        return (None, diagnostics) if detailed else None
    indices = [item["vertex_index"] for item in selected]
    image_points = np.asarray([item["point"] for item in selected], dtype=np.float32)
    line_samples = {
        int(item["edge_id"]): np.asarray(item["line_samples"], dtype=np.float32).copy()
        for item in matches
    }
    observed = projected.copy()
    detection = {
        "partial": True,
        "full": False,
        "profile": "cube",
        "contour": None,
        "observed_contour": None,
        "matched_vertex_indices": indices,
        "matched_object_points": object_points[indices].copy(),
        "matched_image_points": image_points,
        "previous_projected_points": observed.copy(),
        "predicted_silhouette_indices": np.asarray(line_result["cycle"], dtype=int).copy(),
        "matched_edge_ids": [int(item["edge_id"]) for item in matches],
        "matched_edge_vertex_ids": [
            tuple(int(value) for value in item["vertex_indices"]) for item in matches
        ],
        "line_samples": line_samples,
        "matched_lines": [
            np.asarray(item["segment"], dtype=np.float32).copy() for item in matches
        ],
        "edge_coverage": float(line_result["edge_coverage"]),
        "perimeter_coverage": float(line_result["edge_coverage"]),
        "edge_angle_errors_deg": {
            int(item["edge_id"]): float(item["angle_error_deg"]) for item in matches
        },
        "angle_error_deg": diagnostics["angle_error_deg"],
        "edge_angle_error_deg": diagnostics["angle_error_deg"],
        "angle_error": diagnostics["angle_error_deg"],
        "edge_perpendicular_distances": {
            int(item["edge_id"]): float(item["perpendicular_distance_pixels"])
            for item in matches
        },
        "independent_edge_residual_pixels": None,
        "point_count": len(indices),
        "orientation_observable": True,
        "rejection_reason": None,
        "status": "PARTIAL EDGE EVIDENCE",
    }
    diagnostics["rejection_reason"] = None
    return (detection, diagnostics) if detailed else detection


def detect_cube_partial_detailed(frame, previous_pose, gray=None,
                                 exclusion_mask=None):
    """Compatibility-named detailed partial-cube entry point."""
    return detect_cube_partial(
        frame,
        previous_pose,
        gray=gray,
        detailed=True,
        exclusion_mask=exclusion_mask,
    )


def detect_cube(frame, previous_pose=None, previous_points=None, gray=None,
                allow_partial=False, exclusion_mask=None):
    """Detect a full dark cube and retain four-, five-, and six-corner hypotheses.

    ``previous_pose`` is the preferred association reference.  The legacy
    ``previous_points`` keyword/second positional argument remains accepted so
    callers written against the original point tracker continue to work.
    ``allow_partial`` is retained for call compatibility; partial tracking is
    intentionally performed only by the explicit trusted local fallback.
    """

    if exclusion_mask is None and isinstance(allow_partial, np.ndarray):
        # A few integrations passed the new optional mask positionally where
        # the old boolean handheld flag lived.  Keep that call shape safe.
        exclusion_mask = allow_partial
        allow_partial = False
    if previous_points is None and previous_pose is not None and not isinstance(previous_pose, dict):
        previous_points = previous_pose
        previous_pose = None

    if gray is None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    raw_dark = cv.inRange(gray, 0, CUBE_DARK_THRESHOLD)
    # Ball pixels are removed both before and after morphology.  The second
    # removal matters because closing would otherwise bridge a ball edge back
    # into a dark cube/shadow component.
    exclusion = None
    if exclusion_mask is not None:
        exclusion = np.asarray(exclusion_mask, dtype=np.uint8)
        if exclusion.shape != raw_dark.shape:
            raise ValueError("exclusion_mask must match the grayscale frame shape")
        kernel_size = 2 * int(CUBE_EXCLUSION_DILATION_PIXELS) + 1
        exclusion = cv.dilate(
            (exclusion > 0).astype(np.uint8) * 255,
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        )
        raw_dark = cv.bitwise_and(raw_dark, cv.bitwise_not(exclusion))
    mask = cv.morphologyEx(raw_dark, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    if exclusion is not None:
        mask = cv.bitwise_and(mask, cv.bitwise_not(exclusion))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Debug output is intentionally not a binary copy of the morphology mask:
    # rejected dark material remains dim gray, while only contours that pass
    # the preliminary gates are white.
    debug_mask = np.zeros_like(gray, dtype=np.uint8)
    debug_mask[raw_dark > 0] = np.uint8(CUBE_PARTIAL_DEBUG_DIM_VALUE)
    edge_map = cv.Canny(
        cv.GaussianBlur(gray, (3, 3), 0),
        CUBE_EDGE_CANNY_LOW,
        CUBE_EDGE_CANNY_HIGH,
    )

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
            edge_ok, edge_support, total_edge_support = _polygon_has_physical_edges(
                gray, approx.reshape(-1, 2), edge_map=edge_map,
            )
            if not edge_ok:
                continue
            contour_hypotheses.extend(cube_corner_hypotheses(gray, contour, approx))
        hypotheses = _deduplicate_hypotheses(contour_hypotheses)
        if not hypotheses:
            continue
        cv.drawContours(debug_mask, [contour], -1, 255, -1)
        dense_hull = _convex_hull(contour)
        candidates.append({
            "area": area,
            "contour": contour,
            "dense_contour": contour,
            "observed_contour": contour,
            "observed_hull": dense_hull,
            "point_hypotheses": hypotheses,
            "points": _select_default_points(hypotheses, previous_points),
            "edge_support": edge_support,
            "edge_support_total": total_edge_support,
        })

    if not candidates:
        return None, debug_mask

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
            return None, debug_mask
        selected = min(tracked, key=lambda item: item["track_score"])

    selected["points"] = _select_default_points(selected["point_hypotheses"], reference_points)
    return selected, debug_mask
