"""Cube model geometry, correspondence tables, and silhouette metrics.

The cube is deliberately modelled about its centre.  A featureless cube has
24 equally valid object frames, so correspondence tables are kept small and
symmetry alignment is performed after a candidate has passed the geometric
checks.
"""

import math
from itertools import permutations, product

import cv2 as cv
import numpy as np

from .settings import (
    CUBE_MAX_FIT_ERROR_FRACTION, CUBE_MAX_FIT_ERROR_PIXELS,
    CUBE_MAX_PLANAR_FIT_ERROR_PIXELS, CUBE_SIZE_M,
)


def cube_object_points():
    """Return the eight cube vertices centred at the cube origin."""

    half = CUBE_SIZE_M / 2.0
    return np.asarray([
        [-half,  half,  half],
        [ half,  half,  half],
        [ half, -half,  half],
        [-half, -half,  half],
        [-half,  half, -half],
        [ half,  half, -half],
        [ half, -half, -half],
        [-half, -half, -half],
    ], dtype=np.float32)


# The common silhouette orbit contains eight directed six-cycles.  Two extra
# perspective-only orbit representatives are kept separately for solving;
# exposing the canonical table as eight entries preserves the compact
# correspondence contract while still covering calibrated perspective views.
CUBE_SIX_VERTEX_CYCLES = (
    (0, 1, 2, 6, 7, 4),
    (0, 1, 5, 6, 7, 3),
    (0, 3, 2, 6, 5, 4),
    (0, 3, 7, 6, 5, 1),
    (0, 4, 5, 6, 2, 3),
    (0, 4, 7, 6, 2, 1),
    (1, 2, 3, 7, 4, 5),
    (1, 5, 4, 7, 3, 2),
)

CUBE_SIX_PERSPECTIVE_CYCLE_REPRESENTATIVES = (
    (0, 1, 2, 3, 7, 4),
    (0, 1, 2, 6, 5, 4),
)


# Undirected representatives are useful when constructing the smaller
# five-point table.  Reversed mappings are retained explicitly because a
# proper cube rotation does not identify the two image winding directions.
CUBE_SILHOUETTE_CYCLES = (
    CUBE_SIX_VERTEX_CYCLES[0],
    CUBE_SIX_VERTEX_CYCLES[1],
    CUBE_SIX_VERTEX_CYCLES[2],
    CUBE_SIX_VERTEX_CYCLES[6],
)

# One representative per graph/symmetry orbit is sufficient for solving: the
# selected pose is later expressed in the nearest of the 24 equivalent cube
# frames.  Reverse image winding is still enumerated below.
CUBE_SIX_CORRESPONDENCE_REPRESENTATIVES = (
    (0, 1, 2, 6, 7, 4),
    *CUBE_SIX_PERSPECTIVE_CYCLE_REPRESENTATIVES,
)


CUBE_FACE_CYCLES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def cube_symmetry_matrices():
    """Return the 24 proper rotational symmetries of a cube."""

    matrices = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.zeros((3, 3), dtype=np.float64)
        for row, column in enumerate(permutation):
            permutation_matrix[row, column] = 1.0
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = permutation_matrix @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                matrices.append(candidate)
    return tuple(matrices)


CUBE_SYMMETRIES = cube_symmetry_matrices()


def _vertex_permutations():
    """Map each proper symmetry to a permutation of cube vertex indices."""

    object_points = cube_object_points()
    by_sign = {
        tuple(np.sign(point).astype(np.int8)): index
        for index, point in enumerate(object_points)
    }
    permutations_for_symmetry = []
    for symmetry in CUBE_SYMMETRIES:
        permutations_for_symmetry.append(tuple(
            by_sign[tuple(np.sign(symmetry @ point).astype(np.int8))]
            for point in object_points
        ))
    return tuple(permutations_for_symmetry)


CUBE_VERTEX_PERMUTATIONS = _vertex_permutations()


def _five_mapping_classes():
    """Return the symmetry-reduced directed five-vertex object mappings."""

    # Omitting a vertex from all six-cycle orbits produces six directed
    # classes under proper cube rotations.  Include both winding directions
    # and keep image-side cyclic shifts in iter_cube_five_point_candidates().
    representatives = (
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 7),
        (0, 1, 2, 5, 4),
        (0, 1, 2, 6, 4),
        (0, 1, 2, 6, 5),
        (0, 1, 2, 6, 7),
    )
    # The six representatives already include both winding classes across
    # the three six-cycle orbits.  The five cyclic starts below cover the
    # arbitrary detector start; adding explicit reversed object mappings
    # would only double the global solve cost.
    return tuple(representatives)


CUBE_FIVE_VERTEX_MAPPINGS = _five_mapping_classes()


def iter_cyclic_orders(points):
    """Yield every cyclic and reverse order of an observed point list."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    for ordered_points in (points, points[::-1]):
        for shift in range(len(points)):
            yield np.roll(ordered_points, shift, axis=0).astype(np.float32)


def iter_cube_six_point_candidates(points):
    """Yield cyclic/reverse six-point cube correspondences."""

    points = np.asarray(points, dtype=np.float32).reshape(6, 2)
    object_cube = cube_object_points()
    for cycle in CUBE_SIX_CORRESPONDENCE_REPRESENTATIVES:
        object_points = object_cube[list(cycle)]
        for image_points in iter_cyclic_orders(points):
            yield object_points, image_points


def iter_cube_five_point_candidates(points):
    """Yield symmetry-reduced non-coplanar five-point correspondences."""

    points = np.asarray(points, dtype=np.float32).reshape(5, 2)
    object_cube = cube_object_points()
    for mapping in CUBE_FIVE_VERTEX_MAPPINGS:
        object_points = object_cube[list(mapping)]
        for order_index, image_points in enumerate(iter_cyclic_orders(points)):
            if order_index >= 5:
                continue
            yield object_points, image_points


def iter_cube_face_candidates():
    """Yield the canonical planar cube face.

    The six physical faces are symmetry-equivalent for an unmarked cube.  A
    single face plus the 24 post-solve rotations is both complete and much
    cheaper than solving the same planar problem six times.
    """

    object_cube = cube_object_points()
    yield object_cube[list(CUBE_FACE_CYCLES[0])]


def iter_quad_orders(points):
    """Yield all eight cyclic and reverse-cyclic quad orders."""

    yield from iter_cyclic_orders(np.asarray(points, dtype=np.float32).reshape(4, 2))


def point_to_segment_distance(point, start, end):
    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 0:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / segment_length_squared)
    projection = max(0.0, min(1.0, projection))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def polygon_edge_distance(points, polygon):
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(polygon) < 2:
        return float("inf")
    return float(np.mean([
        min(
            point_to_segment_distance(
                point,
                polygon[index],
                polygon[(index + 1) % len(polygon)],
            )
            for index in range(len(polygon))
        )
        for point in points
    ]))


def _convex_hull(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None
    hull = cv.convexHull(points).reshape(-1, 2)
    return hull if len(hull) >= 3 else None


def cube_hull_metrics(observed_points, projected_points):
    """Measure bidirectional edge error, IoU, and area ratio of silhouettes."""

    observed_hull = _convex_hull(observed_points)
    projected_hull = _convex_hull(projected_points)
    if observed_hull is None or projected_hull is None:
        return None

    observed_area = abs(float(cv.contourArea(observed_hull)))
    projected_area = abs(float(cv.contourArea(projected_hull)))
    if observed_area <= 0 or projected_area <= 0:
        return None

    edge_error = (
        polygon_edge_distance(observed_hull, projected_hull)
        + polygon_edge_distance(projected_hull, observed_hull)
    )
    try:
        intersection_area, _ = cv.intersectConvexConvex(
            observed_hull.astype(np.float32),
            projected_hull.astype(np.float32),
        )
    except cv.error:
        return None

    union_area = observed_area + projected_area - float(intersection_area)
    if union_area <= 0:
        return None

    metrics = {
        "hull_edge_error": float(edge_error),
        "hull_iou": float(intersection_area / union_area),
        "area_ratio": float(projected_area / observed_area),
    }
    return metrics if all(math.isfinite(value) for value in metrics.values()) else None


def cube_geometry_limits(observed_points):
    observed_points = np.asarray(observed_points, dtype=np.float32).reshape(-1, 2)
    if len(observed_points) < 2:
        diagonal = 0.0
    else:
        diagonal = float(np.linalg.norm(np.ptp(observed_points, axis=0)))
    return {
        "reprojection_limit": max(6.0, 0.03 * diagonal),
        "hull_edge_limit": max(8.0, 0.04 * diagonal),
        "hull_iou_minimum": 0.88,
        "area_ratio_minimum": 0.75,
        "area_ratio_maximum": 1.33,
    }


def cube_pose_rank(reprojection_error, hull_metrics):
    return (
        reprojection_error
        + hull_metrics["hull_edge_error"]
        + 20.0 * abs(math.log(max(hull_metrics["area_ratio"], 1e-12)))
        + 20.0 * (1.0 - hull_metrics["hull_iou"])
    )


def rotation_distance_radians(first_rvec, second_rvec):
    first_rotation, _ = cv.Rodrigues(np.asarray(first_rvec, dtype=np.float64))
    second_rotation, _ = cv.Rodrigues(np.asarray(second_rvec, dtype=np.float64))
    relative_rotation = first_rotation @ second_rotation.T
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cosine))))


def cube_max_fit_error(observed_points):
    observed_points = np.asarray(observed_points, dtype=np.float32).reshape(-1, 2)
    diagonal = float(np.linalg.norm(np.ptp(observed_points, axis=0))) if len(observed_points) else 0.0
    if len(observed_points) == 4:
        return max(CUBE_MAX_PLANAR_FIT_ERROR_PIXELS, 0.08 * diagonal)
    return max(CUBE_MAX_FIT_ERROR_PIXELS, CUBE_MAX_FIT_ERROR_FRACTION * diagonal)
