"""Cube PnP candidate generation, validation, symmetry, and diagnostics."""

import math

import cv2 as cv
import numpy as np

from .cube_geometry import (
    CUBE_SYMMETRIES, CUBE_VERTEX_PERMUTATIONS, cube_geometry_limits, cube_hull_metrics,
    cube_max_fit_error, cube_object_points, cube_pose_rank,
    iter_cube_face_candidates, iter_cube_five_point_candidates,
    iter_cube_six_point_candidates, iter_quad_orders,
    rotation_distance_radians,
)
from .settings import (
    CUBE_LOCAL_MAPPING_LIMIT,
    CUBE_PARTIAL_MAX_ANGLE_ERROR_DEG,
    CUBE_PARTIAL_MAX_EDGE_RESIDUAL_PIXELS,
    CUBE_PARTIAL_MAX_REPROJECTION_ERROR_PIXELS,
)


def _observation_mode(point_count):
    return {
        4: "face_on",
        5: "transition",
        6: "multi_face",
    }.get(int(point_count), "unknown")


def _is_finite_pose(rvec, tvec):
    return np.isfinite(rvec).all() and np.isfinite(tvec).all()


def _non_coplanar(object_points):
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 5:
        return False
    centered = points - np.mean(points, axis=0)
    return int(np.linalg.matrix_rank(centered, tol=1e-9)) >= 3


def _point_indices(object_points):
    cube = cube_object_points().astype(np.float64)
    indices = []
    for point in np.asarray(object_points, dtype=np.float64).reshape(-1, 3):
        distances = np.linalg.norm(cube - point, axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > 1e-7:
            return None
        indices.append(index)
    return indices


def _mapping_temporal_error(object_points, image_points, previous_pose):
    if previous_pose is None or previous_pose.get("projected_points") is None:
        return float("inf")
    indices = _point_indices(object_points)
    if indices is None:
        return float("inf")
    projected = np.asarray(previous_pose["projected_points"], dtype=np.float32).reshape(8, 2)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    # The previous pose is already symmetry-normalized, but a reduced mapping
    # may be expressed in a different cube frame.  Compare against all vertex
    # permutations in 2D without reprojecting, then use the nearest score for
    # local correspondence ranking.
    return min(
        float(np.mean(np.linalg.norm(image_points - projected[list(permutation)][indices], axis=1)))
        for permutation in CUBE_VERTEX_PERMUTATIONS
    )


def _solve_pnp(object_points, image_points, camera_matrix, dist_coeffs, previous_pose=None):
    """Solve one correspondence, using SQPNP only as a five-point initializer."""

    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    point_count = len(object_points)
    solutions = []

    if point_count == 4:
        try:
            result = cv.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_IPPE,
            )
        except cv.error:
            return solutions
        if len(result) >= 3 and result[0]:
            solutions.extend((np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                              np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy())
                             for rvec, tvec in zip(result[1], result[2]))
        return solutions

    guess_rvec = None
    guess_tvec = None
    if previous_pose is not None:
        # OpenCV may update the extrinsic guess buffers in place.  Never pass
        # a view into the trusted pose, or a candidate solve can corrupt the
        # reference used by the next mapping/frame.
        guess_rvec = np.asarray(previous_pose.get("rvec"), dtype=np.float64).reshape(3, 1).copy()
        guess_tvec = np.asarray(previous_pose.get("center_tvec", previous_pose.get("tvec")), dtype=np.float64).reshape(3, 1).copy()

    if point_count == 5:
        # OpenCV's iterative DLT initializer requires six points.  SQPNP is
        # the classical non-planar five-point initializer; immediately refine
        # it with the requested iterative solver.
        sqpnp = getattr(cv, "SOLVEPNP_SQPNP", None)
        if sqpnp is not None:
            try:
                success, rvec, tvec = cv.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs,
                    flags=sqpnp,
                )
            except cv.error:
                success = False
            if success:
                guess_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
                guess_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()

    try:
        if guess_rvec is not None and guess_tvec is not None:
            success, rvec, tvec = cv.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                guess_rvec,
                guess_tvec,
                True,
                cv.SOLVEPNP_ITERATIVE,
            )
        else:
            success, rvec, tvec = cv.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_ITERATIVE,
            )
    except cv.error:
        success = False

    if success and _is_finite_pose(rvec, tvec):
        solutions.append((np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                          np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()))
    elif point_count == 5 and guess_rvec is not None and guess_tvec is not None:
        # Keep a valid SQPNP solution if iterative refinement is unavailable in
        # an older OpenCV build.
        solutions.append((guess_rvec.copy(), guess_tvec.copy()))
    return solutions


def build_cube_pose(object_points, image_points, rvec, tvec, observed_points,
                    camera_matrix, dist_coeffs, refinement_method="input"):
    pose, _ = build_cube_pose_detailed(
        object_points, image_points, rvec, tvec, observed_points,
        camera_matrix, dist_coeffs, refinement_method,
    )
    return pose


def build_cube_pose_detailed(object_points, image_points, rvec, tvec,
                             observed_points, camera_matrix, dist_coeffs,
                             refinement_method="input"):
    """Validate one solved pose against the dense observed silhouette."""

    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    observed_points = np.asarray(observed_points, dtype=np.float32).reshape(-1, 2)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
    point_count = len(image_points)
    evaluation = {
        "rejection_reason": "INVALID DEPTH",
        "point_count": point_count,
        "refinement_method": refinement_method,
        "reprojection_rms": None,
        "reprojection_limit": None,
        "hull_edge_error": None,
        "hull_edge_limit": None,
        "hull_iou": None,
        "hull_iou_minimum": None,
        "area_ratio": None,
    }
    if not _is_finite_pose(rvec, tvec):
        return None, evaluation

    rotation_matrix, _ = cv.Rodrigues(rvec)
    object_cube = cube_object_points()
    camera_cube_points = (rotation_matrix @ object_cube.T + tvec).T
    if np.any(camera_cube_points[:, 2] <= 0):
        return None, evaluation

    try:
        projected_cube, _ = cv.projectPoints(object_cube, rvec, tvec, camera_matrix, dist_coeffs)
        projected_object, _ = cv.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    except cv.error:
        evaluation["rejection_reason"] = "GEOMETRY"
        return None, evaluation
    projected_cube = projected_cube.reshape(8, 2)
    projected_object = projected_object.reshape(-1, 2)
    if not np.isfinite(projected_cube).all() or not np.isfinite(projected_object).all():
        evaluation["rejection_reason"] = "GEOMETRY"
        return None, evaluation

    paired_error = float(np.sqrt(np.mean(np.sum((projected_object - image_points) ** 2, axis=1))))
    hull_metrics = cube_hull_metrics(observed_points, projected_cube)
    if hull_metrics is None or not math.isfinite(paired_error):
        evaluation["rejection_reason"] = "GEOMETRY"
        return None, evaluation
    limits = cube_geometry_limits(observed_points)
    evaluation.update({
        "rejection_reason": "GEOMETRY",
        "reprojection_rms": paired_error,
        "reprojection_limit": limits["reprojection_limit"],
        "hull_edge_error": hull_metrics["hull_edge_error"],
        "hull_edge_limit": limits["hull_edge_limit"],
        "hull_iou": hull_metrics["hull_iou"],
        "hull_iou_minimum": limits["hull_iou_minimum"],
        "area_ratio": hull_metrics["area_ratio"],
    })

    if point_count == 4:
        fit_error = paired_error + hull_metrics["hull_edge_error"] + 20.0 * abs(math.log(max(hull_metrics["area_ratio"], 1e-12)))
        # The dense contour is intentionally used for the hull gates, but a
        # planar solve keeps the historical, looser quad fit limit.  Passing
        # the extracted four points here avoids treating the many dense
        # contour samples as a non-planar observation.
        geometry_valid = fit_error <= cube_max_fit_error(image_points)
    else:
        fit_error = cube_pose_rank(paired_error, hull_metrics)
        geometry_valid = (
            paired_error <= limits["reprojection_limit"]
            and hull_metrics["hull_edge_error"] <= limits["hull_edge_limit"]
            and hull_metrics["hull_iou"] >= limits["hull_iou_minimum"]
            and limits["area_ratio_minimum"] <= hull_metrics["area_ratio"] <= limits["area_ratio_maximum"]
        )
    if not geometry_valid:
        return None, evaluation

    evaluation["rejection_reason"] = None
    pose = {
        "rvec": rvec,
        "tvec": tvec,
        "center_tvec": tvec.copy(),
        "projected_points": projected_cube,
        "score": float(fit_error),
        "fit_error": float(fit_error),
        "reprojection_error": paired_error,
        "hull_edge_error": hull_metrics["hull_edge_error"],
        "hull_iou": hull_metrics["hull_iou"],
        "area_ratio": hull_metrics["area_ratio"],
        "point_count": point_count,
        "observation_mode": _observation_mode(point_count),
        "orientation_ambiguous": point_count == 4,
        "refinement_method": refinement_method,
        "image_points": image_points.copy(),
        "object_points": object_points.copy(),
        "observed_points": observed_points.copy(),
    }
    return pose, evaluation


def pose_with_rotation(pose, rvec, camera_matrix, dist_coeffs):
    rotated_pose = {key: value for key, value in pose.items()}
    rotated_pose["rvec"] = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rotated_pose["tvec"] = pose["tvec"].copy()
    return refresh_cube_pose_geometry(rotated_pose, camera_matrix, dist_coeffs)


def _nearest_symmetric_pose(pose, previous_pose):
    """Return the nearest equivalent rotation without refreshing geometry."""

    rotation_matrix, _ = cv.Rodrigues(pose["rvec"])
    reference_rvec = previous_pose["rvec"] if previous_pose is not None else np.zeros((3, 1), dtype=np.float64)
    best_rotation = None
    best_distance = float("inf")
    for symmetry in CUBE_SYMMETRIES:
        equivalent_rotation = rotation_matrix @ symmetry
        equivalent_rvec, _ = cv.Rodrigues(equivalent_rotation)
        if previous_pose is None:
            distance = float(np.linalg.norm(equivalent_rvec))
        else:
            distance = rotation_distance_radians(equivalent_rvec, reference_rvec)
        if distance < best_distance:
            best_distance = distance
            best_rotation = equivalent_rvec

    aligned = {key: value for key, value in pose.items()}
    aligned["rvec"] = np.asarray(best_rotation, dtype=np.float64).reshape(3, 1)
    aligned["tvec"] = pose["tvec"].copy()
    return aligned


def align_cube_pose_symmetry(pose, previous_pose, camera_matrix, dist_coeffs):
    """Select the nearest symmetry and refresh projected geometry once."""

    return refresh_cube_pose_geometry(
        _nearest_symmetric_pose(pose, previous_pose),
        camera_matrix,
        dist_coeffs,
    )


def select_cube_pose(candidate_poses, previous_pose, camera_matrix, dist_coeffs):
    if not candidate_poses:
        return None
    # When the silhouette exposes more than one face, prefer the richer
    # non-planar observation over a simultaneously valid four-point IPPE
    # solution.  The planar candidate is retained as a fallback when no
    # five-/six-point solve survives the dense-hull gates.
    richer = [pose for pose in candidate_poses if pose.get("point_count", 0) > 4]
    if richer:
        candidate_poses = richer
    best_score = min(float(pose["score"]) for pose in candidate_poses)
    # Near face-on, several IPPE/cyclic candidates have almost identical
    # silhouette scores but differ by a small cube symmetry.  Give temporal
    # proximity enough room to retain the previous trusted orientation; the
    # motion gate remains the final rejection authority.
    stability_margin = 50.0 if any(pose.get("point_count") == 4 for pose in candidate_poses) else 1.0
    stable = [pose for pose in candidate_poses if pose["score"] <= best_score + stability_margin]
    stable = [_nearest_symmetric_pose(pose, previous_pose) for pose in stable]
    if previous_pose is None:
        if any(pose.get("point_count") == 4 for pose in stable):
            # A first face-on frame has a planar-depth ambiguity.  Start from
            # the symmetry-equivalent orientation closest to the camera
            # frame; later frames can then carry the normal continuously.
            selected = min(stable, key=lambda pose: (
                float(np.linalg.norm(pose["rvec"])),
                float(pose["score"]),
            ))
        else:
            selected = min(stable, key=lambda pose: float(pose["score"]))
    else:
        if any(pose.get("point_count") == 4 for pose in stable):
            # IPPE can return two nearly identical rotations with materially
            # different depths in a face-on view.  Translation proximity is
            # the useful discriminator while the cube remains ambiguous.
            selected = min(stable, key=lambda pose: (
                float(np.linalg.norm(pose["center_tvec"] - previous_pose["center_tvec"])),
                rotation_distance_radians(pose["rvec"], previous_pose["rvec"]),
                float(pose["score"]),
            ))
        else:
            selected = min(stable, key=lambda pose: (
                rotation_distance_radians(pose["rvec"], previous_pose["rvec"]),
                float(np.linalg.norm(pose["center_tvec"] - previous_pose["center_tvec"])),
                float(pose["score"]),
            ))
    return refresh_cube_pose_geometry(selected, camera_matrix, dist_coeffs)


def cube_pose_diagnostics(point_count, attempted_count, accepted_count,
                          pose=None, evaluation=None, rejection_reason=None,
                          solved_count=None, solve_mode="global",
                          hypotheses_attempted=0, hypotheses_accepted=0,
                          point_counts_attempted=None, point_counts_accepted=None):
    source = dict(evaluation or {})
    if pose is not None:
        limits = cube_geometry_limits(pose.get("observed_points", pose["image_points"]))
        source.update({
            "refinement_method": pose.get("refinement_method"),
            "reprojection_rms": pose.get("reprojection_error"),
            "reprojection_limit": limits["reprojection_limit"],
            "hull_edge_error": pose.get("hull_edge_error"),
            "hull_edge_limit": limits["hull_edge_limit"],
            "hull_iou": pose.get("hull_iou"),
            "hull_iou_minimum": limits["hull_iou_minimum"],
            "area_ratio": pose.get("area_ratio"),
        })
        point_count = pose.get("point_count", point_count)
        solve_mode = pose.get("solve_mode", solve_mode)
        rejection_reason = None
    return {
        "rejection_reason": rejection_reason,
        "point_count": point_count,
        "selected_point_count": point_count,
        "observation_mode": pose.get("observation_mode") if pose is not None else _observation_mode(point_count) if point_count in (4, 5, 6) else "unknown",
        "orientation_ambiguous": bool(pose.get("orientation_ambiguous", False)) if pose is not None else False,
        "refinement_method": source.get("refinement_method"),
        "selected_corner_refinement_method": source.get("refinement_method"),
        "corner_refinement_method": source.get("refinement_method"),
        "reprojection_rms": source.get("reprojection_rms"),
        "reprojection_limit": source.get("reprojection_limit"),
        "reprojection_error": source.get("reprojection_rms"),
        "hull_edge_error": source.get("hull_edge_error"),
        "hull_edge_limit": source.get("hull_edge_limit"),
        "hull_error": source.get("hull_edge_error"),
        "hull_iou": source.get("hull_iou"),
        "hull_iou_minimum": source.get("hull_iou_minimum", 0.88),
        "area_ratio": source.get("area_ratio"),
        "pnp_candidates_attempted": attempted_count,
        "pnp_candidates_accepted": accepted_count,
        "candidates_attempted": attempted_count,
        "candidates_accepted": accepted_count,
        "pnp_solutions": accepted_count if solved_count is None else solved_count,
        "solve_mode": solve_mode,
        "hypotheses_attempted": hypotheses_attempted,
        "hypotheses_accepted": hypotheses_accepted,
        "point_counts_attempted": point_counts_attempted or {},
        "point_counts_accepted": point_counts_accepted or {},
    }


def rejected_cube_pose_rank(evaluation):
    reprojection = evaluation.get("reprojection_rms")
    hull_error = evaluation.get("hull_edge_error")
    iou = evaluation.get("hull_iou")
    area_ratio = evaluation.get("area_ratio")
    if None in (reprojection, hull_error, iou, area_ratio):
        return (float("inf"), float("inf"))
    limits = (
        evaluation.get("reprojection_limit", 1.0),
        evaluation.get("hull_edge_limit", 1.0),
    )
    failures = sum((
        reprojection > limits[0],
        hull_error > limits[1],
        iou < evaluation.get("hull_iou_minimum", 0.88),
        not 0.75 <= area_ratio <= 1.33,
    ))
    score = reprojection / max(limits[0], 1e-6) + hull_error / max(limits[1], 1e-6) + (1.0 - iou) + abs(math.log(max(area_ratio, 1e-12)))
    return failures, score


def _deduplicate_hypotheses(hypotheses):
    result = []
    for hypothesis in hypotheses:
        points = np.asarray(hypothesis["points"], dtype=np.float32).reshape(-1, 2)
        if len(points) not in (4, 5, 6) or not np.isfinite(points).all():
            continue
        duplicate = False
        for existing in result:
            if len(existing["points"]) != len(points):
                continue
            # Mean separation is invariant to the cyclic start because all
            # detection hypotheses have already been canonicalized.
            if float(np.mean(np.linalg.norm(points - existing["points"], axis=1))) < 0.5:
                duplicate = True
                break
        if not duplicate:
            result.append({"method": hypothesis.get("method", "input"), "points": points.copy()})
    return result


def _mappings_for_points(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 4:
        object_face = next(iter_cube_face_candidates())
        return [(object_face, image_points) for image_points in iter_quad_orders(points)]
    if len(points) == 5:
        return list(iter_cube_five_point_candidates(points))
    if len(points) == 6:
        return list(iter_cube_six_point_candidates(points))
    return []


def _estimate_from_hypotheses(hypotheses, observed_points, camera_matrix,
                              dist_coeffs, previous_pose=None, solve_mode=None):
    solve_mode = solve_mode or ("local" if previous_pose is not None else "global")
    hypotheses = _deduplicate_hypotheses(hypotheses)
    if solve_mode == "global":
        # Corner extraction deliberately keeps raw/sub-pixel/line-fit values
        # for diagnostics.  A global solve only needs the strongest
        # representative of each point count; local tracking still ranks the
        # full mixed set before taking its three best mappings.
        method_priority = {"line_fit": 0, "corner_subpix": 1, "raw": 2, "input": 3}
        selected_hypotheses = []
        for point_count in (6, 5, 4):
            same_count = [item for item in hypotheses if len(item["points"]) == point_count]
            if same_count:
                selected_hypotheses.append(min(same_count, key=lambda item: method_priority.get(item.get("method"), 4)))
        hypotheses = selected_hypotheses
    mappings = []
    for hypothesis in hypotheses:
        for object_points, image_points in _mappings_for_points(hypothesis["points"]):
            if len(object_points) == 5 and not _non_coplanar(object_points):
                continue
            mappings.append({
                "object_points": object_points,
                "image_points": image_points,
                "method": hypothesis.get("method", "input"),
                "point_count": len(object_points),
            })

    if solve_mode == "local" and previous_pose is not None:
        mappings.sort(key=lambda item: _mapping_temporal_error(item["object_points"], item["image_points"], previous_pose))
        # A silhouette can change from four to five to six visible corners
        # while the tracker remains trusted.  Pure temporal ranking tends to
        # spend all three local trials on the old planar hypothesis exactly at
        # that boundary, so reserve one trial for each point count present and
        # use the remaining slots for the globally closest correspondences.
        ranked_mappings = mappings
        selected_mappings = []
        selected_ids = set()
        for point_count in (6, 5, 4):
            for index, mapping in enumerate(ranked_mappings):
                if mapping["point_count"] == point_count:
                    selected_mappings.append(mapping)
                    selected_ids.add(index)
                    break
        for index, mapping in enumerate(ranked_mappings):
            if len(selected_mappings) >= CUBE_LOCAL_MAPPING_LIMIT:
                break
            if index not in selected_ids:
                selected_mappings.append(mapping)
        mappings = selected_mappings[:CUBE_LOCAL_MAPPING_LIMIT]

    candidate_poses = []
    evaluations = []
    attempted_count = 0
    successful_count = 0
    point_counts_attempted = {4: 0, 5: 0, 6: 0}
    point_counts_accepted = {4: 0, 5: 0, 6: 0}
    for mapping in mappings:
        point_count = mapping["point_count"]
        attempted_count += 1
        point_counts_attempted[point_count] = point_counts_attempted.get(point_count, 0) + 1
        solutions = _solve_pnp(
            mapping["object_points"], mapping["image_points"],
            camera_matrix, dist_coeffs,
            previous_pose if solve_mode == "local" else None,
        )
        successful_count += len(solutions)
        for rvec, tvec in solutions:
            pose, evaluation = build_cube_pose_detailed(
                mapping["object_points"], mapping["image_points"],
                rvec, tvec, observed_points,
                camera_matrix, dist_coeffs, mapping["method"],
            )
            evaluations.append(evaluation)
            if pose is not None:
                pose["solve_mode"] = solve_mode
                candidate_poses.append(pose)
                point_counts_accepted[point_count] = point_counts_accepted.get(point_count, 0) + 1

    selected_pose = select_cube_pose(candidate_poses, previous_pose, camera_matrix, dist_coeffs)
    if selected_pose is not None:
        return selected_pose, cube_pose_diagnostics(
            selected_pose["point_count"], attempted_count, len(candidate_poses),
            pose=selected_pose, solved_count=successful_count,
            solve_mode=solve_mode, hypotheses_attempted=len(hypotheses),
            hypotheses_accepted=len(candidate_poses),
            point_counts_attempted=point_counts_attempted,
            point_counts_accepted=point_counts_accepted,
        )

    geometry_evaluations = [item for item in evaluations if item.get("reprojection_rms") is not None]
    best_evaluation = min(geometry_evaluations, key=rejected_cube_pose_rank) if geometry_evaluations else (evaluations[0] if evaluations else None)
    if geometry_evaluations:
        rejection_reason = "GEOMETRY"
    elif successful_count:
        rejection_reason = "INVALID DEPTH"
    else:
        rejection_reason = "NO PNP SOLUTION"
    return None, cube_pose_diagnostics(
        len(hypotheses[0]["points"]) if hypotheses else 0,
        attempted_count, 0, evaluation=best_evaluation,
        rejection_reason=rejection_reason, solved_count=successful_count,
        solve_mode=solve_mode, hypotheses_attempted=len(hypotheses),
        point_counts_attempted=point_counts_attempted,
        point_counts_accepted=point_counts_accepted,
    )


def estimate_cube_pose_from_detection(detection, camera_matrix, dist_coeffs, previous_pose=None, solve_mode=None):
    """Estimate a pose from one detection containing mixed 4/5/6-point hypotheses."""

    if detection is None:
        return None, cube_pose_diagnostics(0, 0, 0, rejection_reason="NO CUBE DETECTION")
    hypotheses = detection.get("point_hypotheses") or [{
        "method": "input",
        "points": detection.get("points"),
    }]
    observed_points = detection.get("dense_contour", detection.get("observed_contour", detection.get("contour", detection.get("points"))))
    return _estimate_from_hypotheses(
        hypotheses,
        observed_points,
        camera_matrix,
        dist_coeffs,
        previous_pose,
        solve_mode,
    )


def estimate_cube_pose_from_partial_detection(
    detection, camera_matrix, dist_coeffs, previous_pose=None,
):
    """Refine a trusted cube pose from three or more matched vertices.

    This local path is intentionally impossible to use for global acquisition:
    it requires the previous symmetry-normalized pose and an extrinsic PnP
    guess.  The matched-vertex reprojection gate is the geometric check before
    the normal temporal/depth gates in ``CubeTracker``.
    """
    diagnostics = {
        "solve_mode": "partial_local",
        "partial": True,
        "status": "PARTIAL EDGE EVIDENCE",
        "rejection_reason": None,
    }
    if previous_pose is None or not detection or not detection.get("partial"):
        diagnostics["rejection_reason"] = "NO TRUSTED CUBE POSE"
        return None, diagnostics
    if detection.get("rejection_reason"):
        diagnostics["rejection_reason"] = str(detection["rejection_reason"])
        return None, diagnostics
    object_points = np.asarray(detection.get("matched_object_points"), dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(detection.get("matched_image_points"), dtype=np.float32).reshape(-1, 2)
    if len(object_points) < 3 or len(image_points) != len(object_points):
        diagnostics["rejection_reason"] = "INSUFFICIENT PARTIAL VERTICES"
        return None, diagnostics
    try:
        success, rvec, tvec = cv.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            np.asarray(previous_pose["rvec"], dtype=np.float64).reshape(3, 1).copy(),
            np.asarray(previous_pose["center_tvec"], dtype=np.float64).reshape(3, 1).copy(),
            True,
            cv.SOLVEPNP_ITERATIVE,
        )
    except cv.error:
        success = False
    if not success or not _is_finite_pose(rvec, tvec) or float(tvec[2, 0]) <= 0:
        diagnostics["rejection_reason"] = "PARTIAL PNP"
        return None, diagnostics
    projected, _ = cv.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection = np.linalg.norm(
        projected.reshape(-1, 2) - image_points,
        axis=1,
    )
    reprojection_error = float(np.sqrt(np.mean(reprojection ** 2)))
    diagnostics.update({
        "reprojection_error": reprojection_error,
        "reprojection_rms": reprojection_error,
        "point_count": len(object_points),
        "matched_edge_ids": list(detection.get("matched_edge_ids", [])),
        "edge_coverage": float(
            detection.get("edge_coverage", detection.get("perimeter_coverage", 0.0))
        ),
        "angle_error_deg": detection.get(
            "angle_error_deg", detection.get("edge_angle_error_deg")
        ),
    })
    if reprojection_error > CUBE_PARTIAL_MAX_REPROJECTION_ERROR_PIXELS:
        diagnostics["rejection_reason"] = "PARTIAL GEOMETRY"
        return None, diagnostics

    # Validate the observation independently against the physical line
    # samples.  A seeded PnP fit can make three unrelated corners appear
    # perfect, so fitted vertices are never sufficient evidence by themselves.
    projected_cube, _ = cv.projectPoints(
        cube_object_points(), rvec, tvec, camera_matrix, dist_coeffs,
    )
    projected_cube = projected_cube.reshape(8, 2)
    cycle = np.asarray(
        detection.get("predicted_silhouette_indices", []), dtype=int,
    ).reshape(-1)
    line_samples = detection.get("line_samples") or {}
    edge_ids = [int(item) for item in detection.get("matched_edge_ids", [])]
    residuals = []
    angle_errors = []
    for edge_id in edge_ids:
        if len(cycle) < 3 or edge_id < 0 or edge_id >= len(cycle):
            diagnostics["rejection_reason"] = "PARTIAL EDGE EVIDENCE"
            return None, diagnostics
        if isinstance(line_samples, dict):
            samples = line_samples.get(edge_id, line_samples.get(str(edge_id)))
        elif isinstance(line_samples, (list, tuple)):
            try:
                samples = line_samples[edge_ids.index(edge_id)]
            except (ValueError, IndexError):
                samples = None
        elif isinstance(line_samples, np.ndarray) and line_samples.ndim >= 3:
            try:
                samples = line_samples[edge_ids.index(edge_id)]
            except (ValueError, IndexError):
                samples = None
        else:
            samples = None
        if samples is None and isinstance(detection.get("matched_lines"), (list, tuple)):
            try:
                samples = detection["matched_lines"][edge_ids.index(edge_id)]
            except (ValueError, IndexError):
                samples = None
        if samples is None:
            diagnostics["rejection_reason"] = "PARTIAL EDGE EVIDENCE"
            return None, diagnostics
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 2)
        if len(samples) < 2 or not np.isfinite(samples).all():
            diagnostics["rejection_reason"] = "PARTIAL EDGE EVIDENCE"
            return None, diagnostics
        start_index = int(cycle[edge_id])
        end_index = int(cycle[(edge_id + 1) % len(cycle)])
        predicted_start = projected_cube[start_index]
        predicted_end = projected_cube[end_index]
        vector = predicted_end - predicted_start
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            diagnostics["rejection_reason"] = "PARTIAL EDGE EVIDENCE"
            return None, diagnostics
        sample_direction = samples[-1] - samples[0]
        cosine = abs(float(np.dot(vector, sample_direction))) / max(
            float(np.linalg.norm(vector)) * float(np.linalg.norm(sample_direction)),
            1e-12,
        )
        angle_error = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        angle_errors.append(angle_error)
        if angle_error > CUBE_PARTIAL_MAX_ANGLE_ERROR_DEG:
            diagnostics["rejection_reason"] = "PARTIAL EDGE ANGLE"
            diagnostics["angle_error_deg"] = max(angle_errors)
            return None, diagnostics
        normal_distance = np.abs(
            vector[0] * (samples[:, 1] - predicted_start[1])
            - vector[1] * (samples[:, 0] - predicted_start[0])
        ) / length
        residuals.extend(normal_distance.tolist())
    independent_residual = (
        float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    )
    diagnostics["independent_edge_residual_pixels"] = independent_residual
    diagnostics["independent_edge_residual"] = independent_residual
    diagnostics["edge_residual_rms"] = independent_residual
    diagnostics["angle_error_deg"] = max(angle_errors, default=None)
    if not residuals or independent_residual > CUBE_PARTIAL_MAX_EDGE_RESIDUAL_PIXELS:
        diagnostics["rejection_reason"] = "PARTIAL EDGE RESIDUAL"
        return None, diagnostics
    pose = {key: value for key, value in previous_pose.items()}
    pose.update({
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy(),
        "partial": True,
        "full": False,
        "observation_mode": "partial",
        "orientation_ambiguous": bool(previous_pose.get("orientation_ambiguous", False)),
        "point_count": len(object_points),
        "partial_matched_vertex_indices": list(detection.get("matched_vertex_indices", [])),
        "partial_reprojection_error": reprojection_error,
        "partial_matched_edge_ids": edge_ids,
        "partial_edge_coverage": float(detection.get("edge_coverage", 0.0)),
        "partial_angle_error_deg": diagnostics.get("angle_error_deg"),
        "partial_independent_edge_residual": independent_residual,
        "independent_edge_residual": independent_residual,
        "solve_mode": "partial_local",
    })
    pose = align_cube_pose_symmetry(pose, previous_pose, camera_matrix, dist_coeffs)
    diagnostics["rejection_reason"] = None
    diagnostics["status"] = "PARTIAL EDGE EVIDENCE"
    return pose, diagnostics


def estimate_cube_pose_from_hypotheses(hypotheses, camera_matrix, dist_coeffs, previous_pose=None, observed_points=None, solve_mode=None):
    """Compatibility wrapper for a list of corner hypotheses."""

    if not hypotheses:
        return None, cube_pose_diagnostics(0, 0, 0, rejection_reason="NO PNP SOLUTION")
    if observed_points is None:
        observed_points = hypotheses[0]["points"]
    return _estimate_from_hypotheses(hypotheses, observed_points, camera_matrix, dist_coeffs, previous_pose, solve_mode)


def estimate_cube_pose_detailed(cube_points, camera_matrix, dist_coeffs, previous_pose=None, corner_hypotheses=None):
    """Compatibility entry point accepting a point array or a detection dict."""

    if isinstance(cube_points, dict):
        return estimate_cube_pose_from_detection(cube_points, camera_matrix, dist_coeffs, previous_pose)
    cube_points = np.asarray(cube_points, dtype=np.float32).reshape(-1, 2)
    hypotheses = corner_hypotheses or [{"method": "input", "points": cube_points}]
    # The old API has no dense contour, so the extracted points are the best
    # available geometric observation.
    return _estimate_from_hypotheses(hypotheses, cube_points, camera_matrix, dist_coeffs, previous_pose)


def estimate_cube_pose(cube_points, camera_matrix, dist_coeffs, previous_pose=None):
    pose, _ = estimate_cube_pose_detailed(cube_points, camera_matrix, dist_coeffs, previous_pose)
    return pose


def estimate_cube_pose_from_six_points(cube_points, camera_matrix, dist_coeffs, previous_pose=None):
    pose, _ = estimate_cube_pose_detailed(cube_points, camera_matrix, dist_coeffs, previous_pose)
    return pose


def estimate_cube_pose_from_four_points(cube_points, camera_matrix, dist_coeffs, previous_pose=None):
    pose, _ = estimate_cube_pose_detailed(cube_points, camera_matrix, dist_coeffs, previous_pose)
    return pose


def estimate_cube_pose_from_four_points_detailed(cube_points, camera_matrix, dist_coeffs, previous_pose=None, refinement_method="input"):
    return estimate_cube_pose_detailed(
        cube_points,
        camera_matrix,
        dist_coeffs,
        previous_pose,
        [{"method": refinement_method, "points": cube_points}],
    )


def refresh_cube_pose_geometry(pose, camera_matrix, dist_coeffs):
    """Refresh derived projected geometry exactly once after pose filtering."""

    pose["center_tvec"] = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1).copy()
    pose["projected_points"], _ = cv.projectPoints(
        cube_object_points(), pose["rvec"], pose["tvec"], camera_matrix, dist_coeffs,
    )
    pose["projected_points"] = pose["projected_points"].reshape(8, 2)
    return pose
