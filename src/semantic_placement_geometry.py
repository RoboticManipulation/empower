"""Semantic placement plan parsing and coordinate geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


def reference_not_found_failure_reason(reference_object: str) -> str:
    return f"Reference object not found: {reference_object}"


def is_semantic_placement_success(result: Mapping[str, Any] | None) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("success") is False:
        return False
    if result.get("reference_object") and result.get("reference_position") is None:
        return False
    coordinates = result.get("coordinates")
    if coordinates is None:
        return False
    try:
        point = np.asarray(coordinates, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return point.shape == (3,) and np.isfinite(point).all()


def _semantic_placement_failure(
    *,
    grasp_object: str,
    frame_id: str,
    failure_reason: str,
    planning_source_line: str | None = None,
    relation: str | None = None,
    reference_object: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "failure_reason": failure_reason,
        "grasp_object": grasp_object,
        "normalized_grasp_object": _normalize_result_name(grasp_object),
        "frame_id": frame_id,
    }
    if planning_source_line is not None:
        result["planning_source_line"] = planning_source_line
    if relation is not None:
        result["relation"] = relation
    if reference_object is not None:
        result["reference_object"] = reference_object
    return result


def get_semantic_placement_coordinates_from_plan(
    planning_text: str,
    *,
    placement_pointclouds: np.ndarray | Sequence[np.ndarray],
    grasp_object: str | None = None,
    placement_surface_height_m: float | None = None,
    reference_positions_by_name: Mapping[str, Sequence[float]] | None = None,
    relation_offset_m: float,
    orientation_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    frame_id: str,
    shelf_board_heights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Convert a semantic placement action plan into camera-frame coordinates."""

    intent = parse_semantic_placement_plan(
        planning_text,
        default_grasp_object=grasp_object,
    )
    if grasp_object:
        intent = SemanticPlacementIntent(
            grasp_object=_clean_object_name(grasp_object),
            source_line=intent.source_line,
            relation=intent.relation,
            reference_object=intent.reference_object,
        )

    result = get_semantic_placement_coordinates(
        intent.grasp_object,
        placement_pointclouds=placement_pointclouds,
        placement_surface_height_m=placement_surface_height_m,
        orientation_rpy=orientation_rpy,
        frame_id=frame_id,
        shelf_board_heights=shelf_board_heights,
    )

    if intent.reference_object:
        matched_name, reference_position = resolve_reference_position_from_action(
            intent.source_line,
            reference_positions_by_name or {},
            reference_object=intent.reference_object,
        )
        if reference_position is None:
            return _semantic_placement_failure(
                grasp_object=intent.grasp_object,
                frame_id=frame_id,
                failure_reason=reference_not_found_failure_reason(intent.reference_object),
                planning_source_line=intent.source_line,
                relation=intent.relation,
                reference_object=intent.reference_object,
            )

        result = _apply_reference_relation(
            result,
            reference_position=reference_position,
            relation=intent.relation,
            relation_offset_m=relation_offset_m,
            shelf_board_heights=shelf_board_heights,
        )
        result["success"] = True
        result["planning_source_line"] = intent.source_line
        result["relation"] = intent.relation
        result["relation_offset_m"] = float(relation_offset_m)
        result["reference_object"] = matched_name or intent.reference_object
        result["reference_position"] = {
            "x": float(reference_position[0]),
            "y": float(reference_position[1]),
            "z": float(reference_position[2]),
        }

    return result


def get_semantic_placement_coordinates(
    grasp_object: str,
    *,
    placement_pointclouds: np.ndarray | Sequence[np.ndarray],
    placement_surface_height_m: float | None = None,
    orientation_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    frame_id: str,
    shelf_board_heights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a fallback surface placement coordinate from point-cloud data."""

    points = _coerce_pointclouds(placement_pointclouds)
    shelf_heights = _coerce_optional_heights(shelf_board_heights, "shelf_board_heights")
    surface_height = _select_surface_height(
        points,
        placement_surface_height_m=placement_surface_height_m,
        shelf_board_heights=shelf_heights,
    )
    surface_xy, support_points = _estimate_surface_xy(points, surface_height)
    coordinates = [float(surface_xy[0]), float(surface_xy[1]), float(surface_height)]
    roll, pitch, yaw = _coerce_rpy(orientation_rpy)

    return {
        "grasp_object": grasp_object,
        "normalized_grasp_object": _normalize_result_name(grasp_object),
        "frame_id": frame_id,
        "coordinates": coordinates,
        "pose": {
            "frame_id": frame_id,
            "position": {
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
            },
            "orientation_rpy": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            },
        },
        "surface_position": {
            "x": coordinates[0],
            "y": coordinates[1],
            "z": coordinates[2],
        },
        "support_point_count": int(support_points.shape[0]),
        "support_bounds": _support_bounds(support_points),
    }


def get_empower_style_semantic_coordinates(
    *,
    planning_text: str,
    placement_pointcloud: np.ndarray,
    grasp_object: str,
    reference_positions: Mapping[str, np.ndarray],
    frame_id: str,
    relation_offset_m: float,
    shelf_board_heights: Sequence[float] | None = None,
) -> dict[str, Any]:
    result = get_semantic_placement_coordinates(
        grasp_object,
        placement_pointclouds=placement_pointcloud,
        frame_id=frame_id,
        shelf_board_heights=shelf_board_heights,
    )

    action_line = last_empower_placement_action(planning_text)
    if action_line is None:
        return result

    relation = parse_empower_relation(action_line)
    reference_name, reference_position = resolve_empower_reference(
        action_line=action_line,
        relation=relation,
        reference_positions=reference_positions,
        grasp_object=grasp_object,
    )
    if reference_position is None:
        if reference_name or relation in {"left", "right", "on"}:
            unresolved_reference = reference_name or "reference object"
            return _semantic_placement_failure(
                grasp_object=grasp_object,
                frame_id=frame_id,
                failure_reason=reference_not_found_failure_reason(unresolved_reference),
                planning_source_line=action_line,
                relation=relation,
                reference_object=reference_name,
            )
        result["planning_source_line"] = action_line
        result["relation"] = relation
        result["reference_object"] = reference_name
        return result

    coordinate = apply_relation_offset(
        reference_position,
        relation=relation,
        relation_offset_m=relation_offset_m,
        shelf_board_heights=shelf_board_heights,
    )
    update_result_coordinate(result, coordinate)
    result["success"] = True
    result["planning_source_line"] = action_line
    result["relation"] = relation
    result["relation_offset_m"] = float(relation_offset_m)
    result["reference_object"] = reference_name
    result["reference_position"] = {
        "x": float(reference_position[0]),
        "y": float(reference_position[1]),
        "z": float(reference_position[2]),
    }
    return result


@dataclass(frozen=True)
class SemanticPlacementIntent:
    """Semantic placement action extracted from an action plan."""

    grasp_object: str
    source_line: str
    relation: str | None = None
    reference_object: str | None = None


def parse_semantic_placement_plan(
    planning_text: str,
    default_grasp_object: str | None = None,
) -> SemanticPlacementIntent:
    """Extract the held object and optional relation/reference from a plan."""

    last_grabbed_object: str | None = (
        _clean_object_name(default_grasp_object) if default_grasp_object else None
    )
    for raw_line in planning_text.splitlines():
        line = strip_step_prefix(raw_line)
        if not line:
            continue

        command, argument = _split_command(line)
        if command == "GRAB" and argument:
            last_grabbed_object = _clean_object_name(argument)
            continue

        if command not in {"DROP", "PLACE"} or not argument:
            continue

        parsed = _parse_placement_argument(argument, last_grabbed_object)
        if parsed is not None:
            grasp_object, relation, reference_object = parsed
            return SemanticPlacementIntent(
                grasp_object=grasp_object,
                source_line=line,
                relation=relation,
                reference_object=reference_object,
            )

    raise ValueError(
        "No semantic placement action found. Expected a plan line like "
        "'DROP milk carton right to cereal box'."
    )


def last_empower_placement_action(planning_text: str) -> str | None:
    action_line = None
    for raw_line in planning_text.splitlines():
        line = strip_step_prefix(raw_line)
        command = line.split(maxsplit=1)[0].upper() if line.split() else ""
        if command in {"DROP", "PLACE"}:
            action_line = line
    return action_line


def parse_empower_relation(action_line: str) -> str | None:
    normalized = f" {normalize_text_name(action_line)} "
    relation_patterns = (
        ("next_to", " next to "),
        ("right", " right "),
        ("left", " left "),
        ("up", " up "),
        ("on", " on "),
        ("in", " in "),
        ("near", " near "),
        ("beside", " beside "),
        ("with", " with "),
    )
    for relation, token in relation_patterns:
        if token in normalized:
            return relation
    return None


def resolve_empower_reference(
    *,
    action_line: str,
    relation: str | None,
    reference_positions: Mapping[str, np.ndarray],
    grasp_object: str,
) -> tuple[str | None, np.ndarray | None]:
    search_text = reference_search_text(action_line, relation, grasp_object).strip()
    return resolve_reference_position_from_action(
        action_line,
        reference_positions,
        reference_object=search_text or None,
    )


def reference_search_text(
    action_line: str,
    relation: str | None,
    grasp_object: str,
) -> str:
    text = strip_step_prefix(action_line)
    relation_regex = {
        "left": r"\bleft\s+(?:of|to)\s+(.+)$",
        "right": r"\bright\s+(?:of|to)\s+(.+)$",
        "next_to": r"\bnext\s+to\s+(.+)$",
        "near": r"\bnear\s+(.+)$",
        "beside": r"\bbeside\s+(.+)$",
        "with": r"\bwith\s+(.+)$",
        "on": r"\b(?:on|onto)\s+(.+)$",
        "in": r"\b(?:in|into)\s+(.+)$",
        "up": r"\bup\s+(?:of|to)?\s*(.+)$",
    }
    pattern = relation_regex.get(relation or "")
    if pattern:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    cleaned = re.sub(r"^(?:DROP|PLACE)\s+", "", text, flags=re.IGNORECASE)
    grasp = re.escape(grasp_object.strip())
    cleaned = re.sub(grasp, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def matching_references(
    text: str,
    reference_positions: Mapping[str, np.ndarray],
) -> list[tuple[int, str, np.ndarray]]:
    normalized_text = normalize_text_name(text)
    matches = []
    for name, position in reference_positions.items():
        normalized_name = normalize_text_name(name)
        if not normalized_name:
            continue
        if normalized_name in normalized_text or normalized_text in normalized_name:
            matches.append((len(normalized_name), name, position))
    return matches


def apply_relation_offset(
    reference_position: np.ndarray,
    *,
    relation: str | None,
    relation_offset_m: float,
    shelf_board_heights: Sequence[float] | None = None,
) -> np.ndarray:
    coordinate = np.asarray(reference_position, dtype=float).copy()
    offset = abs(float(relation_offset_m))
    if relation == "left":
        coordinate[0] -= offset
    elif relation == "right":
        coordinate[0] += offset
    elif relation == "up":
        coordinate[1] -= offset
    coordinate[2] = _snap_height_to_nearest_board(coordinate[2], shelf_board_heights)
    return coordinate


def update_result_coordinate(result: dict[str, Any], coordinate: np.ndarray) -> None:
    coordinates = [float(coordinate[0]), float(coordinate[1]), float(coordinate[2])]
    result["coordinates"] = coordinates
    result["surface_position"] = {
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
    }
    if "pose" in result:
        result["pose"] = dict(result["pose"])
        result["pose"]["position"] = {
            "x": coordinates[0],
            "y": coordinates[1],
            "z": coordinates[2],
        }


def normalize_text_name(name: str | None) -> str:
    if not name:
        return ""
    normalized = name.strip().lower()
    normalized = re.sub(r"^[0-9]+\)\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def strip_step_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+[\).:-]?\s*", "", line)
    return line.strip()


def _parse_placement_argument(
    argument: str,
    last_grabbed_object: str | None,
) -> tuple[str, str | None, str | None] | None:
    relation_parsed = _parse_relation_argument(argument, last_grabbed_object)
    if relation_parsed is not None:
        return relation_parsed

    match = re.match(
        r"(?P<object>.+?)\s+(?:on|onto|in|into|at|to)\s+.+$",
        argument,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_object_name(match.group("object")), None, None

    if last_grabbed_object:
        return last_grabbed_object, None, None

    return _clean_object_name(argument), None, None


def _parse_relation_argument(
    argument: str,
    last_grabbed_object: str | None,
) -> tuple[str, str, str] | None:
    relation_pattern = r"(?P<relation>left|right)"
    match = re.match(
        rf"(?P<object>.+?)\s+{relation_pattern}\s+(?:of|to\s+)?(?P<reference>.+)$",
        argument,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.match(
            rf"{relation_pattern}\s+(?:of|to\s+)?(?P<reference>.+)$",
            argument,
            flags=re.IGNORECASE,
        )
        if not match or not last_grabbed_object:
            return None
        grasp_object = last_grabbed_object
    else:
        grasp_object = _clean_object_name(match.group("object"))

    reference = _clean_reference_name(match.group("reference"))
    if not reference:
        return None

    return grasp_object, _clean_phrase(match.group("relation")), reference


def extract_labels_per_step(step: str) -> list[str]:
    """Build n-gram labels from a plan step, matching original Empower main."""
    tokens = step.strip().split()
    if len(tokens) <= 1:
        return []
    return _token_ngrams(tokens[1:])


def _token_ngrams(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    labels = [tokens[0]]
    for index in range(len(tokens) - 1):
        labels.append(f"{tokens[index]} {tokens[index + 1]}")
    labels.append(tokens[-1])
    return labels


def reference_names_match(relation_object: str, detection_label: str) -> bool:
    """Match like original main ``find_bb_relation``: label substring of relation object."""
    target = normalize_text_name(relation_object)
    label = normalize_text_name(detection_label)
    if not target or not label:
        return False
    return label in target


def _reference_point(position: Sequence[float]) -> np.ndarray | None:
    point = np.asarray(position, dtype=float)
    if point.shape == (3,) and np.isfinite(point).all():
        return point
    return None


def _match_reference_labels(
    labels: Sequence[str],
    reference_positions_by_name: Mapping[str, Sequence[float]],
) -> tuple[str | None, np.ndarray | None]:
    """Resolve a reference centroid like original main DROP + relabeled detection keys."""
    for label in labels:
        for name, position in reference_positions_by_name.items():
            if label == name:
                point = _reference_point(position)
                if point is not None:
                    return name, point

        candidates: list[tuple[int, str, np.ndarray]] = []
        label_lower = label.lower()
        for name, position in reference_positions_by_name.items():
            if label_lower not in name.lower():
                continue
            point = _reference_point(position)
            if point is None:
                continue
            candidates.append((len(name), name, point))

        if candidates:
            _, name, point = max(candidates, key=lambda item: (item[0], item[1]))
            return name, point

    return None, None


def resolve_reference_position_from_action(
    action_line: str,
    reference_positions_by_name: Mapping[str, Sequence[float]],
    *,
    reference_object: str | None = None,
) -> tuple[str | None, np.ndarray | None]:
    """Resolve a reference centroid from relabeled detection keys and plan n-grams."""
    if not reference_positions_by_name:
        return None, None

    label_groups: list[list[str]] = []
    if reference_object and reference_object.strip():
        label_groups.append(_token_ngrams(reference_object.strip().split()))
    label_groups.append(extract_labels_per_step(action_line))

    for labels in label_groups:
        matched_name, point = _match_reference_labels(labels, reference_positions_by_name)
        if point is not None:
            return matched_name, point

    return None, None


def _apply_reference_relation(
    result: dict[str, Any],
    *,
    reference_position: np.ndarray,
    relation: str | None,
    relation_offset_m: float,
    shelf_board_heights: Sequence[float] | None = None,
) -> dict[str, Any]:
    updated = dict(result)
    offset = abs(float(relation_offset_m))

    if relation == "left":
        direction = -1.0
    elif relation == "right":
        direction = 1.0
    else:
        return updated

    fallback_height = _result_surface_height(result)
    surface = _candidate_surface_point(
        reference_position=reference_position,
        direction=direction,
        preferred_offset_m=offset,
        fallback_height=fallback_height,
        shelf_board_heights=shelf_board_heights,
    )
    coordinates = [float(surface[0]), float(surface[1]), float(surface[2])]
    updated["coordinates"] = coordinates
    updated["surface_position"] = {
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
    }
    updated["pose"] = dict(result["pose"])
    updated["pose"]["position"] = {
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
    }
    return updated


def _candidate_surface_point(
    *,
    reference_position: np.ndarray,
    direction: float,
    preferred_offset_m: float,
    fallback_height: float,
    shelf_board_heights: Sequence[float] | None = None,
) -> np.ndarray:
    return np.array(
        [
            float(reference_position[0] + direction * preferred_offset_m),
            float(reference_position[1]),
            _snap_height_to_nearest_board(reference_position[2], shelf_board_heights, fallback_height),
        ],
        dtype=float,
    )


def _result_surface_height(result: Mapping[str, Any]) -> float:
    surface = result.get("surface_position")
    if isinstance(surface, Mapping) and "z" in surface:
        return float(surface["z"])
    coordinates = result.get("coordinates")
    if coordinates is not None:
        return float(np.asarray(coordinates, dtype=float).reshape(-1)[2])
    return 0.0


def _estimate_surface_xy(
    points: np.ndarray,
    surface_height: float,
    *,
    initial_band_m: float = 0.015,
    grid_resolution_m: float = 0.03,
    min_points_per_cell: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    last_band_points = np.empty((0, 3), dtype=float)
    for band in _band_schedule(initial_band_m):
        band_points = points[np.abs(points[:, 2] - surface_height) <= band]
        last_band_points = band_points
        if band_points.shape[0] >= 50:
            break

    if last_band_points.shape[0] == 0:
        raise ValueError(
            f"No pointcloud support points found near surface height {surface_height:.3f} m"
        )

    support_points = _largest_xy_component(
        last_band_points,
        grid_resolution_m=grid_resolution_m,
        min_points_per_cell=min_points_per_cell,
    )
    return _robust_bbox_center(support_points[:, :2]), support_points


def _select_surface_height(
    points: np.ndarray,
    *,
    placement_surface_height_m: float | None,
    shelf_board_heights: np.ndarray | None,
) -> float:
    if placement_surface_height_m is not None:
        return _coerce_height(placement_surface_height_m, "placement_surface_height_m")
    if shelf_board_heights is not None and shelf_board_heights.size:
        return _select_shelf_height_with_support(points, shelf_board_heights)
    return _infer_default_surface_height(points)


def _select_shelf_height_with_support(points: np.ndarray, shelf_board_heights: np.ndarray) -> float:
    z = points[:, 2]
    z = z[np.isfinite(z)]
    if z.shape[0] == 0:
        raise ValueError("placement_pointclouds contains no finite z values")

    best_height = float(shelf_board_heights[0])
    best_count = -1
    for height in shelf_board_heights:
        count = int(np.count_nonzero(np.abs(z - float(height)) <= 0.04))
        if count > best_count:
            best_height = float(height)
            best_count = count
    return best_height


def _infer_default_surface_height(points: np.ndarray) -> float:
    z = points[:, 2]
    z = z[np.isfinite(z)]
    if z.shape[0] == 0:
        raise ValueError("placement_pointclouds contains no finite z values")

    bin_width_m = 0.01
    z_min = float(np.quantile(z, 0.01))
    z_max = float(np.quantile(z, 0.99))
    if z_max <= z_min:
        return float(np.median(z))

    bins = np.arange(z_min, z_max + bin_width_m, bin_width_m)
    counts, edges = np.histogram(z, bins=bins)
    if counts.size < 3:
        return float(np.median(z))

    centers = (edges[:-1] + edges[1:]) / 2.0
    smoothed = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    min_peak_count = max(100, int(0.002 * z.shape[0]))

    for idx in range(1, smoothed.shape[0] - 1):
        if (
            smoothed[idx] >= smoothed[idx - 1]
            and smoothed[idx] >= smoothed[idx + 1]
            and smoothed[idx] >= min_peak_count
        ):
            return float(centers[idx])

    raise ValueError("Cannot infer a placement surface from the supplied point cloud")


def _coerce_pointclouds(pointclouds: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    if isinstance(pointclouds, np.ndarray):
        point_sets = [_coerce_xyz_points(pointclouds, "placement_pointclouds")]
    else:
        point_sets = [
            _coerce_xyz_points(pointcloud, "placement_pointclouds")
            for pointcloud in pointclouds
        ]

    if not point_sets:
        raise ValueError("placement_pointclouds cannot be empty")

    return np.vstack(point_sets)


def _coerce_xyz_points(points: np.ndarray, name: str) -> np.ndarray:
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{name} must be an array with shape (N, 3+) coordinates")

    xyz = xyz[:, :3]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] == 0:
        raise ValueError(f"{name} contains no finite XYZ points")

    return xyz


def _coerce_height(value: float, name: str) -> float:
    height = float(value)
    if not math.isfinite(height):
        raise ValueError(f"{name} must be finite")
    return height


def _coerce_optional_heights(values: Sequence[float] | None, name: str) -> np.ndarray | None:
    if values is None:
        return None
    heights = np.asarray(values, dtype=float).reshape(-1)
    if heights.size == 0:
        return None
    if not np.isfinite(heights).all():
        raise ValueError(f"{name} must contain only finite values")
    return heights


def _snap_height_to_nearest_board(
    height: float,
    shelf_board_heights: Sequence[float] | None,
    fallback_height: float | None = None,
) -> float:
    heights = _coerce_optional_heights(shelf_board_heights, "shelf_board_heights")
    if heights is None:
        return float(height if fallback_height is None else fallback_height)
    idx = int(np.argmin(np.abs(heights - float(height))))
    return float(heights[idx])


def _coerce_rpy(orientation_rpy: Sequence[float]) -> tuple[float, float, float]:
    rpy = np.asarray(orientation_rpy, dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("orientation_rpy must be finite (roll, pitch, yaw)")
    return tuple(float(value) for value in rpy)


def _band_schedule(initial_band: float) -> tuple[float, ...]:
    bands = [initial_band, 0.025, 0.04, 0.06]
    return tuple(dict.fromkeys(float(max(0.001, band)) for band in bands))


def _largest_xy_component(
    band_points: np.ndarray,
    *,
    grid_resolution_m: float,
    min_points_per_cell: int,
) -> np.ndarray:
    xy = band_points[:, :2]
    xy_min = xy.min(axis=0)
    cells = np.floor((xy - xy_min) / grid_resolution_m).astype(np.int64)

    counts: dict[tuple[int, int], int] = {}
    for cell_x, cell_y in cells:
        key = (int(cell_x), int(cell_y))
        counts[key] = counts.get(key, 0) + 1

    occupied = {
        key: count
        for key, count in counts.items()
        if count >= min_points_per_cell
    }
    if not occupied:
        return band_points

    seen: set[tuple[int, int]] = set()
    best_component: set[tuple[int, int]] = set()
    best_score = (-1, -1)

    for start in occupied:
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        component: set[tuple[int, int]] = set()
        point_count = 0

        while stack:
            cell = stack.pop()
            component.add(cell)
            point_count += occupied[cell]
            cell_x, cell_y = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (cell_x + dx, cell_y + dy)
                    if neighbor in occupied and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

        score = (len(component), point_count)
        if score > best_score:
            best_component = component
            best_score = score

    component_mask = np.fromiter(
        ((int(cell[0]), int(cell[1])) in best_component for cell in cells),
        dtype=bool,
        count=cells.shape[0],
    )
    support_points = band_points[component_mask]
    return support_points if support_points.shape[0] else band_points


def _robust_bbox_center(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] == 1:
        return xy[0]

    lower = np.quantile(xy, 0.05, axis=0)
    upper = np.quantile(xy, 0.95, axis=0)
    center = (lower + upper) / 2.0

    if not np.isfinite(center).all():
        center = np.median(xy, axis=0)

    return center.astype(float)


def _support_bounds(points: np.ndarray) -> dict[str, list[float]]:
    if points.shape[0] == 0:
        return {"min": [], "max": [], "p05": [], "p95": []}

    return {
        "min": _float_list(np.min(points, axis=0)),
        "max": _float_list(np.max(points, axis=0)),
        "p05": _float_list(np.quantile(points, 0.05, axis=0)),
        "p95": _float_list(np.quantile(points, 0.95, axis=0)),
    }


def _float_list(values: np.ndarray) -> list[float]:
    return [
        float(value) if math.isfinite(float(value)) else float("nan")
        for value in values
    ]


def _split_command(line: str) -> tuple[str, str]:
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    command = parts[0].upper().rstrip(":")
    argument = parts[1].strip() if len(parts) > 1 else ""
    return command, argument


def _clean_object_name(value: str) -> str:
    value = _clean_phrase(value)
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    return value


def _clean_reference_name(value: str) -> str:
    value = _clean_phrase(value)
    value = re.sub(r"^(?:the|a|an|other)\s+", "", value)
    value = re.sub(r"^(?:the\s+)?other\s+", "", value)
    return value


def _normalize_result_name(grasp_object: str) -> str:
    value = grasp_object.strip().lower()
    value = re.sub(r"^[0-9]+[_\-\s]+", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _clean_phrase(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[\"'`]+|[\"'`.,;:]+$", "", value)
    value = re.sub(r"[_\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_reference_key(value: str) -> str:
    normalized = _normalize_key(value)
    return re.sub(r"(?:_\d+)+$", "", normalized)


def _xyz_from_result_value(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping):
        if not all(key in value for key in ("x", "y", "z")):
            return None
        value = [value["x"], value["y"], value["z"]]
    try:
        xyz = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if xyz.shape[0] != 3 or not np.isfinite(xyz).all():
        return None
    return xyz


def _shelf_board_id_for_snapped_height(
    height: float,
    shelf_board_heights: Sequence[float] | None,
) -> int | None:
    heights = _coerce_optional_heights(shelf_board_heights, "shelf_board_heights")
    if heights is None:
        return None
    return int(np.argmin(np.abs(heights - float(height))))


def _normalize_placement_direction(relation: str | None) -> str | None:
    relation = str(relation or "").strip().lower()
    if relation in {"left", "left_of"}:
        return "left"
    if relation in {"right", "right_of"}:
        return "right"
    return None


def _infer_placement_direction_from_offset(
    target_xyz: np.ndarray,
    reference_xyz: np.ndarray,
    *,
    axis: int = 0,
    epsilon: float = 1e-6,
) -> str | None:
    delta = float(target_xyz[axis] - reference_xyz[axis])
    if delta > epsilon:
        return "right"
    if delta < -epsilon:
        return "left"
    return None


def _snap_xyz_to_shelf_board(
    xyz: np.ndarray,
    shelf_board_heights: Sequence[float] | None,
) -> np.ndarray:
    snapped = np.asarray(xyz, dtype=float).copy()
    snapped[2] = _snap_height_to_nearest_board(float(xyz[2]), shelf_board_heights)
    return snapped


def _write_snapped_reference(result: dict[str, Any], reference_xyz: np.ndarray) -> None:
    result["reference_coordinates"] = reference_xyz.tolist()
    result["reference_position"] = {
        "x": float(reference_xyz[0]),
        "y": float(reference_xyz[1]),
        "z": float(reference_xyz[2]),
    }


def annotate_semantic_placement_context(
    result: dict[str, Any] | None,
    *,
    shelf_board_heights: Sequence[float] | None = None,
) -> None:
    """Add shelf-snapped coordinates and side-aware placement metadata."""

    if not isinstance(result, dict):
        return

    shelf_heights = _coerce_optional_heights(shelf_board_heights, "shelf_board_heights")

    target_xyz = _xyz_from_result_value(result.get("coordinates"))
    reference_xyz = _xyz_from_result_value(result.get("reference_position"))
    if reference_xyz is None:
        reference_xyz = _xyz_from_result_value(result.get("reference_coordinates"))

    if shelf_heights is not None:
        if target_xyz is not None:
            target_xyz = _snap_xyz_to_shelf_board(target_xyz, shelf_board_heights)
            update_result_coordinate(result, target_xyz)
        if reference_xyz is not None:
            reference_xyz = _snap_xyz_to_shelf_board(reference_xyz, shelf_board_heights)
            _write_snapped_reference(result, reference_xyz)

    result["coordinate_origin"] = "object_bottom"

    if target_xyz is not None:
        shelf_board_id = _shelf_board_id_for_snapped_height(target_xyz[2], shelf_board_heights)
        if shelf_board_id is not None:
            result["shelf_board_id"] = shelf_board_id
            result["shelf_id"] = f"shelf_{shelf_board_id}"

    if reference_xyz is not None:
        result["reference_coordinates"] = reference_xyz.tolist()
        reference_shelf_board_id = _shelf_board_id_for_snapped_height(
            reference_xyz[2],
            shelf_board_heights,
        )
        if reference_shelf_board_id is not None:
            result["reference_shelf_board_id"] = reference_shelf_board_id

    placement_direction = _normalize_placement_direction(result.get("relation"))
    if placement_direction is None and target_xyz is not None and reference_xyz is not None:
        placement_direction = _infer_placement_direction_from_offset(target_xyz, reference_xyz)
    if placement_direction is not None:
        result["placement_direction"] = placement_direction


__all__ = [
    "SemanticPlacementIntent",
    "annotate_semantic_placement_context",
    "apply_relation_offset",
    "extract_labels_per_step",
    "get_empower_style_semantic_coordinates",
    "get_semantic_placement_coordinates",
    "get_semantic_placement_coordinates_from_plan",
    "is_semantic_placement_success",
    "last_empower_placement_action",
    "matching_references",
    "normalize_text_name",
    "parse_empower_relation",
    "parse_semantic_placement_plan",
    "reference_names_match",
    "reference_not_found_failure_reason",
    "reference_search_text",
    "resolve_empower_reference",
    "resolve_reference_position_from_action",
    "strip_step_prefix",
    "update_result_coordinate",
]
