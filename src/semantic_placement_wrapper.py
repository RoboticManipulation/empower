"""Single-call semantic placement wrapper for Empower.

The public entry point is ``run_semantic_placement``. It stages one placement
scene, runs Empower's existing LLM + SAM grounding path through ``Detection``,
and returns placement coordinates without ROS, MoveIt, CuRoBo, sockets, or
multiple terminals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np


USE_CASE = "semantic_placement"
DEFAULT_FRAME_ID = "gemini336_color_optical_frame"
DEFAULT_RELATION_OFFSET_M = 0.15


def run_semantic_placement(
    *,
    grasp_object: str,
    image_path: str | os.PathLike[str],
    pointcloud_path: str | os.PathLike[str],
    camera_info_path: str | os.PathLike[str] | None = None,
    frame_id: str = DEFAULT_FRAME_ID,
    use_case: str = USE_CASE,
    images_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run semantic placement from Python and return the coordinate result."""

    if not grasp_object or not str(grasp_object).strip():
        raise ValueError("grasp_object is required for semantic placement")

    image_path = _existing_path(image_path, "image_path")
    pointcloud_path = _existing_path(pointcloud_path, "pointcloud_path")
    camera_info = _resolve_camera_info_path(
        camera_info_path=camera_info_path,
        image_path=image_path,
        pointcloud_path=pointcloud_path,
    )

    scan_dir, dump_dir = _stage_semantic_placement_inputs(
        use_case=use_case,
        image_path=image_path,
        pointcloud_path=pointcloud_path,
        camera_info_path=camera_info,
        grasp_object=grasp_object,
        images_root=images_root,
        output_root=output_root,
    )

    loader_instance = _build_semantic_loader(
        use_case=use_case,
        scan_dir=scan_dir,
        dump_dir=dump_dir,
        grasp_object=str(grasp_object).strip(),
        frame_id=frame_id,
    )

    _ensure_src_on_path()
    from detection import Detection

    detection_instance = Detection()
    detection_instance.set_loader(loader_instance)
    return detection_instance.semantic_placement_result


def get_semantic_placement_coordinates_from_plan(
    planning_text: str,
    *,
    placement_pointclouds: np.ndarray | Sequence[np.ndarray],
    grasp_object: str | None = None,
    placement_surface_height_m: float | None = None,
    reference_positions_by_name: Mapping[str, Sequence[float]] | None = None,
    relation_offset_m: float = DEFAULT_RELATION_OFFSET_M,
    orientation_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    frame_id: str = DEFAULT_FRAME_ID,
) -> dict[str, Any]:
    """Convert an Empower LLM plan into semantic placement coordinates."""

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
    )

    if intent.reference_object and reference_positions_by_name:
        reference_position = _resolve_reference_position(
            intent.reference_object,
            reference_positions_by_name,
        )
        if reference_position is not None:
            result = _apply_reference_relation(
                result,
                reference_position=reference_position,
                relation=intent.relation,
                relation_offset_m=relation_offset_m,
            )

    return result


def get_semantic_placement_coordinates(
    grasp_object: str,
    *,
    placement_pointclouds: np.ndarray | Sequence[np.ndarray],
    placement_surface_height_m: float | None = None,
    orientation_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    frame_id: str = DEFAULT_FRAME_ID,
) -> dict[str, Any]:
    """Return a surface placement coordinate from loaded point-cloud data."""

    points = _coerce_pointclouds(placement_pointclouds)
    surface_height = (
        _coerce_height(placement_surface_height_m, "placement_surface_height_m")
        if placement_surface_height_m is not None
        else _infer_default_surface_height(points)
    )
    surface_xy, support_points = _estimate_surface_xy(points, surface_height)
    coordinates = [float(surface_xy[0]), float(surface_xy[1]), float(surface_height)]
    roll, pitch, yaw = _coerce_rpy(orientation_rpy)

    return {
        "use_case": USE_CASE,
        "grasp_object": grasp_object,
        "normalized_grasp_object": _normalize_object_name(grasp_object),
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


@dataclass(frozen=True)
class SemanticPlacementIntent:
    """Semantic placement action extracted from an Empower action plan."""

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
        line = _strip_step_prefix(raw_line)
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


def _resolve_reference_position(
    reference_object: str,
    reference_positions_by_name: Mapping[str, Sequence[float]],
) -> np.ndarray | None:
    normalized_reference = _normalize_key(reference_object)

    for name, position in reference_positions_by_name.items():
        normalized_name = _normalize_key(name)
        if normalized_name == normalized_reference:
            point = np.asarray(position, dtype=float)
            if point.shape == (3,) and np.isfinite(point).all():
                return point

    base_reference = _normalize_reference_key(reference_object)
    for name, position in reference_positions_by_name.items():
        normalized_name = _normalize_reference_key(name)
        if normalized_name == base_reference:
            point = np.asarray(position, dtype=float)
            if point.shape == (3,) and np.isfinite(point).all():
                return point
    return None


def _apply_reference_relation(
    result: dict[str, Any],
    *,
    reference_position: np.ndarray,
    relation: str | None,
    relation_offset_m: float,
) -> dict[str, Any]:
    updated = dict(result)
    surface_position = dict(result["surface_position"])
    offset = abs(float(relation_offset_m))

    if relation == "left":
        direction = -1.0
    elif relation == "right":
        direction = 1.0
    else:
        return updated

    surface = _candidate_surface_point(
        reference_position=reference_position,
        surface_z=float(surface_position["z"]),
        direction=direction,
        preferred_offset_m=offset,
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
    surface_z: float,
    direction: float,
    preferred_offset_m: float,
) -> np.ndarray:
    return np.array(
        [
            float(reference_position[0] + direction * preferred_offset_m),
            float(reference_position[1]),
            surface_z,
        ],
        dtype=float,
    )


def _stage_semantic_placement_inputs(
    *,
    use_case: str,
    image_path: Path,
    pointcloud_path: Path,
    camera_info_path: Path | None,
    grasp_object: str,
    images_root: str | os.PathLike[str] | None,
    output_root: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    scan_root, dump_root = _resolve_empower_roots(
        images_root=images_root,
        output_root=output_root,
    )
    scan_dir = scan_root / use_case
    dump_dir = dump_root / use_case
    scan_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    _write_scan_image(image_path, scan_dir / "scan.jpg")
    shutil.copy2(pointcloud_path, dump_dir / "depth_pointcloud.pcd")

    staged_camera_info = dump_dir / "camera_info.json"
    if camera_info_path is not None:
        shutil.copy2(camera_info_path, staged_camera_info)
    elif staged_camera_info.exists():
        staged_camera_info.unlink()

    with open(dump_dir / "grasp_object.txt", "w") as grasp_file:
        grasp_file.write(str(grasp_object).strip())

    return scan_dir, dump_dir


def _resolve_empower_roots(
    *,
    images_root: str | os.PathLike[str] | None,
    output_root: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    if images_root is None or output_root is None:
        from paths import IMAGES_DIR, OUTPUT_DIR

        default_images_root = Path(IMAGES_DIR)
        default_output_root = Path(OUTPUT_DIR)
    else:
        default_images_root = Path(images_root)
        default_output_root = Path(output_root)

    return (
        Path(images_root) if images_root is not None else default_images_root,
        Path(output_root) if output_root is not None else default_output_root,
    )


def _write_scan_image(source_path: Path, destination_path: Path) -> None:
    import cv2

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Unable to read placement image: {source_path}")
    if not cv2.imwrite(str(destination_path), image):
        raise ValueError(f"Unable to write staged scan image: {destination_path}")


def _build_semantic_loader(
    *,
    use_case: str,
    scan_dir: Path,
    dump_dir: Path,
    grasp_object: str,
    frame_id: str,
):
    _ensure_src_on_path()
    import loader

    loader_instance = loader.Loader(use_case)
    loader_instance.use_case = use_case
    loader_instance.SCAN_DIR = str(scan_dir) + os.sep
    loader_instance.DUMP_DIR = str(dump_dir) + os.sep
    loader_instance.grasp_object = grasp_object
    loader_instance.semantic_frame_id = frame_id
    return loader_instance


def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _resolve_camera_info_path(
    *,
    camera_info_path: str | os.PathLike[str] | None,
    image_path: Path,
    pointcloud_path: Path,
) -> Path | None:
    if camera_info_path is not None:
        return _existing_path(camera_info_path, "camera_info_path")

    return None


def _existing_path(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser()
    if not value.exists():
        raise FileNotFoundError(f"{name} does not exist: {value}")
    return value


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


def _strip_step_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+[\).:-]?\s*", "", line)
    return line.strip()


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


def _normalize_object_name(grasp_object: str) -> str:
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


__all__ = [
    "DEFAULT_RELATION_OFFSET_M",
    "DEFAULT_FRAME_ID",
    "USE_CASE",
    "run_semantic_placement",
]
