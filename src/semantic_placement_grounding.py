"""Ground semantic placement plans against SAM masks and point clouds."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

import cv2
import numpy as np

from semantic_placement_wrapper import get_semantic_placement_coordinates_from_plan
from semantic_placement_wrapper import parse_semantic_placement_plan


def run_grounded_semantic_placement(
    *,
    loader_instance: Any,
    results_multi: Mapping[str, str],
    detections: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return semantic placement coordinates using Empower's LLM + SAM output."""

    pcd_path = os.path.join(loader_instance.DUMP_DIR, "depth_pointcloud.pcd")
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(
            "Semantic placement requires a point cloud at "
            f"{pcd_path}. Run prepare_local_data.py or provide "
            "output/semantic_placement/depth_pointcloud.pcd first."
        )

    placement_pointcloud = load_pointcloud_points(pcd_path)
    reference_positions = get_semantic_reference_geometry(
        loader_instance=loader_instance,
        detections=detections,
        placement_pointcloud=placement_pointcloud,
    )
    grasp_object = get_semantic_grasp_object(loader_instance, required=True)
    planning_text = plan_with_same_type_reference(
        loader_instance=loader_instance,
        planning_text=results_multi.get("planning_agent_info", ""),
        grasp_object=grasp_object,
        reference_positions=reference_positions,
    )

    result = get_semantic_placement_coordinates_from_plan(
        planning_text,
        placement_pointclouds=placement_pointcloud,
        grasp_object=grasp_object,
        reference_positions_by_name=reference_positions,
        frame_id=semantic_frame_id(loader_instance),
    )

    result_path = os.path.join(
        loader_instance.DUMP_DIR,
        "semantic_placement_result.json",
    )
    with open(result_path, "w") as f:
        json.dump(json_ready(result), f, indent=2)

    print(f"[OK] Semantic placement result -> {result_path}")
    print(f"[OK] Semantic placement coordinates: {result['coordinates']}")
    return result


def get_semantic_grasp_object(loader_instance: Any, required: bool = False) -> str | None:
    candidates = [
        getattr(loader_instance, "grasp_object", None),
        os.environ.get("EMPOWER_GRASP_OBJECT"),
        os.environ.get("GRASP_OBJECT"),
    ]

    for path in (
        os.path.join(loader_instance.DUMP_DIR, "grasp_object.txt"),
        os.path.join(loader_instance.SCAN_DIR, "grasp_object.txt"),
    ):
        if os.path.exists(path):
            with open(path) as f:
                candidates.append(f.read())

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()

    if required:
        raise ValueError(
            "semantic_placement requires the already-grasped object name. "
            "Set EMPOWER_GRASP_OBJECT, pass it as the second argument to "
            "models_cacher.py, or write grasp_object.txt in the use-case "
            "scan/output directory."
        )
    return None


def semantic_placement_task_description(grasp_object: str) -> str:
    return (
        f"The robot is already holding this object: {grasp_object}. "
        "The held object is not visible in the image. Use the image only as "
        "the placement scene. Choose exactly one visible reference object "
        "where the held object semantically belongs, such as placing a carton "
        "next to one visible carton or a condiment next to one visible "
        "condiment. If a visible object has the same type or name as the held "
        "object, use one same-type object as the reference before choosing any "
        "other category. Choose only a LEFT or RIGHT placement relative to "
        "that single visible reference object, and choose the side that has "
        "open shelf space in the image. Do not use behind, in front of, near, "
        "beside, into, on, group, area, shelf, or category as the placement "
        "relation/reference. Return exactly one action line such as "
        f"'DROP {grasp_object} left to cereal box' or 'DROP {grasp_object} "
        "right to cereal box'."
    )


def semantic_placement_prompt_objects(
    *,
    planning_text: str,
    grasp_object: str | None,
) -> list[str]:
    objects = []
    if grasp_object:
        objects.append(grasp_object)

    try:
        intent = parse_semantic_placement_plan(
            planning_text,
            default_grasp_object=grasp_object,
        )
    except ValueError:
        return _unique_objects(objects)

    objects.append(intent.grasp_object)
    if intent.reference_object:
        objects.append(intent.reference_object)

    return _unique_objects(objects)


def _unique_objects(objects: list[str]) -> list[str]:
    unique = []
    seen = set()
    for object_name in objects:
        if not object_name or not str(object_name).strip():
            continue
        key = normalize_object_name(str(object_name))
        if key and key not in seen:
            seen.add(key)
            unique.append(str(object_name).strip())
    return unique


def semantic_frame_id(loader_instance: Any) -> str:
    return (
        getattr(loader_instance, "semantic_frame_id", None)
        or os.environ.get(
            "EMPOWER_SEMANTIC_FRAME_ID",
            "gemini336_color_optical_frame",
        )
    )


def plan_with_same_type_reference(
    *,
    loader_instance: Any,
    planning_text: str,
    grasp_object: str,
    reference_positions: Mapping[str, np.ndarray],
) -> str:
    same_type_reference = select_same_type_semantic_reference(
        loader_instance=loader_instance,
        grasp_object=grasp_object,
        reference_positions=reference_positions,
    )
    if same_type_reference is None:
        return planning_text

    reference_name, relation = same_type_reference
    print(
        "[DEBUG] Semantic placement same-type reference: "
        f"DROP {grasp_object} {relation} to {reference_name}"
    )
    return f"DROP {grasp_object} {relation} to {reference_name}"


def select_same_type_semantic_reference(
    *,
    loader_instance: Any,
    grasp_object: str,
    reference_positions: Mapping[str, np.ndarray],
) -> tuple[str, str] | None:
    image = cv2.imread(loader_instance.SCAN_DIR + "scan.jpg")
    intrinsics = load_camera_intrinsics(loader_instance)
    if image is None or intrinsics is None:
        return None

    image_shape = image.shape[:2]
    image_center_x = image_shape[1] / 2.0
    candidates = []
    for name, position in reference_positions.items():
        if not semantic_reference_matches_grasp(name, grasp_object):
            continue

        pixel = project_semantic_point_to_image(position, intrinsics, image_shape)
        if pixel is None:
            continue

        u, _ = pixel
        relation = "left" if u >= image_center_x else "right"
        score = abs(u - image_center_x)
        candidates.append((score, name, relation))

    if not candidates:
        return None

    _, name, relation = max(candidates, key=lambda item: item[0])
    return name, relation


def semantic_reference_matches_grasp(reference_name: str, grasp_object: str) -> bool:
    reference = normalize_object_name(reference_name)
    reference = re.sub(r"\s+\d+$", "", reference)
    grasp = normalize_object_name(grasp_object)
    return reference == grasp


def load_pointcloud_points(pcd_path: str) -> np.ndarray:
    import open3d as o3d

    pointcloud = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pointcloud.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"No valid XYZ points found in {pcd_path}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"No finite XYZ points found in {pcd_path}")
    return points


def get_semantic_reference_geometry(
    *,
    loader_instance: Any,
    detections: Mapping[int, Mapping[str, Any]],
    placement_pointcloud: np.ndarray,
) -> dict[str, np.ndarray]:
    image = cv2.imread(loader_instance.SCAN_DIR + "scan.jpg")
    if image is None:
        print("[WARN] Unable to read scan image; semantic references disabled.")
        return {}

    intrinsics = load_camera_intrinsics(loader_instance)
    if intrinsics is None:
        print("[WARN] camera_info.json missing; semantic references disabled.")
        return {}

    projection = project_pointcloud_to_image(
        placement_pointcloud,
        intrinsics,
        image.shape[:2],
    )
    if projection is None:
        print("[WARN] Point cloud does not project into the image.")
        return {}

    reference_positions: dict[str, np.ndarray] = {}
    label_counts: dict[str, int] = {}
    min_points = int(os.environ.get("EMPOWER_SEMANTIC_MIN_OBJECT_POINTS", "25"))

    for detection in detections.values():
        label = detection.get("label")
        mask = detection.get("mask")
        if not label or mask is None:
            continue

        object_points = points_for_detection_mask(
            placement_pointcloud,
            projection,
            mask,
        )
        if len(object_points) < min_points:
            continue

        centroid = summarize_object_points(object_points)
        if centroid is None:
            continue

        normalized_label = normalize_object_name(label)
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1
        occurrence = label_counts[normalized_label]

        names = [label]
        if occurrence > 1:
            names.extend([f"{label} {occurrence}", f"{label}_{occurrence}"])

        for name in names:
            reference_positions.setdefault(name, centroid)

    return reference_positions


def load_camera_intrinsics(loader_instance: Any) -> dict[str, float] | None:
    cam_path = os.path.join(loader_instance.DUMP_DIR, "camera_info.json")
    if not os.path.exists(cam_path):
        return None

    with open(cam_path) as f:
        info = json.load(f)

    if "K" in info:
        return {
            "fx": float(info["K"][0]),
            "fy": float(info["K"][4]),
            "cx": float(info["K"][2]),
            "cy": float(info["K"][5]),
        }

    if "camera_matrix" in info and "data" in info["camera_matrix"]:
        matrix = info["camera_matrix"]["data"]
        return {
            "fx": float(matrix[0]),
            "fy": float(matrix[4]),
            "cx": float(matrix[2]),
            "cy": float(matrix[5]),
        }

    required = ("fx", "fy", "cx", "cy")
    if all(key in info for key in required):
        return {key: float(info[key]) for key in required}

    raise ValueError(
        f"{cam_path} must contain fx/fy/cx/cy, K, or camera_matrix.data"
    )


def project_pointcloud_to_image(
    points: np.ndarray,
    intrinsics: Mapping[str, float],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    height, width = image_shape
    best_projection = None
    best_count = 0

    for z_sign in (1.0, -1.0):
        z = points[:, 2] * z_sign
        valid_z = np.abs(z) > 1e-9
        u = np.zeros(len(points), dtype=np.int32)
        v = np.zeros(len(points), dtype=np.int32)

        u_float = (
            intrinsics["fx"] * points[:, 0] / np.where(valid_z, z, 1.0)
            + intrinsics["cx"]
        )
        v_float = (
            intrinsics["fy"] * points[:, 1] / np.where(valid_z, z, 1.0)
            + intrinsics["cy"]
        )
        finite = valid_z & np.isfinite(u_float) & np.isfinite(v_float)
        u[finite] = np.rint(u_float[finite]).astype(np.int32)
        v[finite] = np.rint(v_float[finite]).astype(np.int32)
        inside = finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        inside_count = int(inside.sum())

        if inside_count > best_count:
            best_count = inside_count
            best_projection = {
                "u": u,
                "v": v,
                "inside": inside,
            }

    return best_projection


def points_for_detection_mask(
    points: np.ndarray,
    projection: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    mask_2d = np.asarray(mask)
    if mask_2d.ndim == 3:
        mask_2d = np.any(mask_2d > 0, axis=2)
    elif mask_2d.ndim == 2:
        mask_2d = mask_2d > 0
    else:
        return np.empty((0, 3), dtype=float)

    valid_indices = np.flatnonzero(projection["inside"])
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=float)

    u = projection["u"][valid_indices]
    v = projection["v"][valid_indices]
    hits = valid_indices[mask_2d[v, u]]

    flipped_mask = cv2.flip(mask_2d.astype(np.uint8), 1).astype(bool)
    flipped_hits = valid_indices[flipped_mask[v, u]]
    if len(flipped_hits) > len(hits):
        hits = flipped_hits

    return points[hits]


def summarize_object_points(object_points: np.ndarray) -> np.ndarray | None:
    object_points = np.asarray(object_points, dtype=float)
    object_points = object_points[np.isfinite(object_points).all(axis=1)]
    if len(object_points) == 0:
        return None

    median = np.median(object_points, axis=0)
    distances = np.linalg.norm(object_points - median, axis=1)
    if len(distances) > 10:
        cutoff = np.quantile(distances, 0.85)
        object_points = object_points[distances <= cutoff]

    if len(object_points) == 0:
        return None

    return np.mean(object_points, axis=0)


def project_semantic_point_to_image(
    point: np.ndarray,
    intrinsics: Mapping[str, float],
    image_shape: tuple[int, int],
) -> tuple[int, int] | None:
    height, width = image_shape
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None

    for z_sign in (1.0, -1.0):
        z = point[2] * z_sign
        if abs(z) <= 1e-9:
            continue
        u = intrinsics["fx"] * point[0] / z + intrinsics["cx"]
        v = intrinsics["fy"] * point[1] / z + intrinsics["cy"]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        u_int = int(round(u))
        v_int = int(round(v))
        if 0 <= u_int < width and 0 <= v_int < height:
            return u_int, v_int

    return None


def normalize_object_name(name: str | None) -> str:
    if not name:
        return ""
    normalized = name.strip().lower()
    normalized = re.sub(r"^[0-9]+\)\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
