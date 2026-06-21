"""Reference-object geometry from detector masks and point clouds."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import cv2
import numpy as np

from semantic_placement_geometry import normalize_text_name
from utils.common_utils import read_json


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

    projection_points = projection_pointcloud_points(loader_instance, placement_pointcloud)
    if projection_points is None:
        return {}

    projection = project_pointcloud_to_image(
        projection_points,
        intrinsics,
        image.shape[:2],
    )
    if projection is None:
        print("[WARN] Point cloud does not project into the image.")
        return {}

    reference_positions: dict[str, np.ndarray] = {}
    reference_quality: dict[str, tuple[float, int]] = {}
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

        normalized_label = normalize_text_name(label)
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1
        occurrence = label_counts[normalized_label]

        names = [label]
        if occurrence > 1:
            names.extend([f"{label} {occurrence}", f"{label}_{occurrence}"])

        quality = (float(detection.get("score", 0.0) or 0.0), int(len(object_points)))
        for name in names:
            if quality > reference_quality.get(name, (-1.0, -1)):
                reference_positions[name] = centroid
                reference_quality[name] = quality

    return reference_positions


def load_camera_intrinsics(loader_instance: Any) -> dict[str, float] | None:
    cam_path = os.path.join(loader_instance.DUMP_DIR, "camera_info.json")
    if not os.path.exists(cam_path):
        return None

    info = read_json(cam_path)

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


def pointcloud_origin(loader_instance: Any) -> str:
    origin = getattr(loader_instance, "semantic_pointcloud_origin", "camera")
    return str(origin).strip().lower() or "camera"


def projection_pointcloud_points(
    loader_instance: Any,
    placement_pointcloud: np.ndarray,
) -> np.ndarray | None:
    origin = pointcloud_origin(loader_instance)
    if origin == "camera":
        return placement_pointcloud
    if origin != "world":
        print(f"[WARN] Unsupported pointcloud origin '{origin}'; semantic references disabled.")
        return None

    world_to_camera = load_world_to_camera_transform(loader_instance)
    if world_to_camera is None:
        print("[WARN] camera_extrinsics.json missing or invalid for world point cloud; semantic references disabled.")
        return None
    return transform_points(placement_pointcloud, world_to_camera)


def load_world_to_camera_transform(loader_instance: Any) -> np.ndarray | None:
    extrinsics_path = os.path.join(loader_instance.DUMP_DIR, "camera_extrinsics.json")
    if not os.path.exists(extrinsics_path):
        return None

    info = read_json(extrinsics_path)
    matrix = extrinsics_to_matrix(info)
    if matrix is None:
        return None

    parent_frame = str(info.get("parent_frame", "")).lower() if isinstance(info, Mapping) else ""
    child_frame = str(info.get("child_frame", "")).lower() if isinstance(info, Mapping) else ""
    if "camera" in parent_frame or "optical" in parent_frame:
        return matrix
    if "camera" in child_frame or "optical" in child_frame:
        return np.linalg.inv(matrix)
    return np.linalg.inv(matrix)


def extrinsics_to_matrix(info: Any) -> np.ndarray | None:
    if not isinstance(info, Mapping):
        return None
    if "matrix" in info:
        matrix = np.asarray(info["matrix"], dtype=float)
        if matrix.shape == (4, 4) and np.isfinite(matrix).all():
            return matrix
        return None
    if "translation" in info and "rotation_quaternion" in info:
        translation = info["translation"]
        quaternion = info["rotation_quaternion"]
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = quaternion_to_rotation_matrix(
            float(quaternion["x"]),
            float(quaternion["y"]),
            float(quaternion["z"]),
            float(quaternion["w"]),
        )
        matrix[:3, 3] = [
            float(translation["x"]),
            float(translation["y"]),
            float(translation["z"]),
        ]
        return matrix
    return None


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("camera_extrinsics quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


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


__all__ = [
    "get_semantic_reference_geometry",
    "json_ready",
    "load_camera_intrinsics",
    "load_pointcloud_points",
    "points_for_detection_mask",
    "pointcloud_origin",
    "project_pointcloud_to_image",
    "projection_pointcloud_points",
    "summarize_object_points",
]
