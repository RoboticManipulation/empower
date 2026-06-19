"""Common IO and visualization helpers for Empower semantic placement."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import open3d as o3d
from PIL import Image

from semantic_placement_config import DEFAULT_FRAME_ID


ImageInput = str | os.PathLike[str] | np.ndarray | Image.Image
PointCloudInput = str | os.PathLike[str] | o3d.geometry.PointCloud | np.ndarray
CameraInfoInput = str | os.PathLike[str] | Mapping[str, Any] | None


def existing_path(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser()
    if not value.exists():
        raise FileNotFoundError(f"{name} does not exist: {value}")
    return value


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        return Image.open(existing_path(image, "image")).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def load_pointcloud(pointcloud: PointCloudInput) -> o3d.geometry.PointCloud:
    if isinstance(pointcloud, (str, os.PathLike)):
        pointcloud_obj = o3d.io.read_point_cloud(str(existing_path(pointcloud, "pointcloud")))
    elif isinstance(pointcloud, o3d.geometry.PointCloud):
        pointcloud_obj = copy.deepcopy(pointcloud)
    elif isinstance(pointcloud, np.ndarray):
        points = np.asarray(pointcloud, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Pointcloud arrays must have shape (N, 3) or (N, >=6), "
                f"got {points.shape}"
            )
        pointcloud_obj = o3d.geometry.PointCloud()
        pointcloud_obj.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] >= 6:
            colors = points[:, 3:6].copy()
            if np.nanmax(colors) > 1.0:
                colors = colors / 255.0
            pointcloud_obj.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    else:
        raise ValueError(f"Unsupported pointcloud type: {type(pointcloud)}")

    if pointcloud_obj.is_empty():
        raise ValueError("Pointcloud is empty or unreadable")

    points = np.asarray(pointcloud_obj.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).any():
        raise ValueError("Pointcloud does not contain valid XYZ points")

    return pointcloud_obj


def load_camera_info(camera_info: CameraInfoInput) -> dict[str, Any] | None:
    if camera_info is None:
        return None
    if isinstance(camera_info, (str, os.PathLike)):
        with open(existing_path(camera_info, "camera_info"), encoding="utf-8") as camera_file:
            loaded = json.load(camera_file)
    elif isinstance(camera_info, Mapping):
        loaded = dict(camera_info)
    else:
        raise ValueError(f"Unsupported camera_info type: {type(camera_info)}")

    if not isinstance(loaded, dict):
        raise ValueError("camera_info must load to a JSON object")
    return loaded


def load_camera_intrinsics(camera_info: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, float]:
    info = load_camera_info(camera_info)
    if info is None:
        raise ValueError("camera_info is required to load camera intrinsics")

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
        "camera_info must contain fx/fy/cx/cy, K, or camera_matrix.data"
    )


def resolve_empower_roots(
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


def write_scan_image(image: Image.Image, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination_path, format="JPEG")


def write_pointcloud(pointcloud: o3d.geometry.PointCloud, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(destination_path), pointcloud):
        raise ValueError(f"Unable to write pointcloud: {destination_path}")


def write_camera_info(camera_info: Mapping[str, Any], destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as camera_file:
        json.dump(dict(camera_info), camera_file, indent=2)


def stage_semantic_placement_inputs(
    *,
    mode: str,
    image: Image.Image,
    pointcloud: o3d.geometry.PointCloud,
    camera_info: Mapping[str, Any] | None,
    grasp_object: str,
    images_root: str | os.PathLike[str] | None,
    output_root: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    scan_root, dump_root = resolve_empower_roots(
        images_root=images_root,
        output_root=output_root,
    )
    scan_dir = scan_root / mode
    dump_dir = dump_root / mode
    scan_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    write_scan_image(image, scan_dir / "scan.jpg")
    write_pointcloud(pointcloud, dump_dir / "depth_pointcloud.pcd")

    staged_camera_info = dump_dir / "camera_info.json"
    if camera_info is not None:
        write_camera_info(camera_info, staged_camera_info)
    elif staged_camera_info.exists():
        staged_camera_info.unlink()

    with open(dump_dir / "grasp_object.txt", "w", encoding="utf-8") as grasp_file:
        grasp_file.write(str(grasp_object).strip())

    return scan_dir, dump_dir


def visualization_pointcloud(
    pointcloud: PointCloudInput,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    pointcloud_obj = load_pointcloud(pointcloud)
    if voxel_size > 0:
        pointcloud_obj = pointcloud_obj.voxel_down_sample(voxel_size)

    if pointcloud_obj.is_empty():
        raise ValueError("Pointcloud is empty after downsampling")

    if not pointcloud_obj.has_colors():
        pointcloud_obj.paint_uniform_color((0.55, 0.55, 0.55))

    return pointcloud_obj


def surface_from_result(result: Mapping[str, Any]) -> np.ndarray | None:
    surface = result.get("surface_position")
    if surface is None:
        return None
    if isinstance(surface, dict):
        surface = [surface["x"], surface["y"], surface["z"]]
    return as_point(surface)


def as_point(values: object) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"Expected finite XYZ point, got {values!r}")
    return point


def sphere(
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker.translate(center)
    marker.paint_uniform_color(color)
    marker.compute_vertex_normals()
    return marker


def line(
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.LineSet:
    segment = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([start, end]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    segment.colors = o3d.utility.Vector3dVector([color])
    return segment


def semantic_placement_geometries(
    *,
    pointcloud: o3d.geometry.PointCloud,
    result: Mapping[str, Any],
    marker_radius: float,
) -> list[o3d.geometry.Geometry]:
    coordinate = as_point(result["coordinates"])
    surface = surface_from_result(result)

    geometries: list[o3d.geometry.Geometry] = [pointcloud]
    geometries.append(sphere(coordinate, marker_radius, (1.0, 0.05, 0.05)))

    if surface is not None and not np.allclose(coordinate, surface):
        geometries.append(sphere(surface, marker_radius * 0.65, (0.05, 0.8, 0.1)))
        geometries.append(line(surface, coordinate, (1.0, 0.8, 0.05)))

    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=marker_radius * 5.0,
            origin=coordinate,
        )
    )
    return geometries


def write_marker_files(
    prefix: str | os.PathLike[str],
    pointcloud: o3d.geometry.PointCloud,
    markers: list[o3d.geometry.Geometry],
) -> None:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path = prefix_path.with_name(prefix_path.name + "_scene.ply")
    o3d.io.write_point_cloud(str(scene_path), pointcloud)
    print(f"[OK] wrote {scene_path}")

    for idx, marker in enumerate(markers):
        marker_path = prefix_path.with_name(prefix_path.name + f"_marker_{idx}.ply")
        if isinstance(marker, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(marker_path), marker)
        elif isinstance(marker, o3d.geometry.LineSet):
            o3d.io.write_line_set(str(marker_path), marker)
        print(f"[OK] wrote {marker_path}")


def project_point_to_image(
    point: np.ndarray,
    intrinsics: Mapping[str, float],
    image_shape: tuple[int, int],
) -> tuple[int, int] | None:
    height, width = image_shape
    for z_sign in (1.0, -1.0):
        z = float(point[2]) * z_sign
        if abs(z) < 1e-9:
            continue
        u = int(round(intrinsics["fx"] * float(point[0]) / z + intrinsics["cx"]))
        v = int(round(intrinsics["fy"] * float(point[1]) / z + intrinsics["cy"]))
        if 0 <= u < width and 0 <= v < height:
            return u, v
    return None


def write_image_overlay(
    prefix: str | os.PathLike[str],
    *,
    image: Image.Image,
    camera_info: Mapping[str, Any],
    coordinate: np.ndarray,
    label: str,
) -> None:
    import cv2

    intrinsics = load_camera_intrinsics(camera_info)
    rgb_image = np.asarray(image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    pixel = project_point_to_image(coordinate, intrinsics, bgr_image.shape[:2])
    if pixel is None:
        print(
            "[WARN] placement coordinate projects outside the image; "
            "skipping 2D image overlay"
        )
        return

    x, y = pixel
    radius = max(10, min(bgr_image.shape[:2]) // 45)
    color = (0, 0, 255)
    cv2.circle(bgr_image, (x, y), radius, color, thickness=3)
    cv2.drawMarker(
        bgr_image,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 3,
        thickness=3,
    )
    text = f"place {label} here"
    cv2.putText(
        bgr_image,
        text,
        (min(x + radius + 8, bgr_image.shape[1] - 1), max(y - radius, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    prefix_path = Path(prefix)
    output_path = prefix_path.with_name(prefix_path.name + "_placement_2d.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), bgr_image):
        raise ValueError(f"Unable to write 2D overlay: {output_path}")
    print(f"[OK] wrote {output_path}")


def fmt_point(point: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in point) + "]"


def print_semantic_placement_result(
    result: Mapping[str, Any],
    *,
    grasp_object: str,
    frame_id: str = DEFAULT_FRAME_ID,
) -> None:
    coordinate = as_point(result["coordinates"])
    print(f"[OK] grasp object: {result.get('grasp_object', grasp_object)}")
    print(f"[OK] frame_id    : {result.get('frame_id', frame_id)}")
    if "relation_offset_m" in result:
        print(f"[OK] offset_m    : {float(result['relation_offset_m']):.4f}")
    print(f"[OK] coordinate  : {fmt_point(coordinate)}")


def save_semantic_placement_outputs(
    *,
    pointcloud: PointCloudInput,
    result: Mapping[str, Any],
    prefix: str | os.PathLike[str] | None,
    image: Image.Image | None,
    camera_info: Mapping[str, Any] | None,
    label: str,
    voxel_size: float,
    marker_radius: float,
    show_window: bool,
    window_name: str = "Empower Semantic Placement",
) -> list[o3d.geometry.Geometry]:
    pointcloud_obj = visualization_pointcloud(pointcloud, voxel_size)
    geometries = semantic_placement_geometries(
        pointcloud=pointcloud_obj,
        result=result,
        marker_radius=marker_radius,
    )

    if prefix is not None:
        write_marker_files(prefix, pointcloud_obj, geometries[1:])
        if camera_info is not None and image is not None:
            write_image_overlay(
                prefix,
                image=image,
                camera_info=camera_info,
                coordinate=as_point(result["coordinates"]),
                label=label,
            )
        else:
            print("[WARN] camera_info not provided; skipping 2D image overlay")

    if show_window:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)

    return geometries


__all__ = [
    "CameraInfoInput",
    "DEFAULT_FRAME_ID",
    "ImageInput",
    "PointCloudInput",
    "as_point",
    "existing_path",
    "fmt_point",
    "load_camera_info",
    "load_camera_intrinsics",
    "load_image",
    "load_pointcloud",
    "line",
    "print_semantic_placement_result",
    "project_point_to_image",
    "resolve_empower_roots",
    "save_semantic_placement_outputs",
    "semantic_placement_geometries",
    "sphere",
    "stage_semantic_placement_inputs",
    "surface_from_result",
    "visualization_pointcloud",
    "write_camera_info",
    "write_image_overlay",
    "write_marker_files",
    "write_pointcloud",
    "write_scan_image",
]
