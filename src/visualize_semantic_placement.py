#!/usr/bin/env python3
"""Run semantic placement and visualize the returned coordinate in Open3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from semantic_placement_wrapper import run_semantic_placement
from semantic_placement_wrapper import DEFAULT_FRAME_ID


def main() -> None:
    args = _parse_args()

    result = run_semantic_placement(
        grasp_object=args.grasp_object,
        image_path=args.image,
        pointcloud_path=args.pointcloud,
        camera_info_path=args.camera_info,
        frame_id=args.frame_id,
        images_root=args.images_root,
        output_root=args.output_root,
    )

    pointcloud = _load_pointcloud(args.pointcloud, args.voxel_size)
    coordinate = _as_point(result["coordinates"])
    surface = _surface_from_result(result)

    geometries: list[o3d.geometry.Geometry] = [pointcloud]
    geometries.append(_sphere(coordinate, args.marker_radius, (1.0, 0.05, 0.05)))

    if surface is not None and not np.allclose(coordinate, surface):
        geometries.append(_sphere(surface, args.marker_radius * 0.65, (0.05, 0.8, 0.1)))
        geometries.append(_line(surface, coordinate, (1.0, 0.8, 0.05)))

    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=args.marker_radius * 5.0,
            origin=coordinate,
        )
    )

    print(f"[OK] grasp object: {result.get('grasp_object', args.grasp_object)}")
    print(f"[OK] frame_id    : {result.get('frame_id', args.frame_id)}")
    print(f"[OK] coordinate  : {_fmt_point(coordinate)}")

    if args.write_prefix:
        _write_marker_files(args.write_prefix, pointcloud, geometries[1:])
        if args.camera_info is not None:
            _write_image_overlay(
                args.write_prefix,
                image_path=args.image,
                camera_info_path=args.camera_info,
                coordinate=coordinate,
                label=result.get("grasp_object", args.grasp_object),
            )
        else:
            print("[WARN] --camera-info not provided; skipping 2D image overlay")

    if not args.no_window:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Empower Semantic Placement",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call run_semantic_placement(...) and show the returned placement "
            "coordinate in the input point cloud. Red sphere = returned coordinate."
        )
    )
    parser.add_argument("image", type=Path, help="Placement scene RGB image")
    parser.add_argument("pointcloud", type=Path, help="Placement scene .pcd/.ply file")
    parser.add_argument(
        "--grasp-object",
        required=True,
        help="Already-held object name, for example 'milk carton'",
    )
    parser.add_argument(
        "--camera-info",
        type=Path,
        help="Optional camera_info.json for image/point-cloud grounding",
    )
    parser.add_argument(
        "--frame-id",
        default=DEFAULT_FRAME_ID,
        help="Frame label for the returned coordinate",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Optional Empower images root for staging",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional Empower output root for staging",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Downsample voxel size in meters; use 0 to disable",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.03,
        help="Marker sphere radius in meters",
    )
    parser.add_argument(
        "--write-prefix",
        type=Path,
        help="Optional output prefix for marker .ply files and 2D overlay PNG",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Only run wrapper and print/write files; do not open Open3D",
    )
    return parser.parse_args()


def _load_pointcloud(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    pointcloud = o3d.io.read_point_cloud(str(path))
    if pointcloud.is_empty():
        raise SystemExit(f"[ERROR] Empty or unreadable point cloud: {path}")

    if voxel_size > 0:
        pointcloud = pointcloud.voxel_down_sample(voxel_size)

    if not pointcloud.has_colors():
        pointcloud.paint_uniform_color((0.55, 0.55, 0.55))

    return pointcloud


def _surface_from_result(result: dict) -> np.ndarray | None:
    surface = result.get("surface_position")
    if surface is None:
        return None
    if isinstance(surface, dict):
        surface = [surface["x"], surface["y"], surface["z"]]
    return _as_point(surface)


def _as_point(values: object) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise SystemExit(f"[ERROR] Expected finite XYZ point, got {values!r}")
    return point


def _sphere(
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    marker.translate(center)
    marker.paint_uniform_color(color)
    marker.compute_vertex_normals()
    return marker


def _line(
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([start, end]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line.colors = o3d.utility.Vector3dVector([color])
    return line


def _write_marker_files(
    prefix: Path,
    pointcloud: o3d.geometry.PointCloud,
    markers: list[o3d.geometry.Geometry],
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scene_path = prefix.with_name(prefix.name + "_scene.ply")
    o3d.io.write_point_cloud(str(scene_path), pointcloud)
    print(f"[OK] wrote {scene_path}")

    for idx, marker in enumerate(markers):
        marker_path = prefix.with_name(prefix.name + f"_marker_{idx}.ply")
        if isinstance(marker, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(marker_path), marker)
        elif isinstance(marker, o3d.geometry.LineSet):
            o3d.io.write_line_set(str(marker_path), marker)
        print(f"[OK] wrote {marker_path}")


def _write_image_overlay(
    prefix: Path,
    *,
    image_path: Path,
    camera_info_path: Path,
    coordinate: np.ndarray,
    label: str,
) -> None:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"[ERROR] Unable to read image for overlay: {image_path}")

    intrinsics = _load_intrinsics(camera_info_path)
    pixel = _project_point_to_image(coordinate, intrinsics, image.shape[:2])
    if pixel is None:
        print(
            "[WARN] placement coordinate projects outside the image; "
            "skipping 2D image overlay"
        )
        return

    x, y = pixel
    radius = max(10, min(image.shape[:2]) // 45)
    color = (0, 0, 255)
    cv2.circle(image, (x, y), radius, color, thickness=3)
    cv2.drawMarker(
        image,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=radius * 3,
        thickness=3,
    )
    text = f"place {label} here"
    cv2.putText(
        image,
        text,
        (min(x + radius + 8, image.shape[1] - 1), max(y - radius, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    output_path = prefix.with_name(prefix.name + "_placement_2d.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise SystemExit(f"[ERROR] Unable to write 2D overlay: {output_path}")
    print(f"[OK] wrote {output_path}")


def _load_intrinsics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as intrinsics_file:
        info = json.load(intrinsics_file)

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

    raise SystemExit(
        f"[ERROR] {path} must contain fx/fy/cx/cy, K, or camera_matrix.data"
    )


def _project_point_to_image(
    point: np.ndarray,
    intrinsics: dict[str, float],
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


def _fmt_point(point: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in point) + "]"


if __name__ == "__main__":
    main()
