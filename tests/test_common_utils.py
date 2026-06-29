from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from utils.common_utils import combine_debug_pointcloud
from utils.common_utils import write_marker_files
from utils.common_utils import write_scene_debug_files


def test_combine_debug_pointcloud_includes_scene_and_markers() -> None:
    scene = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=float)
        )
    )
    scene.paint_uniform_color((0.5, 0.5, 0.5))

    reference = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    reference.translate(np.array([0.2, 0.2, 0.2]))
    reference.paint_uniform_color((0.55, 0.55, 0.55))

    placement = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    placement.translate(np.array([0.35, 0.2, 0.2]))
    placement.paint_uniform_color((1.0, 0.0, 0.0))

    combined = combine_debug_pointcloud(scene, [reference, placement])

    assert len(combined.points) > len(scene.points)
    assert combined.has_colors()


def test_write_marker_files_writes_single_placement_3d_ply(tmp_path: Path) -> None:
    scene = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=float))
    )
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    marker.paint_uniform_color((1.0, 0.0, 0.0))

    prefix = tmp_path / "salt_box"
    write_marker_files(prefix, scene, [marker])

    placement_path = tmp_path / "salt_box_placement_3d.ply"
    assert placement_path.exists()
    assert not (tmp_path / "salt_box_scene.ply").exists()
    assert not (tmp_path / "salt_box_marker_0.ply").exists()

    loaded = o3d.io.read_point_cloud(str(placement_path))
    assert not loaded.is_empty()
    assert len(loaded.points) > len(scene.points)


def test_write_scene_debug_files_writes_unmarked_outputs(tmp_path: Path) -> None:
    scene = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float))
    )
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    prefix = tmp_path / "ketchup_bottle"
    write_scene_debug_files(prefix, scene, image=image)

    loaded = o3d.io.read_point_cloud(str(tmp_path / "ketchup_bottle_placement_3d.ply"))
    assert len(loaded.points) == len(scene.points)
    assert (tmp_path / "ketchup_bottle_placement_2d.png").exists()
