from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest


EMPOWER_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EMPOWER_SRC))

from semantic_placement_wrapper import run_semantic_placement  # noqa: E402


def test_run_semantic_placement_calls_high_level_wrapper_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    scene_root = tmp_path / "orbbec_gemini_336"
    scene_dir = scene_root / "place" / "5"
    scene_dir.mkdir(parents=True)
    image_path = scene_dir / "placement.png"
    pointcloud_path = scene_dir / "placement.pcd"
    camera_info_path = scene_root / "camera_intrinsics.json"
    images_root = tmp_path / "images"
    output_root = tmp_path / "output"

    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    pointcloud_path.write_text("fake point cloud")
    camera_info_path.write_text('{"fx": 1, "fy": 1, "cx": 0, "cy": 0}')

    seen: dict[str, object] = {}

    class FakeLoader:
        def __init__(self, use_case: str) -> None:
            self.use_case = use_case

    class FakeDetection:
        def set_loader(self, loader_instance: FakeLoader) -> None:
            dump_dir = Path(loader_instance.DUMP_DIR)
            scan_dir = Path(loader_instance.SCAN_DIR)
            seen.update(
                {
                    "use_case": loader_instance.use_case,
                    "grasp_object": loader_instance.grasp_object,
                    "frame_id": loader_instance.semantic_frame_id,
                    "scan_exists": (scan_dir / "scan.jpg").exists(),
                    "pointcloud_text": (
                        dump_dir / "depth_pointcloud.pcd"
                    ).read_text(),
                    "camera_exists": (dump_dir / "camera_info.json").exists(),
                    "camera_text": (dump_dir / "camera_info.json").read_text(),
                    "grasp_file_text": (
                        dump_dir / "grasp_object.txt"
                    ).read_text(),
                }
            )
            self.semantic_placement_result = {
                "coordinates": [1.0, 2.0, 3.0],
                "grasp_object": loader_instance.grasp_object,
            }

    monkeypatch.setitem(sys.modules, "loader", types.SimpleNamespace(Loader=FakeLoader))
    monkeypatch.setitem(
        sys.modules,
        "detection",
        types.SimpleNamespace(Detection=FakeDetection),
    )

    result = run_semantic_placement(
        grasp_object="milk carton",
        image_path=image_path,
        pointcloud_path=pointcloud_path,
        camera_info_path=camera_info_path,
        images_root=images_root,
        output_root=output_root,
    )

    assert result == {"coordinates": [1.0, 2.0, 3.0], "grasp_object": "milk carton"}
    assert seen == {
        "use_case": "semantic_placement",
        "grasp_object": "milk carton",
        "frame_id": "gemini336_color_optical_frame",
        "scan_exists": True,
        "pointcloud_text": "fake point cloud",
        "camera_exists": True,
        "camera_text": '{"fx": 1, "fy": 1, "cx": 0, "cy": 0}',
        "grasp_file_text": "milk carton",
    }
