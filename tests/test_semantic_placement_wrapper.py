from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from empower.semantic_placement_wrapper import EmpowerSemanticPlacementWrapper


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
        def __init__(self, task_name: str) -> None:
            self.task_name = task_name
            self._segmentation = object()

        @property
        def segmentation(self):
            return self._segmentation

    class FakeDetection:
        def set_loader(self, loader_instance: FakeLoader) -> None:
            dump_dir = Path(loader_instance.DUMP_DIR)
            scan_dir = Path(loader_instance.SCAN_DIR)
            seen.update(
                {
                    "task_name": loader_instance.task_name,
                    "grasp_object": loader_instance.grasp_object,
                    "frame_id": loader_instance.semantic_frame_id,
                    "mode": loader_instance.mode,
                    "scan_exists": (scan_dir / "scan.jpg").exists(),
                    "pointcloud_exists": (dump_dir / "depth_pointcloud.pcd").exists(),
                    "camera_exists": (dump_dir / "camera_info.json").exists(),
                    "camera_info": json.loads((dump_dir / "camera_info.json").read_text()),
                    "shelf_board_heights": loader_instance.semantic_shelf_board_heights,
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

    wrapper = EmpowerSemanticPlacementWrapper(mode="original", output_root=output_root)
    wrapper.set_inputs(
        grasp_object="milk carton",
        image=image_path,
        pointcloud=np.array([[0.0, 0.0, 0.0]]),
        camera_info=camera_info_path,
        frame_id="gemini336_color_optical_frame",
        images_root=images_root,
        shelf_board_heights=[0.72, 1.07],
    )
    result = wrapper.run()

    assert result == {"coordinates": [1.0, 2.0, 3.0], "grasp_object": "milk carton"}
    assert seen == {
        "task_name": "semantic_placement",
        "grasp_object": "milk carton",
        "frame_id": "gemini336_color_optical_frame",
        "mode": "original",
        "scan_exists": True,
        "pointcloud_exists": True,
        "camera_exists": True,
        "camera_info": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
        "shelf_board_heights": (0.72, 1.07),
        "grasp_file_text": "milk carton",
    }


def test_refined_mode_forwards_shelf_board_heights(
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
        def __init__(self, task_name: str) -> None:
            self.task_name = task_name
            self._segmentation = object()

        @property
        def segmentation(self):
            return self._segmentation

    class FakeDetection:
        def set_loader(self, loader_instance: FakeLoader) -> None:
            seen["shelf_board_heights"] = loader_instance.semantic_shelf_board_heights
            self.semantic_placement_result = {"coordinates": [1.0, 2.0, 3.0]}

    monkeypatch.setitem(sys.modules, "loader", types.SimpleNamespace(Loader=FakeLoader))
    monkeypatch.setitem(
        sys.modules,
        "detection",
        types.SimpleNamespace(Detection=FakeDetection),
    )

    wrapper = EmpowerSemanticPlacementWrapper(mode="refined", output_root=output_root)
    wrapper.set_inputs(
        grasp_object="milk carton",
        image=image_path,
        pointcloud=np.array([[0.0, 0.0, 0.0]]),
        camera_info=camera_info_path,
        frame_id="gemini336_color_optical_frame",
        images_root=images_root,
        shelf_board_heights=[0.72, 1.07],
    )
    wrapper.run()

    assert seen["shelf_board_heights"] == (0.72, 1.07)
