"""Reusable class wrapper for Empower semantic placement."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from semantic_placement_config import DEFAULT_FRAME_ID
from utils.common_utils import CameraInfoInput
from utils.common_utils import ImageInput
from utils.common_utils import PointCloudInput
from utils.common_utils import load_camera_info
from utils.common_utils import load_image
from utils.common_utils import load_pointcloud
from utils.common_utils import print_semantic_placement_result
from utils.common_utils import save_semantic_placement_outputs
from utils.common_utils import stage_semantic_placement_inputs


class EmpowerSemanticPlacementWrapper:
    """Empower semantic placement runner.

    The constructor only prepares Empower's loader, detection pipeline, and
    selected detector model once. Use ``set_inputs(...)`` for scene data,
    ``run()`` to predict placement, and ``save_outputs(...)`` for optional
    debug exports and visualization.
    """

    def __init__(
        self,
        *,
        detector_backend: str,
        mode: str,
        relation_offset_m: float,
        preload_models: bool = True,
        segmentation: Any | None = None,
    ) -> None:
        _ensure_src_on_path()

        if relation_offset_m is None:
            raise ValueError("relation_offset_m is required")
        if mode is None or not mode.strip():
            raise ValueError("mode is required")

        self.mode = mode
        self.detector_backend = detector_backend
        self.relation_offset_m = relation_offset_m

        self.frame_id = DEFAULT_FRAME_ID
        self.images_root: str | os.PathLike[str] | None = None
        self.output_root: str | os.PathLike[str] | None = None
        self.grasp_object: str | None = None
        self.image = None
        self.pointcloud = None
        self.camera_info: dict[str, Any] | None = None
        self.scan_dir: Path | None = None
        self.dump_dir: Path | None = None
        self.semantic_placement_result: dict[str, Any] | None = None

        import loader
        from detection import Detection

        self.loader_instance = loader.Loader("semantic_placement")
        if segmentation is not None:
            self.loader_instance.segmentation = segmentation
        self.detection_instance = Detection()
        if preload_models:
            self.load_models()

    def load_models(self) -> None:
        """Load the selected detector backend once for reuse across runs."""

        if self.detector_backend == "sam3":
            _ = self.loader_instance.segmentation
            return

        if self.detector_backend == "yolo_world":
            from models import YOLOW

            if self.loader_instance.yolow_model is None:
                self.loader_instance.yolow_model = YOLOW(self.loader_instance.YOLOW_PATH)

    def set_inputs(
        self,
        *,
        grasp_object: str,
        image: ImageInput,
        pointcloud: PointCloudInput,
        camera_info: CameraInfoInput = None,
        frame_id: str = DEFAULT_FRAME_ID,
        images_root: str | os.PathLike[str] | None = None,
        output_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """Set scene-specific inputs for the next semantic placement run."""

        if not grasp_object or not str(grasp_object).strip():
            raise ValueError("grasp_object is required for semantic placement")
        if not frame_id or not str(frame_id).strip():
            raise ValueError("frame_id is required for semantic placement")

        self.grasp_object = str(grasp_object).strip()
        self.image = load_image(image)
        self.pointcloud = load_pointcloud(pointcloud)
        self.camera_info = load_camera_info(camera_info)
        self.frame_id = str(frame_id).strip()
        self.images_root = images_root
        self.output_root = output_root
        self.scan_dir = None
        self.dump_dir = None
        self.semantic_placement_result = None

    def run(self) -> dict[str, Any]:
        """Run semantic placement using the previously set scene inputs."""

        self._ensure_inputs_ready()

        scan_dir, dump_dir = stage_semantic_placement_inputs(
            mode=self.mode,
            image=self.image,
            pointcloud=self.pointcloud,
            camera_info=self.camera_info,
            grasp_object=self.grasp_object,
            images_root=self.images_root,
            output_root=self.output_root,
        )
        self.scan_dir = scan_dir
        self.dump_dir = dump_dir

        loader_instance = _build_semantic_loader(
            mode=self.mode,
            scan_dir=scan_dir,
            dump_dir=dump_dir,
            grasp_object=self.grasp_object,
            frame_id=self.frame_id,
            detector_backend=self.detector_backend,
            relation_offset_m=self.relation_offset_m,
            loader_instance=self.loader_instance,
        )

        self.detection_instance.set_loader(loader_instance)
        self.semantic_placement_result = self.detection_instance.semantic_placement_result
        return self.semantic_placement_result

    def save_outputs(
        self,
        *,
        write_prefix: str | os.PathLike[str] | None = None,
        voxel_size: float = 0.01,
        marker_radius: float = 0.03,
        show_window: bool = True,
    ) -> list[Any]:
        """Print results and optionally save visualization/debug artifacts."""

        if self.semantic_placement_result is None:
            raise ValueError("No semantic placement result available. Call run() first.")

        print_semantic_placement_result(
            self.semantic_placement_result,
            grasp_object=self.grasp_object or "",
            frame_id=self.frame_id,
        )
        return save_semantic_placement_outputs(
            pointcloud=self.pointcloud,
            result=self.semantic_placement_result,
            prefix=write_prefix,
            image=self.image,
            camera_info=self.camera_info,
            label=self.semantic_placement_result.get("grasp_object", self.grasp_object or ""),
            voxel_size=voxel_size,
            marker_radius=marker_radius,
            show_window=show_window,
        )

    def _ensure_inputs_ready(self) -> None:
        if self.grasp_object is None:
            raise ValueError("No grasp_object set. Call set_inputs(...) first.")
        if self.image is None:
            raise ValueError("No image set. Call set_inputs(...) first.")
        if self.pointcloud is None:
            raise ValueError("No pointcloud set. Call set_inputs(...) first.")


def _build_semantic_loader(
    *,
    mode: str,
    scan_dir: Path,
    dump_dir: Path,
    grasp_object: str,
    frame_id: str,
    detector_backend: str,
    relation_offset_m: float,
    loader_instance: Any | None = None,
):
    _ensure_src_on_path()
    if loader_instance is None:
        import loader

        loader_instance = loader.Loader("semantic_placement")
    loader_instance.task_name = "semantic_placement"
    loader_instance.SCAN_DIR = str(scan_dir) + os.sep
    loader_instance.DUMP_DIR = str(dump_dir) + os.sep
    loader_instance.grasp_object = grasp_object
    loader_instance.semantic_frame_id = frame_id
    loader_instance.detector_backend = detector_backend
    loader_instance.mode = mode
    loader_instance.semantic_relation_offset_m = relation_offset_m
    return loader_instance


def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


__all__ = [
    "EmpowerSemanticPlacementWrapper",
]
