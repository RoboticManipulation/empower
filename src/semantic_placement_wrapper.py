"""Reusable class wrapper for Empower semantic placement."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any


class EmpowerSemanticPlacementWrapper:
    """Empower semantic placement runner.

    The constructor prepares Empower's loader, detection pipeline, and selected
    detector model once. Call ``run(...)`` for each RGB/point-cloud placement
    scene.
    """

    def __init__(
        self,
        *,
        detector_backend: str,
        frame_id: str,
        semantic_mode: str,
        relation_offset_m: float,
        use_case: str,
        images_root: str | os.PathLike[str] | None = None,
        output_root: str | os.PathLike[str] | None = None,
        preload_models: bool = True,
        segmentation: Any | None = None,
    ) -> None:
        _ensure_src_on_path()

        if relation_offset_m is None:
            raise ValueError("relation_offset_m is required")
        if use_case is None or not use_case.strip():
            raise ValueError("use_case is required")

        self.semantic_mode = semantic_mode
        self.use_case = use_case
        self.frame_id = frame_id
        self.detector_backend = detector_backend
        self.relation_offset_m = relation_offset_m
        self.images_root = images_root
        self.output_root = output_root

        import loader
        from detection import Detection

        self.loader_instance = loader.Loader(self.use_case)
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

    def run(
        self,
        *,
        grasp_object: str,
        image_path: str | os.PathLike[str],
        pointcloud_path: str | os.PathLike[str],
        camera_info_path: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Run semantic placement for one scene and return the coordinate result."""

        if not grasp_object or not str(grasp_object).strip():
            raise ValueError("grasp_object is required for semantic placement")

        image_path = _existing_path(image_path, "image_path")
        pointcloud_path = _existing_path(pointcloud_path, "pointcloud_path")
        camera_info = _resolve_camera_info_path(camera_info_path)

        scan_dir, dump_dir = _stage_semantic_placement_inputs(
            use_case=self.use_case,
            image_path=image_path,
            pointcloud_path=pointcloud_path,
            camera_info_path=camera_info,
            grasp_object=grasp_object,
            images_root=self.images_root,
            output_root=self.output_root,
        )

        loader_instance = _build_semantic_loader(
            use_case=self.use_case,
            scan_dir=scan_dir,
            dump_dir=dump_dir,
            grasp_object=str(grasp_object).strip(),
            frame_id=self.frame_id,
            detector_backend=self.detector_backend,
            semantic_mode=self.semantic_mode,
            relation_offset_m=self.relation_offset_m,
            loader_instance=self.loader_instance,
        )

        self.detection_instance.set_loader(loader_instance)
        return self.detection_instance.semantic_placement_result


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
    detector_backend: str,
    semantic_mode: str,
    relation_offset_m: float,
    loader_instance: Any | None = None,
):
    _ensure_src_on_path()
    if loader_instance is None:
        import loader

        loader_instance = loader.Loader(use_case)
    loader_instance.use_case = use_case
    loader_instance.SCAN_DIR = str(scan_dir) + os.sep
    loader_instance.DUMP_DIR = str(dump_dir) + os.sep
    loader_instance.grasp_object = grasp_object
    loader_instance.semantic_frame_id = frame_id
    loader_instance.detector_backend = detector_backend
    loader_instance.semantic_mode = semantic_mode
    loader_instance.semantic_relation_offset_m = relation_offset_m
    return loader_instance


def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _resolve_camera_info_path(
    camera_info_path: str | os.PathLike[str] | None,
) -> Path | None:
    if camera_info_path is not None:
        return _existing_path(camera_info_path, "camera_info_path")
    return None


def _existing_path(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path).expanduser()
    if not value.exists():
        raise FileNotFoundError(f"{name} does not exist: {value}")
    return value


__all__ = [
    "EmpowerSemanticPlacementWrapper",
]
