"""Reusable class wrapper for Empower semantic placement."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import Any

from std_msgs.msg import Bool

from utils.common_utils import CameraExtrinsicsInput
from utils.common_utils import CameraInfoInput
from utils.common_utils import ImageInput
from utils.common_utils import PointCloudInput
from utils.common_utils import get_config
from utils.common_utils import load_camera_extrinsics
from utils.common_utils import load_camera_info
from utils.common_utils import load_image
from utils.common_utils import load_pointcloud
from utils.common_utils import print_semantic_placement_result
from utils.common_utils import save_semantic_placement_outputs
from utils.common_utils import stage_semantic_placement_inputs


class EmpowerSemanticPlacementWrapper:
    """Empower semantic placement runner.

    The constructor loads Empower's semantic placement configuration, prepares
    Empower's loader and detection pipeline, and preloads the
    selected detector model once. Use ``set_inputs(...)`` for scene data,
    ``run()`` to predict placement, and ``save_outputs(...)`` for optional
    debug exports and visualization.
    """

    def __init__(
        self,
        *,
        mode: str | None = None,
        segmentation: Any | None = None,
        simulation: bool | None = None,
        output_root: str | os.PathLike[str] | None = None,
        ai: str | None = None,
        camera_info: CameraInfoInput = None,
        camera_extrinsics: CameraExtrinsicsInput = None,
    ) -> None:
        _ensure_src_on_path()

        self.config = _load_empower_config()
        self.mode_configs = _mode_config_mapping(self.config)
        self.default_mode = _select_mode(
            _required_str(self.config, "default_mode"),
            self.mode_configs,
        )
        self.mode = _select_mode(mode or self.default_mode, self.mode_configs)
        self.mode_config = self.mode_configs[self.mode]
        self.original_mode = _mode_at_index(self.mode_configs, 0)
        self.refined_mode = _mode_at_index(self.mode_configs, 1)
        self.semantic_placement_modes = tuple(self.mode_configs)

        self.default_frame_id = _required_str(self.config, "frame_id")
        self.frame_id = self.default_frame_id
        self.detector_backend = _canonical_detector_backend(
            _required_str(self.mode_config, "detector_backend")
        )
        self.relation_offset_m = _required_float(
            self.mode_config,
            "relation_offset_m",
        )

        visualization_config = _required_mapping(self.config, "visualization")
        self.default_voxel_size = _required_float(visualization_config, "voxel_size")
        self.default_marker_radius = _required_float(
            visualization_config,
            "marker_radius",
        )
        self.default_show_window = _required_bool(visualization_config, "show_window")

        self.images_root: str | os.PathLike[str] | None = None
        self.output_root: str | os.PathLike[str] | None = output_root
        self.ai = _canonical_llm_provider(ai) if ai is not None else None
        self.grasp_object: str | None = None
        self.image = None
        self.pointcloud = None
        self.pointcloud_origin = "camera"
        self.camera_info = _checked_camera_info(camera_info)
        self.camera_info_format = _camera_info_format(self.camera_info)
        self.simulation = bool(simulation)
        self.camera_extrinsics = load_camera_extrinsics(
            camera_extrinsics,
            simulation=self.simulation,
        )
        self.scan_dir: Path | None = None
        self.dump_dir: Path | None = None
        self.semantic_placement_result: dict[str, Any] | None = None

        import loader
        from detection import Detection

        self.loader_instance = loader.Loader("semantic_placement")
        if segmentation is not None:
            self.loader_instance.segmentation = segmentation
        self.detection_instance = Detection()
        self.load_models()

    def load_models(self) -> None:
        """Load the selected detector backend once for reuse across runs."""

        if self.detector_backend == "sam3":
            _ = self.loader_instance.segmentation
            return

        if self.detector_backend == "yolow":
            from models import YOLOW

            if self.loader_instance.yolow_model is None:
                self.loader_instance.yolow_model = YOLOW(self.loader_instance.YOLOW_PATH)
            return

        raise ValueError(f"Unsupported detector backend: {self.detector_backend}")

    def set_inputs(
        self,
        *,
        grasp_object: str,
        image: ImageInput,
        pointcloud: PointCloudInput,
        camera_info: CameraInfoInput = None,
        camera_extrinsics: CameraExtrinsicsInput = None,
        pointcloud_origin: str = "camera",
        frame_id: str | None = None,
        images_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """Set scene-specific inputs for the next semantic placement run."""

        resolved_frame_id = self.default_frame_id if frame_id is None else str(frame_id).strip()
        if not grasp_object or not str(grasp_object).strip():
            raise ValueError("grasp_object is required for semantic placement")
        if not resolved_frame_id:
            raise ValueError("frame_id is required for semantic placement")
        resolved_pointcloud_origin = _canonical_pointcloud_origin(pointcloud_origin)

        self.grasp_object = str(grasp_object).strip()
        self.image = load_image(image)
        self.pointcloud = load_pointcloud(pointcloud)
        if camera_info is not None:
            self.camera_info = _checked_camera_info(camera_info)
            self.camera_info_format = _camera_info_format(self.camera_info)
        if camera_extrinsics is not None:
            self.camera_extrinsics = load_camera_extrinsics(
                camera_extrinsics,
                simulation=self.simulation,
            )
        self.pointcloud_origin = resolved_pointcloud_origin
        self.frame_id = resolved_frame_id
        self.images_root = images_root
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
            camera_extrinsics=self.camera_extrinsics,
            pointcloud_origin=self.pointcloud_origin,
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
            pointcloud_origin=self.pointcloud_origin,
            semantic_placement_modes=self.semantic_placement_modes,
            semantic_default_mode=self.default_mode,
            semantic_refined_mode=self.refined_mode,
            semantic_mode_config=self.mode_config,
            semantic_placement_config=self.config,
            llm_provider=self.ai,
            loader_instance=self.loader_instance,
        )

        self.detection_instance.set_loader(loader_instance)
        self.semantic_placement_result = self.detection_instance.semantic_placement_result
        return self.semantic_placement_result

    def save_outputs(
        self,
        *,
        write_prefix: str | os.PathLike[str] | None = None,
        voxel_size: float | None = None,
        marker_radius: float | None = None,
        show_window: bool | None = None,
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
            prefix=self._resolve_write_prefix(write_prefix),
            image=self.image,
            camera_info=self.camera_info,
            label=self.semantic_placement_result.get("grasp_object", self.grasp_object or ""),
            voxel_size=self.default_voxel_size if voxel_size is None else float(voxel_size),
            marker_radius=(
                self.default_marker_radius if marker_radius is None else float(marker_radius)
            ),
            show_window=self.default_show_window if show_window is None else bool(show_window),
            camera_extrinsics=self.camera_extrinsics,
            pointcloud_origin=self.pointcloud_origin,
        )

    def _resolve_write_prefix(
        self,
        write_prefix: str | os.PathLike[str] | None,
    ) -> Path | None:
        if write_prefix is None:
            return None

        prefix = Path(write_prefix)
        if prefix.is_absolute():
            return prefix
        if self.dump_dir is not None:
            return self.dump_dir / prefix
        return prefix

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
    pointcloud_origin: str,
    semantic_placement_modes: tuple[str, ...],
    semantic_default_mode: str,
    semantic_refined_mode: str,
    semantic_mode_config: Mapping[str, Any],
    semantic_placement_config: Mapping[str, Any],
    llm_provider: str | None = None,
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
    loader_instance.semantic_pointcloud_origin = pointcloud_origin
    loader_instance.semantic_placement_modes = semantic_placement_modes
    loader_instance.semantic_default_mode = semantic_default_mode
    loader_instance.semantic_refined_mode = semantic_refined_mode
    loader_instance.semantic_mode_config = dict(semantic_mode_config)
    loader_instance.semantic_placement_config = dict(semantic_placement_config)
    loader_instance.llm_provider = llm_provider
    return loader_instance


def _load_empower_config() -> dict[str, Any]:
    config = get_config("empower") or {}
    if not isinstance(config, dict):
        raise ValueError("Empower config must be a mapping: configs/empower.yaml")
    return config


def _mode_config_mapping(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    mode_config = _required_mapping(config, "mode")
    if not mode_config:
        raise ValueError("Empower config must define at least one semantic placement mode")

    resolved_modes: dict[str, Mapping[str, Any]] = {}
    for mode_name, mode_values in mode_config.items():
        if not isinstance(mode_values, Mapping):
            raise ValueError(f"Empower mode config must be a mapping: mode.{mode_name}")
        resolved_modes[str(mode_name)] = mode_values
    return resolved_modes


def _select_mode(mode: str | None, mode_configs: Mapping[str, Mapping[str, Any]]) -> str:
    mode_names = tuple(mode_configs)
    if not mode_names:
        raise ValueError("Empower config must define semantic placement modes")

    requested_mode = str(mode or "").strip()
    if not requested_mode:
        raise ValueError("mode is required")

    mode_aliases = {"original": mode_names[0]}
    if len(mode_names) > 1:
        mode_aliases["refined"] = mode_names[1]

    selected_mode = mode_aliases.get(requested_mode, requested_mode)
    if selected_mode not in mode_configs:
        supported_modes = ", ".join(mode_names)
        raise ValueError(f"Unsupported semantic placement mode '{requested_mode}'. Use one of: {supported_modes}")
    return selected_mode


def _mode_at_index(mode_configs: Mapping[str, Mapping[str, Any]], index: int) -> str:
    mode_names = tuple(mode_configs)
    return mode_names[min(index, len(mode_names) - 1)]


def _canonical_detector_backend(detector_backend: str) -> str:
    backend = detector_backend.strip().lower()
    if backend in {"sam", "sam3"}:
        return "sam3"
    if backend in {"yolo", "yolo-world", "yolo_world", "yoloworld", "yolow"}:
        return "yolow"
    raise ValueError("detector_backend must be one of: sam3, yolow")


def _checked_camera_info(camera_info: CameraInfoInput) -> dict[str, Any] | None:
    loaded = load_camera_info(camera_info)
    _camera_info_format(loaded)
    return loaded


def _camera_info_format(camera_info: Mapping[str, Any] | None) -> str | None:
    if camera_info is None:
        return None
    if "K" in camera_info:
        k_values = camera_info["K"]
        try:
            import numpy as np

            matrix = np.asarray(k_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("camera_info K must be numeric") from exc
        if matrix.shape not in {(3, 3), (9,)}:
            raise ValueError(f"camera_info K must be a 3x3 matrix or 9 values, got shape {matrix.shape}")
        if set(camera_info).issubset({"K", "camera_info_format"}):
            return "K"
        return "complete"
    if "camera_matrix" in camera_info or {"fx", "fy", "cx", "cy"}.issubset(camera_info):
        return "complete"
    raise ValueError("camera_info must be a full camera_info object or a 3x3 K matrix")


def _canonical_llm_provider(ai: str) -> str:
    provider = str(ai).strip().lower()
    if provider in {"chatgpt", "openai", "gpt"}:
        return "chatgpt"
    if provider == "mistral":
        return "mistral"
    if provider in {"openrouter", "open_router"}:
        return "openrouter"
    raise ValueError("ai must be one of: chatgpt, mistral, openrouter")


def _canonical_pointcloud_origin(pointcloud_origin: str) -> str:
    origin = str(pointcloud_origin).strip().lower()
    if origin in {"camera", "world"}:
        return origin
    raise ValueError("pointcloud_origin must be one of: camera, world")


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required_value(config, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Empower config key must be a mapping: {key}")
    return value


def _required_str(config: Mapping[str, Any], key: str) -> str:
    value = str(_required_value(config, key)).strip()
    if not value:
        raise ValueError(f"Empower config key must not be empty: {key}")
    return value


def _required_float(config: Mapping[str, Any], key: str) -> float:
    return float(_required_value(config, key))


def _required_bool(config: Mapping[str, Any], key: str) -> bool:
    value = _required_value(config, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Empower config key must be a boolean: {key}")


def _required_value(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing Empower config key: {key}")
    return config[key]


def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


__all__ = [
    "EmpowerSemanticPlacementWrapper",
]
