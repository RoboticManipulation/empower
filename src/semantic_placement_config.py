"""Configuration helpers for Empower semantic placement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
SEMANTIC_PLACEMENT_CONFIG_PATH = ROOT_DIR / "configs" / "semantic_placement.yaml"
_SEMANTIC_PLACEMENT_CONFIG: dict[str, Any] | None = None


def load_semantic_placement_config() -> dict[str, Any]:
    global _SEMANTIC_PLACEMENT_CONFIG
    if _SEMANTIC_PLACEMENT_CONFIG is None:
        with open(SEMANTIC_PLACEMENT_CONFIG_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                "Semantic placement config must be a mapping: "
                f"{SEMANTIC_PLACEMENT_CONFIG_PATH}"
            )
        _SEMANTIC_PLACEMENT_CONFIG = loaded
    return _SEMANTIC_PLACEMENT_CONFIG


def required_config_value(key: str) -> Any:
    config = load_semantic_placement_config()
    if key not in config:
        raise KeyError(f"Missing semantic placement config key: {key}")
    return config[key]


ORIGINAL_MODE = str(required_config_value("original_mode"))
REFINED_MODE = str(required_config_value("refined_mode"))
DEFAULT_MODE = str(required_config_value("default_mode"))
DEFAULT_FRAME_ID = str(required_config_value("default_frame_id"))
DEFAULT_RELATION_OFFSET_M = float(required_config_value("default_relation_offset_m"))
DEFAULT_EMPOWER_RELATION_OFFSET_M = float(
    required_config_value("default_empower_relation_offset_m")
)
SUPPORTED_DETECTOR_BACKENDS = tuple(
    str(backend) for backend in required_config_value("supported_detector_backends")
)
SUPPORTED_SEMANTIC_PLACEMENT_MODES = (ORIGINAL_MODE, REFINED_MODE)
__all__ = [
    "DEFAULT_EMPOWER_RELATION_OFFSET_M",
    "DEFAULT_FRAME_ID",
    "DEFAULT_MODE",
    "DEFAULT_RELATION_OFFSET_M",
    "ORIGINAL_MODE",
    "REFINED_MODE",
    "ROOT_DIR",
    "SEMANTIC_PLACEMENT_CONFIG_PATH",
    "SUPPORTED_DETECTOR_BACKENDS",
    "SUPPORTED_SEMANTIC_PLACEMENT_MODES",
    "load_semantic_placement_config",
    "required_config_value",
]
