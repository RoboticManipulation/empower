"""Run semantic placement grounding from LLM plans, detections, and point clouds."""

from __future__ import annotations

import os
from typing import Any, Mapping

from utils.common_utils import save_json
from utils.config_utils import DEFAULT_MODE
from utils.config_utils import REFINED_MODE
from semantic_placement_geometry import get_empower_style_semantic_coordinates
from semantic_placement_geometry import get_semantic_placement_coordinates_from_plan
from semantic_placement_reference_geometry import get_semantic_reference_geometry
from semantic_placement_reference_geometry import json_ready
from semantic_placement_reference_geometry import load_pointcloud_points


def run_grounded_semantic_placement(
    *,
    loader_instance: Any,
    results_multi: Mapping[str, str],
    detections: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return semantic placement coordinates using Empower LLM + detector output."""

    pcd_path = os.path.join(loader_instance.DUMP_DIR, "depth_pointcloud.pcd")
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(
            "Semantic placement requires a point cloud at "
            f"{pcd_path}. Run prepare_local_data.py or provide "
            "output/semantic_placement/depth_pointcloud.pcd first."
        )

    placement_pointcloud = load_pointcloud_points(pcd_path)
    reference_positions = get_semantic_reference_geometry(
        loader_instance=loader_instance,
        detections=detections,
        placement_pointcloud=placement_pointcloud,
    )
    grasp_object = get_semantic_grasp_object(loader_instance, required=True)
    planning_text = results_multi.get("planning_agent_info", "")
    mode = semantic_placement_mode(loader_instance)
    relation_offset_m = semantic_relation_offset_m(loader_instance)

    if mode == REFINED_MODE:
        result = get_semantic_placement_coordinates_from_plan(
            planning_text,
            placement_pointclouds=placement_pointcloud,
            grasp_object=grasp_object,
            reference_positions_by_name=reference_positions,
            relation_offset_m=relation_offset_m,
            frame_id=semantic_frame_id(loader_instance),
        )
    else:
        result = get_empower_style_semantic_coordinates(
            planning_text=planning_text,
            placement_pointcloud=placement_pointcloud,
            grasp_object=grasp_object,
            reference_positions=reference_positions,
            relation_offset_m=relation_offset_m,
            frame_id=semantic_frame_id(loader_instance),
        )

    result["mode"] = mode
    result["relation_offset_m"] = relation_offset_m

    result_path = os.path.join(
        loader_instance.DUMP_DIR,
        "semantic_placement_result.json",
    )
    save_json(result_path, json_ready(result))

    print(f"[OK] Semantic placement result -> {result_path}")
    print(f"[OK] Semantic placement coordinates: {result['coordinates']}")
    return result


def semantic_placement_mode(loader_instance: Any) -> str:
    return getattr(loader_instance, "mode", None) or DEFAULT_MODE


def semantic_relation_offset_m(loader_instance: Any) -> float:
    return loader_instance.semantic_relation_offset_m


def get_semantic_grasp_object(loader_instance: Any, required: bool = False) -> str | None:
    candidates = [
        getattr(loader_instance, "grasp_object", None),
        os.environ.get("EMPOWER_GRASP_OBJECT"),
        os.environ.get("GRASP_OBJECT"),
    ]

    for path in (
        os.path.join(loader_instance.DUMP_DIR, "grasp_object.txt"),
        os.path.join(loader_instance.SCAN_DIR, "grasp_object.txt"),
    ):
        if os.path.exists(path):
            with open(path) as f:
                candidates.append(f.read())

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()

    if required:
        raise ValueError(
            "semantic_placement requires the already-grasped object name. "
            "Set EMPOWER_GRASP_OBJECT or pass grasp_object to "
            "EmpowerSemanticPlacementWrapper.run(...)."
        )
    return None


def semantic_frame_id(loader_instance: Any) -> str:
    return (
        getattr(loader_instance, "semantic_frame_id", None)
        or os.environ.get(
            "EMPOWER_SEMANTIC_FRAME_ID",
            "gemini336_color_optical_frame",
        )
    )


__all__ = [
    "get_semantic_grasp_object",
    "run_grounded_semantic_placement",
    "semantic_frame_id",
    "semantic_placement_mode",
    "semantic_relation_offset_m",
]
