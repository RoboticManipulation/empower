from __future__ import annotations

import numpy as np
import pytest

from empower.semantic_placement_geometry import get_semantic_placement_coordinates_from_plan


def test_left_right_offsets_follow_camera_x_axis_without_support_clipping() -> None:
    relation_offset_m = 0.15
    frame_id = "gemini336_color_optical_frame"
    xs = np.linspace(0.37, 0.42, 8)
    ys = np.linspace(-0.49, -0.29, 8)
    support_points = np.array(
        [[x, y, 1.0] for x in xs for y in ys],
        dtype=float,
    )
    reference_positions = {
        "viva paper towel roll": np.array([0.05, -0.25, 1.15], dtype=float),
    }

    left = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton left to viva paper towel roll",
        placement_pointclouds=support_points,
        placement_surface_height_m=1.0,
        reference_positions_by_name=reference_positions,
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
    )
    right = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton right to viva paper towel roll",
        placement_pointclouds=support_points,
        placement_surface_height_m=1.0,
        reference_positions_by_name=reference_positions,
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
    )

    assert left["coordinates"] == pytest.approx(
        [0.05 - relation_offset_m, -0.25, 1.0]
    )
    assert right["coordinates"] == pytest.approx(
        [0.05 + relation_offset_m, -0.25, 1.0]
    )


def test_reference_relation_snaps_z_to_nearest_shelf_board_height() -> None:
    relation_offset_m = 0.15
    frame_id = "gemini336_color_optical_frame"
    support_points = np.array(
        [[x, y, 1.42] for x in np.linspace(0.2, 0.4, 8) for y in np.linspace(0.6, 0.8, 8)],
        dtype=float,
    )
    reference_positions = {
        "cereal box": np.array([0.35, 0.7, 1.18], dtype=float),
    }

    result = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton left to cereal box",
        placement_pointclouds=support_points,
        reference_positions_by_name=reference_positions,
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
        shelf_board_heights=[0.72, 1.07, 1.42, 1.77],
    )

    assert result["coordinates"] == pytest.approx([0.2, 0.7, 1.07])


def test_reference_relation_records_reference_object_metadata() -> None:
    relation_offset_m = 0.15
    frame_id = "gemini336_color_optical_frame"
    support_points = np.array(
        [[x, y, 1.0] for x in np.linspace(0.2, 0.4, 8) for y in np.linspace(0.6, 0.8, 8)],
        dtype=float,
    )
    reference_positions = {
        "cereal box": np.array([0.35, 0.7, 1.18], dtype=float),
    }

    result = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton left to cereal box",
        placement_pointclouds=support_points,
        reference_positions_by_name=reference_positions,
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
    )

    assert result["reference_object"] == "cereal box"
    assert result["relation"] == "left"
    assert result["relation_offset_m"] == relation_offset_m
    assert result["reference_position"] == pytest.approx(
        {"x": 0.35, "y": 0.7, "z": 1.18}
    )
