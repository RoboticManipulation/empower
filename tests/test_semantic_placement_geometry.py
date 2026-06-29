from __future__ import annotations

import numpy as np
import pytest

from empower.semantic_placement_geometry import (
    extract_labels_per_step,
    get_semantic_placement_coordinates_from_plan,
    reference_names_match,
    resolve_empower_reference,
    resolve_reference_position_from_action,
)


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


def test_missing_reference_object_marks_failure_without_coordinates() -> None:
    relation_offset_m = 0.15
    frame_id = "gemini336_color_optical_frame"
    support_points = np.array(
        [[x, y, 1.0] for x in np.linspace(0.2, 0.4, 8) for y in np.linspace(0.6, 0.8, 8)],
        dtype=float,
    )

    result = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton left to master chef container",
        placement_pointclouds=support_points,
        reference_positions_by_name={},
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
    )

    assert result["success"] is False
    assert result["failure_reason"] == "Reference object not found: master chef container"
    assert "coordinates" not in result
    assert result["reference_object"] == "master chef container"


def test_extract_labels_per_step_builds_original_main_ngrams() -> None:
    labels = extract_labels_per_step("DROP ketchup bottle left to Master Chef seasoning bottle")
    assert labels == [
        "ketchup",
        "ketchup bottle",
        "bottle left",
        "left to",
        "to Master",
        "Master Chef",
        "Chef seasoning",
        "seasoning bottle",
        "bottle",
    ]


def test_reference_names_match_follows_find_bb_relation_rules() -> None:
    assert reference_names_match("master chef seasoning bottle", "seasoning bottle")
    assert reference_names_match("master chef seasoning bottle", "chef")
    assert not reference_names_match("seasoning bottle", "master chef seasoning bottle")
    assert not reference_names_match("ketchup bottle", "seasoning bottle")


def test_resolve_reference_position_from_action_matches_ngram_key() -> None:
    reference_positions = {
        "seasoning bottle": np.array([0.1, 0.2, 0.3], dtype=float),
        "spray bottle": np.array([0.4, 0.5, 0.6], dtype=float),
    }
    matched_name, point = resolve_reference_position_from_action(
        "DROP ketchup bottle left to Master Chef seasoning bottle",
        reference_positions,
        reference_object="master chef seasoning bottle",
    )

    assert matched_name == "seasoning bottle"
    assert point == pytest.approx(np.array([0.1, 0.2, 0.3]))


def test_resolve_reference_position_from_action_matches_relabeled_name_via_ngram() -> None:
    reference_positions = {
        "chef seasoning": np.array([0.1, 0.2, 0.3], dtype=float),
        "master chef seasoning bottle": np.array([0.4, 0.5, 0.6], dtype=float),
    }
    matched_name, point = resolve_reference_position_from_action(
        "DROP ketchup bottle left to master chef seasoning bottle",
        reference_positions,
        reference_object="master chef seasoning bottle",
    )

    assert matched_name == "master chef seasoning bottle"
    assert point == pytest.approx(np.array([0.4, 0.5, 0.6]))


def test_resolve_reference_position_from_action_substring_prefers_longest_label() -> None:
    reference_positions = {
        "master chef seasoning": np.array([0.1, 0.2, 0.3], dtype=float),
        "master chef seasoning bottle": np.array([0.4, 0.5, 0.6], dtype=float),
    }
    matched_name, point = resolve_reference_position_from_action(
        "DROP ketchup bottle left to master chef seasoning bottle",
        reference_positions,
        reference_object="master chef seasoning bottle",
    )

    assert matched_name == "master chef seasoning bottle"
    assert point == pytest.approx(np.array([0.4, 0.5, 0.6]))


def test_plan_resolution_uses_matched_detection_label() -> None:
    relation_offset_m = 0.15
    frame_id = "gemini336_color_optical_frame"
    support_points = np.array(
        [[x, y, 1.0] for x in np.linspace(0.2, 0.4, 8) for y in np.linspace(0.6, 0.8, 8)],
        dtype=float,
    )
    reference_positions = {
        "seasoning bottle": np.array([0.35, 0.7, 1.18], dtype=float),
    }

    result = get_semantic_placement_coordinates_from_plan(
        "DROP ketchup bottle left to Master Chef seasoning bottle",
        placement_pointclouds=support_points,
        reference_positions_by_name=reference_positions,
        relation_offset_m=relation_offset_m,
        frame_id=frame_id,
    )

    assert result["success"] is True
    assert result["reference_object"] == "seasoning bottle"


def test_resolve_reference_position_from_action_matches_relabeled_full_name() -> None:
    reference_positions = {
        "Jif peanut butter jar": np.array([0.35, 0.7, 1.18], dtype=float),
    }
    matched_name, point = resolve_reference_position_from_action(
        "DROP ketchup bottle right to Jif peanut butter jar",
        reference_positions,
        reference_object="jif peanut butter jar",
    )

    assert matched_name == "Jif peanut butter jar"
    assert point == pytest.approx(np.array([0.35, 0.7, 1.18]))


def test_resolve_empower_reference_uses_action_line_matching() -> None:
    reference_positions = {
        "seasoning bottle": np.array([0.35, 0.7, 1.18], dtype=float),
    }
    matched_name, point = resolve_empower_reference(
        action_line="DROP ketchup bottle left to Master Chef seasoning bottle",
        relation="left",
        reference_positions=reference_positions,
        grasp_object="ketchup bottle",
    )

    assert matched_name == "seasoning bottle"
    assert point == pytest.approx(np.array([0.35, 0.7, 1.18]))
