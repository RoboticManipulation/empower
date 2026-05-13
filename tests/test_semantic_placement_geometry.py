from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


EMPOWER_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EMPOWER_SRC))

from semantic_placement_wrapper import (  # noqa: E402
    DEFAULT_RELATION_OFFSET_M,
    get_semantic_placement_coordinates_from_plan,
)


def test_left_right_offsets_follow_camera_x_axis_without_support_clipping() -> None:
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
    )
    right = get_semantic_placement_coordinates_from_plan(
        "DROP milk carton right to viva paper towel roll",
        placement_pointclouds=support_points,
        placement_surface_height_m=1.0,
        reference_positions_by_name=reference_positions,
    )

    assert left["coordinates"] == pytest.approx(
        [0.05 - DEFAULT_RELATION_OFFSET_M, -0.25, 1.0]
    )
    assert right["coordinates"] == pytest.approx(
        [0.05 + DEFAULT_RELATION_OFFSET_M, -0.25, 1.0]
    )
