#!/usr/bin/env python3
"""Run semantic placement and visualize the returned coordinate in Open3D."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from semantic_placement_config import DEFAULT_FRAME_ID
from semantic_placement_config import DEFAULT_MODE
from semantic_placement_config import SUPPORTED_DETECTOR_BACKENDS
from semantic_placement_config import SUPPORTED_SEMANTIC_PLACEMENT_MODES
from semantic_placement_wrapper import EmpowerSemanticPlacementWrapper


def _create_segmentation():
    segmentation_module = import_module("geo_sem_place.segmentation.segmentation")
    return segmentation_module.Segmentation()


def main() -> None:
    args = _parse_args()

    segmentation = None
    if args.detector_backend == "sam3":
        segmentation = _create_segmentation()

    wrapper = EmpowerSemanticPlacementWrapper(
        detector_backend=args.detector_backend,
        mode=args.mode,
        relation_offset_m=args.relation_offset_m,
        segmentation=segmentation,
    )
    wrapper.set_inputs(
        grasp_object=args.grasp_object,
        image=args.image,
        pointcloud=args.pointcloud,
        camera_info=args.camera_info,
        frame_id=args.frame_id,
        images_root=args.images_root,
        output_root=args.output_root,
    )
    wrapper.run()
    wrapper.save_outputs(
        write_prefix=args.write_prefix,
        voxel_size=args.voxel_size,
        marker_radius=args.marker_radius,
        show_window=not args.no_window,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set semantic placement inputs, run prediction, and show the returned placement "
            "coordinate in the input point cloud. Red sphere = returned coordinate."
        )
    )
    parser.add_argument("image", type=Path, help="Placement scene RGB image")
    parser.add_argument("pointcloud", type=Path, help="Placement scene .pcd/.ply file")
    parser.add_argument(
        "--grasp-object",
        required=True,
        help="Already-held object name, for example 'milk carton'",
    )
    parser.add_argument(
        "--camera-info",
        type=Path,
        help="Optional camera_info.json for image/point-cloud grounding",
    )
    parser.add_argument(
        "--frame-id",
        default=DEFAULT_FRAME_ID,
        help="Frame label for the returned coordinate",
    )
    parser.add_argument(
        "--detector-backend",
        "--detector",
        choices=SUPPORTED_DETECTOR_BACKENDS,
        required=True,
        help="Prompt-conditioned detector backend to use for grounding",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_SEMANTIC_PLACEMENT_MODES,
        default=DEFAULT_MODE,
        help=(
            "refined uses the refined single-reference logic; original uses "
            "the original Empower-style plan and centroid offsets"
        ),
    )
    parser.add_argument(
        "--relation-offset-m",
        "--offset-m",
        type=float,
        required=True,
        help="Left/right placement offset in meters.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Optional Empower images root for staging",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional Empower output root for staging",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Downsample voxel size in meters; use 0 to disable",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.03,
        help="Marker sphere radius in meters",
    )
    parser.add_argument(
        "--write-prefix",
        type=Path,
        help="Optional output prefix for marker .ply files and 2D overlay PNG",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Only run wrapper and print/write files; do not open Open3D",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
