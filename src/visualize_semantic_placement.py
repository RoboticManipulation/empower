#!/usr/bin/env python3
"""Run semantic placement and visualize the returned coordinate in Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_placement_wrapper import EmpowerSemanticPlacementWrapper
from geo_sem_place.segmentation.segmentation import Segmentation


def main() -> None:
    args = _parse_args()
    segmentation = Segmentation()
    ai = "mistral"
    wrapper = EmpowerSemanticPlacementWrapper(
        mode=args.mode,
        segmentation=segmentation,
        output_root=args.output_root,
        ai=ai,
        camera_info=args.camera_info,
        camera_extrinsics=args.camera_extrinsics,
    )
    wrapper.set_inputs(
        grasp_object=args.grasp_object,
        image=args.image,
        pointcloud=args.pointcloud,
        images_root=args.images_root,
    )
    wrapper.run()
    wrapper.save_outputs(
        write_prefix=args.write_prefix,
        show_window=False if args.no_window else None,
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
        "--camera-extrinsics",
        type=Path,
        help="Optional camera extrinsics (.json or .npy) for world-to-camera projection",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "refined"),
        help="Semantic placement mode alias. Omit to use configs/empower.yaml default_mode.",
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
