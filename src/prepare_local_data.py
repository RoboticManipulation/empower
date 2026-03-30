#!/usr/bin/env python3
"""
Stage pre-recorded data from data_ur5e/<session>/<frame_idx> into the
locations expected by the rest of the pipeline (models_cacher / execute_task /
color_pcl), bypassing the need for a live ROS bag or running create_pcl.py.

Usage:
    python3 prepare_local_data.py <use_case> <data_session> <frame_idx>

Example:
    python3 prepare_local_data.py order_by_height 5 0

This copies:
    data_ur5e/<session>/rgb_<frame>.png  →  images/<use_case>/scan.jpg
    data_ur5e/<session>/pc_<frame>.pcd   →  output/<use_case>/depth_pointcloud.pcd

It also writes a camera_info.json to output/<use_case>/ so that color_pcl.py
does not need a live ROS /camera_info topic.  Edit the INTRINSICS dict below
to match your actual camera calibration (check 'rostopic echo /d400/...
/camera_info' or your camera's calibration file).
"""

import argparse
import json
import os
import shutil
import sys

import cv2

from paths import IMAGES_DIR, OUTPUT_DIR, ROOT_DIR

DATA_ROOT = os.path.join(ROOT_DIR, "data_ur5e")

# Canonical intrinsics file saved alongside the recordings.
# If absent, the fallback values below are used instead.
_INTRINSICS_FILE = os.path.join(DATA_ROOT, "camera_intrinsics.json")

# ── Fallback camera intrinsics (only used if camera_intrinsics.json is missing)
_FALLBACK_INTRINSICS = {
    "fx": 1033.11,
    "fy": 1032.71,
    "cx": 966.88,
    "cy": 537.99,
    "width": 1920,
    "height": 1080,
}
# ─────────────────────────────────────────────────────────────────────────────


def _load_intrinsics() -> dict:
    """Read fx/fy/cx/cy from camera_intrinsics.json (K matrix) or fall back."""
    if os.path.exists(_INTRINSICS_FILE):
        with open(_INTRINSICS_FILE) as f:
            raw = json.load(f)
        K = raw["K"]  # row-major 3×3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        return {
            "fx": K[0],
            "fy": K[4],
            "cx": K[2],
            "cy": K[5],
            "width":  raw.get("width",  1920),
            "height": raw.get("height", 1080),
        }
    print(f"[WARN] {_INTRINSICS_FILE} not found — using fallback intrinsics.")
    return _FALLBACK_INTRINSICS


def prepare(use_case: str, session: str, frame_idx: int) -> None:
    src_dir = os.path.join(DATA_ROOT, session)
    if not os.path.isdir(src_dir):
        sys.exit(f"[ERROR] Data session not found: {src_dir}")

    rgb_src = os.path.join(src_dir, f"rgb_{frame_idx}.png")
    pcd_src = os.path.join(src_dir, f"pc_{frame_idx}.pcd")

    for path in (rgb_src, pcd_src):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Expected file not found: {path}")

    scan_dir = os.path.join(IMAGES_DIR, use_case)
    dump_dir = os.path.join(OUTPUT_DIR, use_case)
    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(dump_dir, exist_ok=True)

    # RGB → scan.jpg (pipeline reads JPEG via cv2 / base64)
    rgb_dst = os.path.join(scan_dir, "scan.jpg")
    img = cv2.imread(rgb_src)
    cv2.imwrite(rgb_dst, img)
    print(f"[OK] RGB  : {rgb_src}  →  {rgb_dst}")

    # Point cloud → depth_pointcloud.pcd
    pcd_dst = os.path.join(dump_dir, "depth_pointcloud.pcd")
    shutil.copy2(pcd_src, pcd_dst)
    print(f"[OK] PCD  : {pcd_src}  →  {pcd_dst}")

    # Camera intrinsics JSON (consumed by color_pcl_local.py)
    intrinsics = _load_intrinsics()
    cam_dst = os.path.join(dump_dir, "camera_info.json")
    with open(cam_dst, "w") as f:
        json.dump(intrinsics, f, indent=2)
    src_label = _INTRINSICS_FILE if os.path.exists(_INTRINSICS_FILE) else "fallback defaults"
    print(f"[OK] CAM  : intrinsics from '{src_label}'  →  {cam_dst}")
    print(f"           fx={intrinsics['fx']:.3f}  fy={intrinsics['fy']:.3f}  "
          f"cx={intrinsics['cx']:.3f}  cy={intrinsics['cy']:.3f}")

    print("\nReady. Now run:")
    print(f"  Terminal 4 → python3 models_cacher.py {use_case}")
    print( "  Terminal 5 → python3 execute_task.py")
    print( "  Terminal 6 → python3 color_pcl_local.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage local data for the empower pipeline.")
    parser.add_argument("use_case",   help="Use-case name, e.g. order_by_height")
    parser.add_argument("session",    help="Data session folder inside data_ur5e/, e.g. 5")
    parser.add_argument("frame_idx",  type=int, nargs="?", default=0,
                        help="Frame index to use (default: 0)")
    args = parser.parse_args()
    prepare(args.use_case, args.session, args.frame_idx)
