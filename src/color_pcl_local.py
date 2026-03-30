#!/usr/bin/env python3
"""
ROS-free replacement for color_pcl.py.

Reads camera intrinsics from output/<use_case>/camera_info.json (written by
prepare_local_data.py) instead of subscribing to /xtion/depth/camera_info.
Everything else is identical to color_pcl.py.

Usage (no roscore needed):
    cd ~/ros1/codebase_ros1_empower/empower/src
    USE_CASE=order_by_height python3 color_pcl_local.py
  or just set the env var before launching:
    export USE_CASE=order_by_height
    python3 color_pcl_local.py
"""

import json
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from matplotlib.colors import to_rgb

from paths import IMAGES_DIR, OUTPUT_DIR

USE_CASE = os.environ.get("USE_CASE", "order_by_height")
SCAN_DIR = IMAGES_DIR + USE_CASE + "/"
DUMP_DIR = OUTPUT_DIR + USE_CASE + "/"

COLORS = ["red", "green", "blue", "magenta", "cyan", "yellow"] * 3


def rgb_to_bgr(rgb_color):
    r, g, b = rgb_color
    return [b, g, r]


def load_intrinsics() -> tuple:
    cam_path = os.path.join(DUMP_DIR, "camera_info.json")
    if not os.path.exists(cam_path):
        sys.exit(
            f"[ERROR] {cam_path} not found.\n"
            "Run prepare_local_data.py first, or edit INTRINSICS in that script."
        )
    with open(cam_path) as f:
        info = json.load(f)
    fx = float(info["fx"])
    fy = float(info["fy"])
    cx = float(info["cx"])
    cy = float(info["cy"])
    print(f"[OK] Intrinsics loaded from {cam_path}: fx={fx} fy={fy} cx={cx} cy={cy}")
    return fx, fy, cx, cy


def run() -> None:
    fx, fy, cx, cy = load_intrinsics()

    pcd = o3d.io.read_point_cloud(DUMP_DIR + "depth_pointcloud.pcd")
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(to_rgb("gray"), (len(pcd.points), 1))
    )

    image = cv2.imread(SCAN_DIR + "scan.jpg")
    h, w, _ = image.shape

    with open(DUMP_DIR + "detection.pkl", "rb") as f:
        detections = pickle.load(f)

    masks, masks_flipped = [], []
    for key in detections.keys():
        mask = detections[key]["mask"]
        masks.append(mask[:, :, 0])
        masks_flipped.append(cv2.flip(mask[:, :, 0], 1))

    colors_dict: dict = {}

    for idx, point in enumerate(pcd.points):
        if point[2] < 0:
            x_ = point[0]
            y_ = point[1]
            z_ = point[2]
            x = int((fx * x_ / z_) + cx)
            y = int((fy * y_ / z_) + cy)

            for id_color, (mask, mask_flipped) in enumerate(
                zip(masks, masks_flipped)
            ):
                if 0 <= x < w and 0 <= y < h and mask[y, x] != 0:
                    image[y, x] = rgb_to_bgr(
                        [int(c * 255) for c in to_rgb(COLORS[id_color])]
                    )

                if 0 <= x < w and 0 <= y < h and mask_flipped[y, x] != 0:
                    if id_color not in colors_dict:
                        colors_dict[id_color] = []
                    pcd.colors[idx] = [
                        int(c * 255) for c in to_rgb(COLORS[id_color])
                    ]
                    colors_dict[id_color].append(point)

    cv2.imwrite(DUMP_DIR + "colored_image.png", image)
    o3d.io.write_point_cloud(DUMP_DIR + "colored_pcl.pcd", pcd)
    with open(DUMP_DIR + "colors_dict.pkl", "wb") as f:
        pickle.dump(colors_dict, f)

    print(f"[OK] colored_image.png, colored_pcl.pcd, colors_dict.pkl → {DUMP_DIR}")


if __name__ == "__main__":
    run()
