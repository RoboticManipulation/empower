#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2

class RGBDToPLY:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_img = None
        self.rgb_img = None
        self.intrinsic = None

        rospy.Subscriber('/xtion/depth/image_raw', Image, self.depth_cb)
        rospy.Subscriber('/xtion/rgb/image_rect_color', Image, self.rgb_cb)
        rospy.Subscriber('/xtion/depth/camera_info', CameraInfo, self.caminfo_cb)

    def caminfo_cb(self, msg):
        if self.intrinsic is None:  # Only do once
            K = np.array(msg.K).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(msg.width, msg.height, fx, fy, cx, cy)
            rospy.loginfo("Camera intrinsic received")
            # Unsubscribe from camera_info after receiving it once
            rospy.Subscriber('/xtion/depth/camera_info', CameraInfo, self.caminfo_cb).unregister()

    def depth_cb(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.try_create_ply()

    def rgb_cb(self, msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_create_ply()

    def try_create_ply(self):
        if self.depth_img is None or self.rgb_img is None or self.intrinsic is None:
            return
        
        # Convert depth to meters if uint16
        if self.depth_img.dtype == np.uint16:
            depth = self.depth_img.astype(np.float32) / 1000.0
        else:
            depth = self.depth_img.astype(np.float32)

        depth_o3d = o3d.geometry.Image(depth)
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsic)

        # Flip point cloud for visualization correctness
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        o3d.io.write_point_cloud("live_output.ply", pcd)
        rospy.loginfo("Saved live_output.ply")

        # Reset images to avoid saving every frame repeatedly
        self.depth_img = None
        self.rgb_img = None

if __name__ == "__main__":
    rospy.init_node("rgbd_to_ply_live", anonymous=True)
    node = RGBDToPLY()
    rospy.spin()
