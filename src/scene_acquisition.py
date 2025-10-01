import rospy
from utils.config import *
import numpy as np
from cv_bridge import CvBridge
import cv2
import open3d as o3d
from sensor_msgs.msg import Image, CameraInfo

bridge = CvBridge()


def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    height, width = depth_image.shape
    points = []

    v, u =  np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points

def acquire_image():
    msg_img = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image)
    img = bridge.imgmsg_to_cv2(msg_img, "bgr8")

    img_path = LOG_DIR+'scan.jpg'
    cv2.imwrite(img_path, img)

    msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    proj_matrix = camera_info.K   

    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]

    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    img_g = bridge.imgmsg_to_cv2(msg_img_g)
    depth_image = np.asarray(img_g)

    point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))

    o3d.io.write_point_cloud(LOG_DIR+"depth_pointcloud.pcd", pcd)

    return img, pcd

def local_acquire_image(path):
    img_path = path + "scan.jpg"
    depth_path = path + "depth_pointcloud.pcd"

    img = cv2.imread(img_path)
    depth_image = o3d.io.read_point_cloud(depth_path)
    depth_image.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))

    return img, depth_image