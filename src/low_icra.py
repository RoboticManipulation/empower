#!/usr/bin/env python3

import sys, copy
import rospy, moveit_commander
import numpy as np
import cv2
import open3d as o3d
import tf2_ros
import tf.transformations as tft

from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge

# Import primitive actions
from primitive_actions import *
# Import acquisizione scena e modelli
from scene_acquisition import acquire_image
from models import YoloWorld, VitSam
from utils.config import ENCODER_VITSAM_PATH, DECODER_VITSAM_PATH
from visualization_msgs.msg import Marker
bridge = CvBridge()
    

def grab(group, gripper):
    open_grippers(gripper)
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = init_pose.position.z - 0.08
    wp1.position.y = init_pose.position.y -0.015
    wp2 = copy.deepcopy(wp1)


    wp1.orientation = GRABBING_QUATERNION
    wp2.orientation = GRABBING_QUATERNION

    wp2.position.x = init_pose.position.x + 0.3
    waypoints = [wp1, wp2]
    reversed_waypoints = [init_pose]
    
    reach_waypoints(group, waypoints)
    close_grippers(gripper)
    reach_waypoints(group, reversed_waypoints) 
    
def drop(group, gripper):
    close_grippers(gripper)
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = init_pose.position.z - 0.09
    wp1.position.x = init_pose.position.x + 0.3
    wp1.position.y = init_pose.position.y +0.2
    wp2 = copy.deepcopy(wp1)


    wp1.orientation = GRABBING_QUATERNION
    wp2.orientation = GRABBING_QUATERNION

    waypoints = [wp1]
    reversed_waypoints = [init_pose]
    
    reach_waypoints(group, waypoints)
    open_grippers(gripper)
    reach_waypoints(group, reversed_waypoints) 

# === Main ===
def main():
    rospy.init_node("grasp_with_detection", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    # Init robot
    home()
    #back_init()
    robot = moveit_commander.RobotCommander()
    arm_torso_group = moveit_commander.MoveGroupCommander("arm_torso")
    gripper = moveit_commander.MoveGroupCommander("gripper")  

    #drop(arm_torso_group, gripper)
    #grab(arm_torso_group, gripper)




if __name__ == "__main__":
    main()
