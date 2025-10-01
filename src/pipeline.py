#!/usr/bin/env python

# Standard Library Imports
import math
import random
import rospy
import os
import time
import json
import re
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# ROS and ROS messages
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

# OpenAI and Audio Processing Libraries
from openai import OpenAI
import librosa
import soundfile as sf

# Computer Vision and Point Cloud Processing
import cv2
import open3d as o3d
import numpy as np
import base64
from matplotlib.colors import to_rgb
import tf2_ros

# Custom imports
from utils import *
from scene_acquisition import *
from agents import *
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Initialize OpenAI client
client = OpenAI()
rospy.init_node('empower_pipeline_final', anonymous=True)

# Task definitions
tasks = {
    "T1_task_1": "use the fire extinguisher",
    "T1_task_2": "lift the table",
    "T1_task_3": "detach the TV from the wall",
    "T1_task_4": "Exit climbing the stairs",
    "T2_task_1": "move the juice fruit to the left of the glass",
    "T2_task_2": "Fry an egg",
    "T2_task_3": "Give me the screwdriver",
    "T3_task_1": "Unplug the electrical cable",
    "T3_task_2": "Put screwdriver into the electric plug",
    "T3_task_3": "Pour some water on the pc",
    "icra": "serve the food"
}


class PipelineFinal():
    def __init__(self, task_id="task_1"):
        self.loader_instance = None
        self.COLORS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow'] * 3
        self.task_id = task_id
        if os.path.exists("last_id.txt"):
            with open("last_id.txt", "r") as f:
                self.marker_id_counter = int(f.read())
        else:
            self.marker_id_counter = 1
        self.task = tasks.get(task_id, "Pick the TV and laptop")
        
        # Set up logging directories
        self.setup_logging()
        
        # ROS components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        
        # Publishers
        self.publisher_centroid = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)
        self.publisher_maximum = rospy.Publisher("/pcl_maximum", MarkerArray, queue_size=100)
        self.publisher_names = rospy.Publisher("/pcl_names", MarkerArray, queue_size=100)
        
        # Initialize prompts paths
        self.visual_prompt_path = PROMPT_DIR + "visual_agent_prompt.txt"
        self.conversational_prompt_path = PROMPT_DIR + "conversational_prompt.txt"
        self.planner_prompt_path = PROMPT_DIR + "planner_prompt.txt"
        
    def setup_logging(self):
        """Setup logging directories and files"""
        self.save_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "experiments",
            self.task_id, 
            time.strftime("%Y%m%d-%H%M%S")
        )
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
            
        self.point_cloud_path = os.path.join(self.save_folder, "point_cloud.pcd")
        self.img_path = os.path.join(self.save_folder, "image.png")
        self.audio_path = os.path.join(self.save_folder, "whisper_audio.wav")
        

    def rgb_to_bgr(self, rgb_color):
        r, g, b = rgb_color
        return [b, g, r]

    def set_loader(self, loader_instance):
        self.loader_instance = loader_instance

    def show_mask(self, mask, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.numpy() * 255
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def find_bb_relation(self, relation_object):
        index_ = []
        for index, detection in self.dict_detections.items():
            if detection['label'].lower() in relation_object.lower():
                index_.append(index)
        return index_

    def get_R_and_T(self, trans):
        Tx_base = trans.transform.translation.x
        Ty_base = trans.transform.translation.y
        Tz_base = trans.transform.translation.z
        T = np.array([Tx_base, Ty_base, Tz_base])
        # Quaternion coordinates
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        # Rotation matrix
        R = 2 * np.array([[pow(qw, 2) + pow(qx, 2) - 0.5, qx * qy - qw * qz, qw * qy + qx * qz], [qw * qz + qx * qy, pow(qw, 2) + pow(qy, 2) - 0.5, qy * qz - qw * qx], [qx * qz - qw * qy, qw * qx + qy * qz, pow(qw, 2) + pow(qz, 2) - 0.5]])
        return R, T

    def set_marker(self, point, color, id, R, T):
        point = point / 1000
        transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        point = np.dot(transform, point)  # in xtion
        R = np.transpose(R)
        point = np.dot(R, point - T)  # in map

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time(0)
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.type = marker.SPHERE
        color = to_rgb(color)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.id = id

        return marker

    def set_names(self, marker, text):
        """
        Creates a text marker to display object names above the centroid
        Args:
            marker: The centroid marker to position the text above
            text: The text to display (object label)
        Returns:
            Marker: Text marker to display in RViz
        """
        text_marker = Marker()
        text_marker.header.frame_id = marker.header.frame_id
        text_marker.header.stamp = marker.header.stamp
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        # Position text slightly above the centroid
        text_marker.pose.position.x = marker.pose.position.x
        text_marker.pose.position.y = marker.pose.position.y
        text_marker.pose.position.z = marker.pose.position.z + 0.1  # 10cm above the centroid

        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        text_marker.text = text
        text_marker.scale.z = 0.15  # Text size
        text_marker.color = marker.color  # Same color as the centroid
        text_marker.id = marker.id 
        return text_marker

    # New methods from new_pipeline.py
    def depth_image_to_point_cloud(self, depth_image, camera_intrinsics):
        height, width = depth_image.shape
        points = []

        v, u = np.indices((height, width))

        x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
        y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
        z = depth_image

        points = np.dstack((x, y, z)).reshape(-1, 3)

        return points

    def capture_pcd(self):
        msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
        camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
        proj_matrix = camera_info.K
        fx = proj_matrix[0]
        fy = proj_matrix[4]
        cx = proj_matrix[2]
        cy = proj_matrix[5]

        camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        img_g = self.bridge.imgmsg_to_cv2(msg_img_g)
        depth_image = np.asarray(img_g)
        point_cloud = self.depth_image_to_point_cloud(depth_image, camera_intrinsics)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        pcd.transform(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))
        o3d.io.write_point_cloud(self.point_cloud_path, pcd)
        return pcd

    def create_graph(self, json_structure, save_path):
        G = nx.Graph()
        for obj in json_structure['Objects']:
            label = obj['Label']
            relations = obj['Relations']
            G.add_node(label)

            for relation in relations:
                relation_attribute, label2 = relation.split(";")[0], relation.split(";")[1]
                G.add_edge(label, label2, relation=relation_attribute)

        pos = nx.spring_layout(G, k=0.5, iterations=50)  # Increase k for more space between nodes

        nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", font_size=5, font_weight='bold', width=2)

        edge_labels = nx.get_edge_attributes(G, 'relation')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, font_color='blue')

        # Save the graph
        plt.savefig(save_path)
        plt.close()

    def say_phrase(self, data):
        client_tts = SimpleActionClient('/tts', TtsAction)
        client_tts.wait_for_server()
        goal = TtsGoal()
        goal.rawtext.text = data
        goal.rawtext.lang_id = "en_GB"
        client_tts.send_goal_and_wait(goal)

    def listen_for(self, seconds):
        try:
            seconds = seconds // 3
            rospy.loginfo(f"Received audio message")
            whisper_audio = []
            for _ in range(seconds):
                data = rospy.wait_for_message("/data_topic", Float32MultiArray)
                audio_array = np.array(data.data)
                amplitude_audio = np.abs(audio_array)
                amplitude_audio = np.mean(amplitude_audio)
                rospy.loginfo(f"Amplitude: {amplitude_audio}")
                if np.max(amplitude_audio) < 0.001:
                    rospy.loginfo("Audio troppo silenzioso, ignorato.")
                else:
                    audio = librosa.resample(audio_array, orig_sr=44100, target_sr=16000).astype(np.float32)
                    whisper_audio = np.concatenate((whisper_audio, np.array(audio, dtype=np.float32)))

            sf.write(self.audio_path, whisper_audio, 16000)

            whisper_audio = open(self.audio_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=whisper_audio,
                language='en')
            os.system("rm " + self.audio_path)
            return transcription.text
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")


    def color_pcl(self, image, pcd, detections):
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(to_rgb('gray'), (len(pcd.points), 1))
        )

        h, w, _ = image.shape
        masks, masks_flipped = [], []

        for key in detections.keys():
            mask = detections[key]['mask']
            masks.append(mask[:, :, 0])
            masks_flipped.append(cv2.flip(mask[:, :, 0], 1))

        camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
        proj_matrix = camera_info.K
        fx, fy = proj_matrix[0], proj_matrix[4]
        cx, cy = proj_matrix[2], proj_matrix[5]

        colors_dict = {}
        for idx, point in enumerate(pcd.points):
            x_, y_, z_ = point
            if abs(z_) < 1e-3:
                continue
            x = int((fx * x_ / z_) + cx)
            y = int((fy * y_ / z_) + cy)
            if not (0 <= x < w and 0 <= y < h):
                continue

            for id_color, (mask, mask_flipped) in enumerate(zip(masks, masks_flipped)):
                if mask[y, x] != 0:
                    image[y, x] = self.rgb_to_bgr(
                        [int(c * 255) for c in to_rgb(self.COLORS[id_color])]
                    )
                if mask_flipped[y, x] != 0:
                    if id_color not in colors_dict:
                        colors_dict[id_color] = []
                    pcd.colors[idx] = to_rgb(self.COLORS[id_color])
                    colors_dict[id_color].append(np.array([x_, y_, z_], dtype=float))

        array_centroids = MarkerArray()
        array_name      = MarkerArray()

        trans = self.tf_buffer.lookup_transform(
            "xtion_rgb_optical_frame", "map", rospy.Time(0), rospy.Duration(2.0)
        )
        Rx2m, Tx2m = self.get_R_and_T(trans)

        for id_color, list_points in colors_dict.items():
            if len(list_points) == 0:
                continue

            label = detections.get(id_color, {}).get('label', f"Object {id_color}")
            already_named = False

            # prendi solo 1 punto ogni 20
            for j, point in enumerate(list_points):
                if j % 20 != 0:
                    continue
                self.marker_id_counter += 1
                with open("last_id.txt", "w") as f:
                    f.write(str(self.marker_id_counter))

                new_marker = self.set_marker(point, self.COLORS[id_color],
                                            self.marker_id_counter, Rx2m, Tx2m)
                array_centroids.markers.append(new_marker)

                if not already_named:
                    name_marker = self.set_names(new_marker, label)
                    self.marker_id_counter += 1
                    with open("last_id.txt", "w") as f:
                        f.write(str(self.marker_id_counter))

                    name_marker.id = self.marker_id_counter 
                    array_name.markers.append(name_marker)
                    already_named = True

            rospy.loginfo(f"Spawned {len(list_points)//20} points for object {label}")

        # Pubblica
        self.publisher_centroid.publish(array_centroids)
        self.publisher_names.publish(array_name)

        cv2.imwrite(os.path.join(self.save_folder, 'colored_image.png'), image)
        return


    def run_pipeline(self):
        # Acquire image
        image, depth = acquire_image()
        print("Image acquired")
        
        # Save RGB image for processing
        cv2.imwrite(self.img_path, image)
        
        pcd = self.capture_pcd()
        print("Point cloud captured")
        
        json_scene = ""
        self.run_traditional_detection(image, depth, json_scene)


    def get_classes(self, object_relations):
        relation_list = []

        # Iterate through each object in the list
        for obj in object_relations:
            if not self.is_in_list(obj["Label"], relation_list):
                relation_list.append(obj["Label"])

        return relation_list


    def run_traditional_detection(self, image, depth, scene):
        """Run traditional object detection pipeline"""

        
        labels = ["table",
        "tv", 
        "chips",
        "cola can",
        "water bottle",
        "cup",
        "juice box", 
        "glasses",
        "wine bottle",
        "snack", 
        "glass",
        "usb",
        "paper",
        "napkin",
        "spray"]

        print("Detected labels: ", labels)
        
        # Time YOLO inference
        yolo_start = time.time()
        self.loader_instance.yolow_model.set_classes(labels)
        bboxs, labels_idx, scores = self.loader_instance.yolow_model.predict(image)
        # Time detection processing
        detection_processing_start = time.time()
        
        # Process detections
        masked_image = image.copy()
        image_with_bbox = image.copy()
        image_yolow = image.copy()
        self.dict_detections = {}
        overlay_ = masked_image

        index_detection = 0
        print("YOLO inference results:")
        print("bboxs: ", bboxs)
        print("scores: ", scores)
        print("labels_idx: ", labels_idx)
        for i, (bbox, score, cls_id) in enumerate(zip(bboxs, scores, labels_idx)):
            x1, y1, x2, y2 = bbox

            label = labels_idx[i]
            
            masks, _ = self.loader_instance.vit_sam_model(masked_image, bbox)
            if index_detection not in self.dict_detections.keys():
                self.dict_detections[index_detection] = {'bbox': None, 'label': None}
                self.dict_detections[index_detection]['bbox'] = bbox
                self.dict_detections[index_detection]['label'] = label
                cv2.rectangle(image_yolow, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(image_yolow, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                
            # Convert binary mask to 3-channel image
            mask_count = 0
            for mask in masks:
                binary_mask = self.show_mask(mask)
                overlay = masked_image
                self.dict_detections[index_detection]['mask'] = binary_mask
                overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)
                mask_file = os.path.join(self.save_folder, f"rgb_{index_detection}.jpg")
                cv2.imwrite(mask_file, overlay)
                overlay_ = cv2.addWeighted(overlay_, 1, binary_mask, 0.5, 0)
                mask_count += 1
            print("mask")
            index_detection += 1
        self.data_reordered = self.dict_detections
        cv2.imwrite(os.path.join(self.save_folder, 'yolow_image.png'), image_yolow)

        print("Detection images saved.")
        print("Pipeline finished.", os.path.join(self.save_folder, 'yolow_image.png'))

        # Time point cloud coloring
        pcl_coloring_start = time.time()
        self.color_pcl(image, depth, self.data_reordered)
        pcl_coloring_duration = time.time() - pcl_coloring_start
        
        


