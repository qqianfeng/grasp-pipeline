#!/usr/bin/env python
import rospy
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('grasp_pipeline')
import sys
sys.path.append(pkg_path + '/src')
from utils import pcd_from_ros_to_o3d

from sensor_msgs.msg import Image, PointCloud2
from grasp_pipeline.srv import *
from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
from time import time
import open3d as o3d

# This module should save, restore and display depth images as well as point clouds


class VisualDataSaver():
    def __init__(self):
        rospy.init_node("save_visual_data_server_node")
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # essential, otherwise next line crashes
        self.transform_camera_world = self.tf_buffer.lookup_transform(
            'world', 'camera_color_optical_frame', rospy.Time())
        self.scene_pcd_topic = rospy.get_param('scene_pcd_topic',
                                               default='/depth_registered/points')
        self.color_img_topic = rospy.get_param('color_img_topic',
                                               default='/camera/color/image_raw')
        self.depth_img_topic = rospy.get_param('depth_img_topic',
                                               default='/camera/depth/image_raw')

    def save_depth_img(self, depth_img, depth_img_save_path):
        depth_img_u16 = cv2.normalize(depth_img,
                                      depth_img,
                                      0,
                                      65535,
                                      cv2.NORM_MINMAX,
                                      dtype=cv2.CV_16UC1)
        cv2.imwrite(depth_img_save_path, depth_img_u16)

    def save_color_img(self, color_img, color_img_save_path):
        cv2.imwrite(color_img_save_path, color_img)

    def save_pcd(self, pcd, pcd_save_path):
        # Transform the point cloud into world frame
        o3d.io.write_point_cloud(pcd_save_path, pcd)

    def load_depth_img(self, depth_img_save_path):
        depth_img_u16 = cv2.imread(depth_img_save_path, -1)
        depth_img_f32 = cv2.normalize(depth_img_u16, depth_img_u16, dtype=cv2.CV_32FC1)
        return depth_img_f32

    def load_color_img(self, color_img_save_path):
        color_img = cv2.imread(color_img_save_path)
        return color_img

    def load_pcd(self, pcd_save_path):
        pcd = o3d.io.read_point_cloud(pcd_save_path)
        return pcd

    def handle_visual_data_saver(self, req):
        # First of all, delete any old files with the same name, because they are not being replaced when saving a second time
        if os.path.exists(req.scene_pcd_save_path):
            os.remove(req.scene_pcd_save_path)
        if os.path.exists(req.depth_img_save_path):
            os.remove(req.depth_img_save_path)
        if os.path.exists(req.color_img_save_path):
            os.remove(req.color_img_save_path)

        # Now check if the request contains data for pcd, depth and rgb. If not grab the current scene from the topic
        if req.scene_pcd.data == '':
            pcd = rospy.wait_for_message(self.scene_pcd_topic, PointCloud2, timeout=5)
        else:
            pcd = req.scene_pcd
        if req.depth_img.data == '':
            depth_img = rospy.wait_for_message(self.depth_img_topic, Image, timeout=5)
        else:
            depth_img = req.depth_img
        if req.color_img.data == '':
            color_img = rospy.wait_for_message(self.color_img_topic, Image, timeout=5)
        else:
            color_img = req.color_img

        # Transform the pointcloud message into world frame
        rospy.loginfo(pcd.header)
        start = time()
        pcd_world = do_transform_cloud(pcd, self.transform_camera_world)
        rospy.loginfo(pcd_world.header)
        print('Transforming the point cloud took: ')
        print(time() - start)

        # Transform format in order to save data to disk
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
        color_img = self.bridge.imgmsg_to_cv2(color_img, "bgr8")
        #start = time()
        pcd_world_o3d = pcd_from_ros_to_o3d(pcd_world)
        #print(time() - start)
        # Actually save the stuff
        self.save_depth_img(depth_img, req.depth_img_save_path)
        self.save_color_img(color_img, req.color_img_save_path)
        self.save_pcd(pcd_world_o3d, req.scene_pcd_save_path)

        response = SaveVisualDataResponse()
        response.save_visual_data_success = True
        return response

    def create_save_visual_data_service(self):
        rospy.Service('save_visual_data', SaveVisualData, self.handle_visual_data_saver)
        rospy.loginfo('Service save_visual_data:')
        rospy.loginfo('Ready to save your awesome visual data.')


if __name__ == "__main__":
    Saver = VisualDataSaver()
    Saver.create_save_visual_data_service()
    rospy.spin()
