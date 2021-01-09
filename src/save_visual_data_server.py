#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from grasp_pipeline.srv import *
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os

import open3d as o3d
from ctypes import *
from open3d_ros_helper import open3d_ros_helper as orh

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
        self.point_cloud_topic = rospy.get_param('point_cloud_topic',
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

    def save_point_cloud(self, point_cloud, point_cloud_save_path):
        # Transform the point cloud into world frame
        o3d.io.write_point_cloud(point_cloud_save_path, point_cloud)

    def load_depth_img(self, depth_img_save_path):
        depth_img_u16 = cv2.imread(depth_img_save_path, -1)
        depth_img_f32 = cv2.normalize(depth_img_u16, depth_img_u16, dtype=cv2.CV_32FC1)
        return depth_img_f32

    def load_color_img(self, color_img_save_path):
        color_img = cv2.imread(color_img_save_path)
        return color_img

    def load_point_cloud(self, point_cloud_save_path):
        point_cloud = o3d.io.read_point_cloud(point_cloud_save_path)
        return point_cloud

    ################ Handle below ########################


# NOTE: use do_transform_cloud, receive the transform and convert it to TransformStamped message

    def handle_visual_data_saver(self, req):
        # First of all, delete any old pcd files with the same name, because they are not being replaced when saving a second time
        if os.path.exists(req.point_cloud_save_path):
            os.remove(req.point_cloud_save_path)

        # Now check if the request contains data for pcd, depth and rgb. If not grab the current scene from the topic
        if req.point_cloud.data == '':
            point_cloud = rospy.wait_for_message(self.point_cloud_topic, PointCloud2, timeout=5)
        else:
            point_cloud = req.point_cloud
        if req.depth_img.data == '':
            depth_img = rospy.wait_for_message(self.depth_img_topic, Image, timeout=5)
        else:
            depth_img = req.depth_img
        if req.color_img.data == '':
            color_img = rospy.wait_for_message(self.color_img_topic, Image, timeout=5)
        else:
            color_img = req.color_img

        # Transform the pointcloud message into world frame
        rospy.loginfo(point_cloud.header)
        point_cloud_world = do_transform_cloud(point_cloud, self.transform_camera_world)
        rospy.loginfo(point_cloud_world.header)

        # Transform format in order to save data to disk
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
        color_img = self.bridge.imgmsg_to_cv2(color_img, "bgr8")
        point_cloud = orh.rospc_to_o3dpc(point_cloud_world)

        # Actually save the stuff
        self.save_depth_img(depth_img, req.depth_img_save_path)
        self.save_color_img(color_img, req.color_img_save_path)
        self.save_point_cloud(point_cloud, req.point_cloud_save_path)

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
