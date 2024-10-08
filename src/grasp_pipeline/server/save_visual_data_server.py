#!/usr/bin/env python
"""This script saves the color image, depth image and point cloud in world frame to file.
"""
import rospy
import numpy as np
import os
import time
import open3d as o3d
import ros_numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros
import tf.transformations as tft

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

from grasp_pipeline.srv import *

# This module should save, restore and display depth images as well as point clouds


class VisualDataSaver():
    def __init__(self):
        rospy.init_node("save_visual_data_server_node")
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # essential, otherwise next line crashes

        self.scene_pcd_topic = rospy.get_param('scene_pcd_topic', default='/camera/depth/points')
        # as stated in grasp-pipeline/launch/grasp_pipeline_servers_real.launch, the pcd_topic for
        # realsense is either /camera/depth/points from simulation or the other one in real world
        if self.scene_pcd_topic == '/camera/depth/points' or self.scene_pcd_topic == '/camera/depth/color/points':
            pcd_frame = 'camera_depth_optical_frame'
        elif self.scene_pcd_topic == '/depth_registered_points':
            pcd_frame = 'camera_color_optical_frame'
        else:
            rospy.logerr(
                'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

        self.transform_camera_world = self.tf_buffer.lookup_transform(
            'world', pcd_frame, rospy.Time())
        q = self.transform_camera_world.transform.rotation
        r = self.transform_camera_world.transform.translation
        self.world_T_camera = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        self.world_T_camera[:, 3] = [r.x, r.y, r.z, 1]
        print("world_T_camera:")
        print(self.world_T_camera)

        self.color_img_topic = rospy.get_param('color_img_topic',
                                               default='/camera/color/image_raw')
        self.depth_img_topic = rospy.get_param('depth_img_topic',
                                               default='/camera/depth/image_raw')

    def draw_pcd(self, pcd):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    def save_depth_img(self, depth_img, depth_img_save_path):
        depth_img_u16 = cv2.normalize(depth_img,
                                      depth_img,
                                      0,
                                      65535,
                                      cv2.NORM_MINMAX,
                                      dtype=cv2.CV_16UC1)
        cv2.imwrite(depth_img_save_path, depth_img_u16)

    def save_depth_arr(self, depth_img, depth_img_save_path):
        depth_img_save_path = depth_img_save_path.replace("png", "npy")
        depth_img *= 0.001
        np.save(depth_img_save_path, depth_img)

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
        ############ Handle PCD ###############
        if req.scene_pcd_save_path is not '':
            if os.path.exists(req.scene_pcd_save_path):
                os.remove(req.scene_pcd_save_path)

            if req.scene_pcd.data == '':
                pcd = rospy.wait_for_message(self.scene_pcd_topic, PointCloud2, timeout=5)
            else:
                pcd = req.scene_pcd

            # Transform with ros_numpy
            start = time.time()
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(
                ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcd))
            del pcd

            # Transform point cloud from camera frame to world frame/ robot base frame
            pcd_o3d.transform(self.world_T_camera)

            p = np.asarray(pcd_o3d.points)
            colors = (-3.) * np.linspace(0.1, 0.9, p.shape[0]) - 0.05
            colors = np.exp(colors)
            colors = np.array([colors])
            pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(colors.T, (1, 3)))
            print("Ros numpy took: " + str(time.time() - start))
            #self.draw_pcd(pcd_o3d)
            self.save_pcd(pcd_o3d, req.scene_pcd_save_path)

        ####### Handle depth and color ##########
        if os.path.exists(req.depth_img_save_path):
            os.remove(req.depth_img_save_path)
        if os.path.exists(req.color_img_save_path):
            os.remove(req.color_img_save_path)

        # Now check if the request contains data for pcd, depth and rgb. If not grab the current scene from the topic
        if req.depth_img.data == '':
            depth_img = rospy.wait_for_message(self.depth_img_topic, Image, timeout=5)
        else:
            depth_img = req.depth_img
        if req.color_img.data == '':
            color_img = rospy.wait_for_message(self.color_img_topic, Image, timeout=5)
        else:
            color_img = req.color_img

        # Transform format in order to save data to disk
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
        color_img = self.bridge.imgmsg_to_cv2(color_img, "bgr8")

        # Actually save the stuff
        # self.save_depth_img(depth_img, req.depth_img_save_path)
        self.save_depth_arr(depth_img, req.depth_img_save_path)
        self.save_color_img(color_img, req.color_img_save_path)

        response = SaveVisualDataResponse()
        response.success = True
        return response

    def create_save_visual_data_service(self):
        rospy.Service('save_visual_data', SaveVisualData, self.handle_visual_data_saver)
        rospy.loginfo('Service save_visual_data:')
        rospy.loginfo('Ready to save your awesome visual data.')


if __name__ == "__main__":
    Saver = VisualDataSaver()

    Saver.create_save_visual_data_service()
    rospy.spin()
