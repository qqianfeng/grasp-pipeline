#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image, PointCloud2


class ServerUnitTester():
    def __init__(self):
        print('Started Unit Tests')
        rospy.init_node('server_unit_tester')
        self.test_count = 0
        self.bridge = CvBridge

    def test_manage_gazebo_scene_server(self, object_name, object_pose_array, object_model_name, dataset, model_type):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)
        update_gazebo_object = rospy.ServiceProxy(
            'update_gazebo_object', UpdateObjectGazebo)

        res = update_gazebo_object(
            object_name, object_pose_array, object_model_name, model_type, dataset)

        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def test_save_visual_data_server(self, pc_save_path, depth_save_path, color_save_path):
        self.test_count += 1
        print('Running test_save_visual_data_server, test number %d' %
              self.test_count)
        # Receive one message from depth, color and pointcloud topic, not registered
        msg_depth = rospy.wait_for_message("/camera/depth/image_raw", Image)
        msg_color = rospy.wait_for_message("/camera/color/image_raw", Image)
        msg_pc = rospy.wait_for_message(
            "/depth_registered/points", PointCloud2)
        print('Received depth, color and point cloud messages')
        # Send to server and wait for response
        save_visual_data = rospy.ServiceProxy(
            'save_visual_data', SaveVisualData)
        res = save_visual_data(False, msg_pc, msg_depth, msg_color,
                               pc_save_path, depth_save_path, color_save_path)
        # Print result
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def arm_moveit_planner_test(self):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)

        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def template_test(self):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)

        result = 'SUCCEDED' if res else 'FAILED'
        print(result)


if __name__ == '__main__':
    # Define variables for testing
    # Test spawn/delete Gazebo
    object_name = 'mustard_bottle'
    object_pose_array = [0., 0., 0., 1, 0., 0.]
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'
    # Test save visual data
    pc_save_path = '/home/vm/test_cloud.pcd'
    depth_save_path = '/home/vm/test_depth.pgm'
    color_save_path = '/home/vm/test_color.ppm'
    # Tester
    sut = ServerUnitTester()
    # Test spawning and deleting of objects
    # sut.test_manage_gazebo_scene_server(
    #    object_name, object_pose_array, object_model_name, dataset, model_type)
    # Test visual data save server
    sut.test_save_visual_data_server(
        pc_save_path, depth_save_path, color_save_path)
