#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft


def get_pose_stamped_from_array(pose_array, frame_id='/world'):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    quaternion = tft.quaternion_from_euler(pose_array[0], pose_array[1],
                                           pose_array[2])
    pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = quaternion
    pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = pose_array[
        3:]
    return pose_stamped


class ServerUnitTester():
    def __init__(self):
        print('Started Unit Tests')
        rospy.init_node('server_unit_tester')
        self.test_count = 0
        self.bridge = CvBridge

    def test_manage_gazebo_scene_server(self, object_name, object_pose_array,
                                        object_model_name, dataset,
                                        model_type):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)
        update_gazebo_object = rospy.ServiceProxy('update_gazebo_object',
                                                  UpdateObjectGazebo)

        res = update_gazebo_object(object_name, object_pose_array,
                                   object_model_name, model_type, dataset)

        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def test_save_visual_data_server(self, pc_save_path, depth_save_path,
                                     color_save_path):
        self.test_count += 1
        print('Running test_save_visual_data_server, test number %d' %
              self.test_count)
        # Receive one message from depth, color and pointcloud topic, not registered
        msg_depth = rospy.wait_for_message("/camera/depth/image_raw", Image)
        msg_color = rospy.wait_for_message("/camera/color/image_raw", Image)
        msg_pc = rospy.wait_for_message("/depth_registered/points",
                                        PointCloud2)
        print('Received depth, color and point cloud messages')
        # Send to server and wait for response
        save_visual_data = rospy.ServiceProxy('save_visual_data',
                                              SaveVisualData)
        res = save_visual_data(False, msg_pc, msg_depth, msg_color,
                               pc_save_path, depth_save_path, color_save_path)
        # Print result
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def create_moveit_scene_test(self, pose_stamped, object_mesh_path):
        self.test_count += 1
        print('Running test_create_moveit_scene_server, test number %d' %
              self.test_count)
        create_moveit_scene = rospy.ServiceProxy('create_moveit_scene',
                                                 ManageMoveitScene)
        req = ManageMoveitSceneRequest(create_scene=True,
                                       object_mesh_path=object_mesh_path,
                                       object_pose_world=pose_stamped)
        res = create_moveit_scene(req)
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def clean_moveit_scene_test(self):
        self.test_count += 1
        print('Running test_clean_moveit_scene_server, test number %d' %
              self.test_count)
        clean_moveit_scene = rospy.ServiceProxy('clean_moveit_scene',
                                                ManageMoveitScene)
        req = ManageMoveitSceneRequest(clean_scene=True)
        res = clean_moveit_scene(req)
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def control_hithand_config_test(self, hithand_joint_states):
        self.test_count += 1
        print('Running test_control_hithand_config_server, test number %d' %
              self.test_count)
        control_hithand_config = rospy.ServiceProxy('control_hithand_config',
                                                    ControlHithand)
        req = ControlHithandRequest(
            hithand_target_joint_state=hithand_joint_states)
        res = control_hithand_config(req)

        req = ControlHithandRequest(go_home=True)
        res = control_hithand_config(req)
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def arm_moveit_planner_test(self):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)
        res = True
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)

    def template_test(self):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' %
              self.test_count)
        res = True
        result = 'SUCCEDED' if res else 'FAILED'
        print(result)


if __name__ == '__main__':
    # +++ Define variables for testing +++
    # ++++++++++++++++++++++++++++++++++++
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
    # Test create_moveit_scene
    pose_stamped = get_pose_stamped_from_array(object_pose_array)
    datasets_base_path = '/home/vm/object_datasets'
    object_mesh_path = datasets_base_path + '/' + dataset + \
        '/models/' + object_model_name + '/google_16k/nontextured.stl'
    # Test control_hithand_config
    hithand_joint_states = JointState()
    hithand_joint_states.position = [0.2] * 20

    # Tester
    sut = ServerUnitTester()

    # Test spawning and deleting of objects
    # sut.test_manage_gazebo_scene_server(
    #    object_name, object_pose_array, object_model_name, dataset, model_type)

    # Test visual data save server
    # sut.test_save_visual_data_server(
    #    pc_save_path, depth_save_path, color_save_path)

    # Test moveit spawn object
    # sut.create_moveit_scene_test(pose_stamped, object_mesh_path)

    # Test moveit delete object
    # sut.clean_moveit_scene_test()

    # Test hithand control preshape/config
    # sut.control_hithand_config_test(hithand_joint_states)
