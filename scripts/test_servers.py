#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped, Pose
import tf.transformations as tft
from std_srvs.srv import SetBoolRequest, SetBool
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import open3d as o3d
from trajectory_smoothing.srv import *


def get_pose_stamped_from_array(pose_array, frame_id='/world'):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    quaternion = tft.quaternion_from_euler(pose_array[0], pose_array[1], pose_array[2])
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
        self.arm_moveit_plan = None
        self.rate_hz = 100
        self.dt = 1 / self.rate_hz
        self.loop_rate = rospy.Rate(self.rate_hz)
        self.max_acc = 0.5 * np.ones(7)
        self.max_vel = 0.8 * np.ones(7)
        self.joint_trajectory = None

    def test_manage_gazebo_scene_server(self, object_name, object_pose_array, object_model_name,
                                        dataset, model_type):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' % self.test_count)
        update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)

        res = update_gazebo_object(object_name, object_pose_array, object_model_name, model_type,
                                   dataset)

        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_save_visual_data_server(self, pc_save_path, depth_save_path, color_save_path):
        self.test_count += 1
        print('Running test_save_visual_data_server, test number %d' % self.test_count)
        # Receive one message from depth, color and pointcloud topic, not registered
        msg_depth = rospy.wait_for_message("/camera/depth/image_raw", Image)
        msg_color = rospy.wait_for_message("/camera/color/image_raw", Image)
        msg_pcd = rospy.wait_for_message("/depth_registered/points", PointCloud2)
        print('Received depth, color and point cloud messages')
        # Send to server and wait for response
        save_visual_data = rospy.ServiceProxy('save_visual_data', SaveVisualData)
        res = save_visual_data(False, msg_pcd, msg_depth, msg_color, pc_save_path, depth_save_path,
                               color_save_path)
        # Print result
        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_display_saved_point_cloud(self, pcd_save_path):
        self.test_count += 1
        print('Running test_display_saved_point_cloud, test number %d' % self.test_count)
        pcd = o3d.io.read_point_cloud(pcd_save_path)
        box = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
        box.paint_uniform_color([1, 0, 0])  # create box at origin
        box_cam = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
        box_cam.paint_uniform_color([0, 1, 0])
        box_cam.translate([0.8275, -0.996, 0.361])  # create box at camera location
        o3d.visualization.draw_geometries([pcd, box, box_cam])
        print('SUCCEEDED')

    def test_create_moveit_scene_server(self, pose_stamped, object_mesh_path):
        self.test_count += 1
        print('Running test_create_moveit_scene_server, test number %d' % self.test_count)
        create_moveit_scene = rospy.ServiceProxy('create_moveit_scene', ManageMoveitScene)
        req = ManageMoveitSceneRequest(create_scene=True,
                                       object_mesh_path=object_mesh_path,
                                       object_pose_world=pose_stamped)
        res = create_moveit_scene(req)
        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_clean_moveit_scene_server(self):
        self.test_count += 1
        print('Running test_clean_moveit_scene_server, test number %d' % self.test_count)
        clean_moveit_scene = rospy.ServiceProxy('clean_moveit_scene', ManageMoveitScene)
        req = ManageMoveitSceneRequest(clean_scene=True)
        res = clean_moveit_scene(req)
        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_control_hithand_config_server(self, hithand_target_joint_state):
        self.test_count += 1
        print('Running test_control_hithand_config_server, test number %d' % self.test_count)
        control_hithand_config = rospy.ServiceProxy('control_hithand_config', ControlHithand)
        # Control the hithand to the desired joint state
        req = ControlHithandRequest(hithand_target_joint_state=hithand_target_joint_state)
        res = control_hithand_config(req)
        # Check if the joint angles are within a small range of the desired angles
        hithand_current_joint_state = rospy.wait_for_message('/hithand/joint_states', JointState)
        reach_gap = np.array(hithand_target_joint_state.position) - \
            np.array(hithand_current_joint_state.position)
        rospy.loginfo('Gap between desired and actual joint position: ')
        rospy.loginfo('\n' + str(reach_gap))
        assert np.min(np.abs(reach_gap)) < 0.1
        rospy.loginfo('All gaps are smaller than 0.1')

        # Control the hithand back home
        req = ControlHithandRequest(go_home=True)
        res = control_hithand_config(req)
        # Check if the joint angles are within a small range of the desired angles
        hithand_current_joint_state = rospy.wait_for_message('/hithand/joint_states', JointState)
        reach_gap = np.array(hithand_current_joint_state.position)
        rospy.loginfo('Gap between desired and actual joint position: ')
        rospy.loginfo('\n' + str(reach_gap))
        assert np.min(np.abs(reach_gap)) < 0.1
        rospy.loginfo('All gaps are smaller than 0.1')

        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    # def test_table_object_segmentation_server_py37_conda(self):
    #     self.test_count = +1
    #     print(
    #         'Running test_table_object_segmentation_start_server, test number %d'
    #         % self.test_count)
    #     table_object_segmentation = rospy.ServiceProxy(
    #         'table_object_segmentation_start_server', SetBool)
    #     req = SetBoolRequest()
    #     req.data = True
    #     res = table_object_segmentation(req)
    #     result = 'SUCCEEDED' if res else 'FAILED'
    #     print(result)

    def test_table_object_segmentation_server(self, object_pcd_path):
        self.test_count = +1
        print('Running test_segment_object_server, test number %d' % self.test_count)
        if os.path.exists(object_pcd_path):
            os.remove(object_pcd_path)
        table_object_segmentation = rospy.ServiceProxy('segment_object', SegmentGraspObject)
        req = SegmentGraspObjectRequest()
        req.start = True
        res = table_object_segmentation(req)
        result = 'SUCCEEDED' if res.success else 'FAILED'

        assert os.path.exists(object_pcd_path)
        msg = rospy.wait_for_message('/segmented_object_size', Float64MultiArray, timeout=5)
        assert msg.data is not None
        msg = rospy.wait_for_message('/segmented_object_bounding_box_corner_points',
                                     Float64MultiArray,
                                     timeout=5)
        assert msg.data is not None

        print(result)

    def test_generate_hithand_preshape_server(self):
        self.test_count = +1
        print('Running test_generate_hithand_preshape_server, test number %d' % self.test_count)
        generate_hithand_preshape = rospy.ServiceProxy('generate_hithand_preshape', GraspPreshape)
        req = GraspPreshapeRequest()
        req.sample = True
        res = generate_hithand_preshape(req)
        result = 'SUCCEEDED' if res else 'FAILED'
        ## Check what else should be happening here, what should be published etc and try to visualize it
        msg = rospy.wait_for_message('/publish_box_points', MarkerArray, timeout=5)
        print(result)

    def test_arm_moveit_cartesian_pose_planner_server(self,
                                                      pose,
                                                      go_home=False,
                                                      place_goal_pose=None):
        self.test_count = +1
        print('Running test_arm_moveit_cartesian_pose_planner_server, test number %d' %
              self.test_count)

        arm_moveit_cartesian_pose_planner = rospy.ServiceProxy('arm_moveit_cartesian_pose_planner',
                                                               PalmGoalPoseWorld)
        req = PalmGoalPoseWorldRequest()
        if go_home:
            req.go_home = True
        elif place_goal_pose is not None:
            req.palm_goal_pose_world = place_goal_pose
        else:
            req.palm_goal_pose_world = pose
        res = arm_moveit_cartesian_pose_planner(req)
        self.joint_trajectory = res.plan_traj
        result = 'SUCCEEDED' if res.success else 'FAILED'

        print(result)

    def test_execute_joint_trajectory_server(self, smoothen_trajectory=False):
        self.test_count = +1
        print('Running test_execute_joint_trajectory_server, test number %d' % self.test_count)

        execute_joint_trajectory = rospy.ServiceProxy('execute_joint_trajectory',
                                                      ExecuteJointTrajectory)
        req = ExecuteJointTrajectoryRequest()
        req.smoothen_trajectory = smoothen_trajectory
        req.joint_trajectory = self.joint_trajectory
        res = execute_joint_trajectory(req)
        result = 'SUCCEEDED' if res.success else 'FAILED'
        print(result)

    def test_get_smooth_trajectory_server(self):
        self.test_count = +1
        print('Running test_get_smooth_trajectory_server, test number %d' % self.test_count)

        smoothen_trajectory = rospy.ServiceProxy('/get_smooth_trajectory', GetSmoothTraj)
        res = smoothen_trajectory(self.joint_trajectory, self.max_acc, self.max_vel, 0.1, 0.01)

        self.joint_trajectory = res.smooth_traj  # Joint trajectory is now smooth
        result = 'SUCCEEDED' if res.success else 'FAILED'
        print(result)

    def template_test(self):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' % self.test_count)
        res = True
        result = 'SUCCEEDED' if res else 'FAILED'
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
    # Test segment table and object
    object_pcd_path = "/home/vm/object.pcd"
    # Test arm moveit cartesian pose planner
    pose_arm_moveit = Pose()
    pose_arm_moveit.position.x, pose_arm_moveit.position.y, pose_arm_moveit.position.z = 0.45, 0.1, 0.3
    pose_arm_moveit.orientation.x, pose_arm_moveit.orientation.y, pose_arm_moveit.orientation.z, pose_arm_moveit.orientation.w = 0.0, 0.0, 0.0, 1.0

    # Tester
    sut = ServerUnitTester()

    # Test spawning and deleting of objects
    # sut.test_manage_gazebo_scene_server(
    #    object_name, object_pose_array, object_model_name, dataset, model_type)

    # Test visual data save server
    # sut.test_save_visual_data_server(pc_save_path, depth_save_path,
    #                                 color_save_path)

    # Test display saved point cloud
    # sut.test_display_saved_point_cloud(pc_save_path)

    # Test moveit spawn object
    # sut.test_moveit_scene_server(pose_stamped, object_mesh_path)

    # Test moveit delete object
    # sut.test_clean_moveit_scene_server()

    # Test hithand control preshape/config
    # sut.test_control_hithand_config_server(hithand_joint_states)

    # Test table object segmentation
    # sut.test_table_object_segmentation_server(object_pcd_path)

    # Test hithand preshape generation server
    # sut.test_generate_hithand_preshape_server()

    # Test arm moveit cartesian pose planner
    sut.test_arm_moveit_cartesian_pose_planner_server(pose_arm_moveit)

    # Test arm joint trajectory execution
    sut.test_execute_joint_trajectory_server()

    # Test smoothen trajectory execution
    sut.test_get_smooth_trajectory_server()

    # Make robot go home
    sut.test_arm_moveit_cartesian_pose_planner_server(pose=None, go_home=True)

    # Test execution of smoothen trajectory
    sut.test_execute_joint_trajectory_server(smoothen_trajectory=True)
