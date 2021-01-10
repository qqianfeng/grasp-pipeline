#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from std_srvs.srv import SetBoolRequest, SetBool
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import open3d as o3d
from trajectory_smoothing.srv import *


class ServerUnitTester():
    def __init__(self):
        print('Started Unit Tests')
        rospy.init_node('server_unit_tester')

        self.object_datasets_folder = rospy.get_param('object_datasets_folder',
                                                      default='/home/vm/object_datasets')
        self.color_img_save_path = rospy.get_param('color_img_save_path',
                                                   default='/home/vm/scene.ppm')
        self.depth_img_save_path = rospy.get_param('depth_img_save_path',
                                                   default='/home/vm/depth.pgm')
        self.object_point_cloud_path = rospy.get_param('object_point_cloud_path',
                                                       default='/home/vm/object.pcd')
        self.scene_point_cloud_path = rospy.get_param('scene_point_cloud_path',
                                                      default='/home/vm/scene.pcd')

        self.test_count = 0
        self.bridge = CvBridge
        self.arm_moveit_plan = None
        self.rate_hz = 100
        self.dt = 1 / self.rate_hz
        self.loop_rate = rospy.Rate(self.rate_hz)
        self.max_acc = 0.5 * np.ones(7)
        self.max_vel = 0.8 * np.ones(7)
        self.joint_trajectory = None

        self.spawn_object_x_min, self.spawn_object_x_max = 0.7, 0.75
        self.spawn_object_y_min, self.spawn_object_y_max = -0.1, 0.1
        self.table_height = 0
        self.home_joint_states = np.array([0, 0, 0, -1, 0, 1.9884, -1.57])
        self.heuristic_preshapes = None

    def get_pose_stamped_from_array(self, pose_array, frame_id='/world'):
        """Transforms an array pose into a ROS stamped pose.
        """
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        # RPY to quaternion
        pose_quaternion = tft.quaternion_from_euler(pose_array[0], pose_array[1], pose_array[2])
        pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, \
                pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = pose_quaternion[0], pose_quaternion[1], pose_quaternion[2], pose_quaternion[3]
        pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = \
                pose_array[3], pose_array[4], pose_array[5]
        return pose_stamped

    def get_pose_array_from_stamped(self, pose_stamped):
        """Transforms a stamped pose into a 6D pose array.
        """
        r, p, y = tft.euler_from_quaternion([
            pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w
        ])
        x_p, y_p, z_p = pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z
        pose_array = [r, p, y, x_p, y_p, z_p]
        return pose_array

# +++++++++++++++++ Part II: The tests +++++++++++++++++++++++++++++++++++

    def test_generate_random_object_pose_for_experiment(self):
        self.test_count += 1
        print('Running test_generate_random_object_pose_for_experiment, test number %d' %
              self.test_count)
        rand_x = np.random.uniform(self.spawn_object_x_min, self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min, self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [0, 0, rand_z_orientation, rand_x, rand_y, self.table_height]
        object_pose_stamped = self.get_pose_stamped_from_array(object_pose)
        self.spawned_object_pose = object_pose_stamped

        print('SUCCEEDED')

    def test_manage_gazebo_scene_server(self, object_name, object_model_name, dataset, model_type):
        self.test_count += 1
        print('Running test_manage_gazebo_scene_server, test number %d' % self.test_count)
        update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
        object_pose_array = self.get_pose_array_from_stamped(self.spawned_object_pose)

        res = update_gazebo_object(object_name, object_pose_array, object_model_name, model_type,
                                   dataset)
        self.spawned_object_mesh_path = self.object_datasets_folder + '/' + dataset + \
            '/models/' + object_model_name + '/google_16k/nontextured.stl'
        self.spawned_object_name = object_name
        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_save_visual_data_server(self, provide_server_with_save_data=False):
        self.test_count += 1
        print('Running test_save_visual_data_server, test number %d' % self.test_count)
        # Send to server and wait for response
        save_visual_data = rospy.ServiceProxy('save_visual_data', SaveVisualData)
        req = SaveVisualDataRequest()
        if provide_server_with_save_data:
            # Receive one message from depth, color and pointcloud topic, not registered
            msg_depth = rospy.wait_for_message("/camera/depth/image_raw", Image)
            msg_color = rospy.wait_for_message("/camera/color/image_raw", Image)
            msg_pcd = rospy.wait_for_message("/depth_registered/points", PointCloud2)
            print('Received depth, color and point cloud messages')
            req.color_img = msg_color
            req.depth_img = msg_depth
            req.point_cloud = msg_pcd
        req.color_img_save_path = self.color_img_save_path
        req.depth_img_save_path = self.depth_img_save_path
        req.point_cloud_save_path = self.scene_point_cloud_path
        res = save_visual_data(req)
        # Print result
        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_display_saved_point_cloud(self, pcd_save_path):
        self.test_count += 1
        print('Running test_display_saved_point_cloud, test number %d' % self.test_count)
        pcd = o3d.io.read_point_cloud(pcd_save_path)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        box_cam = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
        box_cam.paint_uniform_color([0, 1, 0])
        box_cam.translate([0.8275, -0.996, 0.361])  # create box at camera location
        o3d.visualization.draw_geometries([pcd, origin, box_cam])
        print('SUCCEEDED')

    def test_create_moveit_scene_server(self):
        self.test_count += 1
        print('Running test_create_moveit_scene_server, test number %d' % self.test_count)
        create_moveit_scene = rospy.ServiceProxy('create_moveit_scene', ManageMoveitScene)
        req = ManageMoveitSceneRequest()
        req.create_scene = True
        req.object_mesh_path = self.spawned_object_mesh_path
        req.object_pose_world = self.spawned_object_pose
        self.create_scene_response = create_moveit_scene(req)
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

    def test_control_hithand_config_server(self, go_home=False):
        self.test_count += 1
        print('Running test_control_hithand_config_server, test number %d' % self.test_count)
        control_hithand_config = rospy.ServiceProxy('control_hithand_config', ControlHithand)
        if not go_home:
            # Control the hithand to the desired joint state
            req = ControlHithandRequest()
            req.hithand_target_joint_state = self.chosen_hithand_joint_state
            res = control_hithand_config(req)
            # Check if the joint angles are within a small range of the desired angles
            hithand_current_joint_state = rospy.wait_for_message('/hithand/joint_states',
                                                                 JointState)
            reach_gap = np.array(self.chosen_hithand_joint_state.position) - \
                np.array(hithand_current_joint_state.position)
            rospy.loginfo('Gap between desired and actual joint position: ')
            rospy.loginfo('\n' + str(reach_gap))
            assert np.min(np.abs(reach_gap)) < 0.1
            rospy.loginfo('All gaps are smaller than 0.1')

        else:
            # Control the hithand back home
            req = ControlHithandRequest(go_home=True)
            res = control_hithand_config(req)
            # Check if the joint angles are within a small range of the desired angles
            hithand_current_joint_state = rospy.wait_for_message('/hithand/joint_states',
                                                                 JointState)
            reach_gap = np.array(hithand_current_joint_state.position)
            rospy.loginfo('Gap between desired and actual joint position: ')
            rospy.loginfo('\n' + str(reach_gap))
            assert np.min(np.abs(reach_gap)) < 0.1
            rospy.loginfo('All gaps are smaller than 0.1')

        result = 'SUCCEEDED' if res else 'FAILED'
        print(result)

    def test_segment_object_server(self):
        self.test_count += 1
        print('Running test_segment_object_server, test number %d' % self.test_count)
        table_object_segmentation = rospy.ServiceProxy('segment_object', SegmentGraspObject)
        req = SegmentGraspObjectRequest()
        req.scene_point_cloud_path = self.scene_point_cloud_path
        req.object_point_cloud_path = self.object_point_cloud_path
        res = table_object_segmentation(req)
        result = 'SUCCEEDED' if res.success else 'FAILED'

        assert os.path.exists(self.object_point_cloud_path)
        msg = rospy.wait_for_message('/segmented_object_bounding_box_corner_points',
                                     Float64MultiArray,
                                     timeout=5)
        assert msg.data is not None

        print(result)

    def test_generate_hithand_preshape_server(self):
        self.test_count += 1
        print('Running test_generate_hithand_preshape_server, test number %d' % self.test_count)
        generate_hithand_preshape = rospy.ServiceProxy('generate_hithand_preshape', GraspPreshape)
        req = GraspPreshapeRequest()
        req.sample = True
        self.heuristic_preshapes = generate_hithand_preshape(req)
        ## Check what else should be happening here, what should be published etc and try to visualize it
        msg = rospy.wait_for_message('/publish_box_points', MarkerArray, timeout=5)
        print('SUCCEEDED')

    def test_choose_specific_grasp_preshape(self, grasp_type):
        """ This chooses one specific grasp preshape from the preshapes in self.heuristic_preshapes.
        """
        self.test_count += 1
        print('Running test_choose_specific_grasp_preshape, test number %d' % self.test_count)

        if self.heuristic_preshapes == None:
            rospy.logerr(
                "generate_hithand_preshape_service needs to be called before calling this in order to generate the needed preshapes!"
            )
            raise Exception

        number_of_preshapes = len(self.heuristic_preshapes.hithand_joint_state)

        # determine the indices of the grasp_preshapes corresponding to top grasps
        top_grasp_idxs = []
        side_grasp_idxs = []
        for i in xrange(number_of_preshapes):
            if self.heuristic_preshapes.is_top_grasp[i]:
                top_grasp_idxs.append(i)
            else:
                side_grasp_idxs.append(i)

        if grasp_type == 'unspecified':
            grasp_idx = np.random.randint(0, number_of_preshapes)
        elif grasp_type == 'side':
            rand_int = np.random.randint(0, len(side_grasp_idxs))
            grasp_idx = side_grasp_idxs[rand_int]
        elif grasp_type == 'top':
            rand_int = np.random.randint(0, len(top_grasp_idxs))
            grasp_idx = side_grasp_idxs[rand_int]
        print('++++++++ CHOSEN INDEX: %d' % grasp_idx)
        self.chosen_palm_pose = self.heuristic_preshapes.palm_goal_pose_world[grasp_idx]
        self.chosen_hithand_joint_state = self.heuristic_preshapes.hithand_joint_state[grasp_idx]
        self.chosen_hithand_joint_state.position = [0] * 20
        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]
        print('SUCCEEDED')

    def test_arm_moveit_cartesian_pose_planner_server(self, go_home=False, place_goal_pose=None):
        self.test_count += 1
        for i in range(15):
            try:
                print('Running test_arm_moveit_cartesian_pose_planner_server, test number %d' %
                      self.test_count)

                arm_moveit_cartesian_pose_planner = rospy.ServiceProxy(
                    'arm_moveit_cartesian_pose_planner', PalmGoalPoseWorld)
                req = PalmGoalPoseWorldRequest()
                if go_home:
                    req.go_home = True
                elif place_goal_pose is not None:
                    req.palm_goal_pose_world = place_goal_pose
                else:
                    self.test_choose_specific_grasp_preshape('unspecified')
                    req.palm_goal_pose_world = self.chosen_palm_pose

                res = arm_moveit_cartesian_pose_planner(req)
                self.joint_trajectory = res.plan_traj
                if res.success:
                    print('[ARM MOVEIT] Achieved success after %d retries' % i)
                    break
            except rospy.ServiceException, e:
                rospy.loginfo('Service arm_moveit_cartesian_pose_planner call failed: %s' % e)

        result = 'SUCCEEDED' if res.success else 'FAILED'

        print(result)
        return res.success

    def test_execute_joint_trajectory_server(self, smoothen_trajectory=False):
        self.test_count += 1
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
        self.test_count += 1
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

    def test_grasp_control_hithand_server(self):
        self.test_count += 1
        print('Running test_grasp_control_hithand_server, test number %d' % self.test_count)
        grasp_control_hithand = rospy.ServiceProxy('/grasp_control_hithand', GraspControl)
        req = GraspControlRequest()
        res = grasp_control_hithand(req)
        result = 'SUCCEEDED' if res.success else 'FAILED'
        print(result)

    def test_reset_hithand_joints(self):
        self.test_count += 1
        print('Running test_reset_hithand_joints, test number %d' % self.test_count)
        reset_hithand_joints = rospy.ServiceProxy('/reset_hithand_joints', SetBool)
        req = SetBoolRequest()
        res = reset_hithand_joints(req)
        result = 'SUCCEEDED' if res.success else 'FAILED'
        print(result)


# ++++++++++++++++++++ Part III Integrated functionality calling multiple test ++++++++++++++++++++++++++++++++++++++

    def lift_object(self):
        # lift the object 15 cm
        print(self.chosen_palm_pose.pose.position)
        self.chosen_palm_pose.pose.position.z += 0.15
        self.test_arm_moveit_cartesian_pose_planner_server()
        self.test_execute_joint_trajectory_server(smoothen_trajectory=True)

    def reset_panda_and_hithand(self):
        self.test_reset_hithand_joints()
        panda_joints = rospy.wait_for_message('panda/joint_states', JointState)
        diff = np.abs(self.home_joint_states - np.array(panda_joints.position))
        if np.sum(diff) > 0.3:
            self.test_arm_moveit_cartesian_pose_planner_server(go_home=True)
            self.test_execute_joint_trajectory_server(smoothen_trajectory=True)

if __name__ == '__main__':
    # +++ Define variables for testing +++
    # ++++++++++++++++++++++++++++++++++++
    # Test spawn/delete Gazebo
    object_name = 'mustard_bottle'
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'
    # Test create_moveit_scene
    datasets_base_path = '/home/vm/object_datasets'
    object_mesh_path = datasets_base_path + '/' + dataset + \
        '/models/' + object_model_name + '/google_16k/nontextured.stl'
    # Test control_hithand_config
    hithand_joint_states = JointState()
    hithand_joint_states.position = [0.2] * 20
    # Test segment table and object
    object_pcd_path = "/home/vm/object.pcd"

    # Tester
    sut = ServerUnitTester()

    # Reset
    #sut.reset_panda_and_hithand()

    # Test random object pose
    #sut.test_generate_random_object_pose_for_experiment()

    # Test spawning and deleting of objects
    #sut.test_manage_gazebo_scene_server(object_name, object_model_name, dataset, model_type)

    # Test visual data save server
    #sut.test_save_visual_data_server()

    # Test display saved point cloud
    #sut.test_display_saved_point_cloud(sut.scene_point_cloud_path)

    # Test object segmentation
    sut.test_segment_object_server()

    # Test display saved point cloud
    #sut.test_display_saved_point_cloud(sut.object_point_cloud_path)

    # Test hithand preshape generation server
    #sut.test_generate_hithand_preshape_server()

    # Test moveit spawn object
    #sut.test_create_moveit_scene_server()

    # while True:
    #     # get new position and go there
    #     sut.test_choose_specific_grasp_preshape('unspecified')
    #     # Try to find a trajectory
    #     moveit_result = False
    #     while (not moveit_result):
    #         moveit_result = sut.test_arm_moveit_cartesian_pose_planner_server()

    #     sut.test_execute_joint_trajectory_server(smoothen_trajectory=True)

    #     input = raw_input("Grasp and lift? Press y/n: ")
    #     if input == 'y':
    #         # try to grasp the object
    #         sut.test_grasp_control_hithand_server()
    #         # try to lift the object
    #         sut.lift_object()

    #     # go home
    #     sut.reset_panda_and_hithand()

    #     # reset object position
    #     sut.test_manage_gazebo_scene_server(object_name, object_model_name, dataset, model_type)

    # Test smoothen trajectory execution
    # sut.test_get_smooth_trajectory_server()

    # # Make robot go home
    # sut.test_arm_moveit_cartesian_pose_planner_server(pose=None, go_home=True)

    # # Test execution of smoothen trajectory
    # sut.test_execute_joint_trajectory_server(smoothen_trajectory=True)

    # Test moveit delete object
    # sut.test_clean_moveit_scene_server()

    # Test hithand control preshape/config
    # sut.test_control_hithand_config_server(hithand_joint_states)