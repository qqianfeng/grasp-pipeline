#!/usr/bin/env python
from __future__ import division
import rospy
from robotiq_3f_srvs.srv import Move,GetPosition,SetMode
from robotiq_3f_gripper_articulated_msgs.msg import Robotiq3FGripperRobotInput
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
import numpy as np
from grasp_pipeline.utils.utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped, plot_voxel
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarkerPose
import tf
import tf.transformations as tft
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBoolRequest, SetBool
from grasp_pipeline.utils.align_object_frame import align_object
from grasp_pipeline.utils import utils
from std_msgs.msg import Header, Bool
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import copy
import datetime
from multiprocessing import Process
import cv2
import math
import open3d as o3d
import os
import time
import sys
from moveit_commander import PlanningSceneInterface
sys.path.append('..')
from grasp_pipeline.utils.open3d_draw_with_timeout import draw_with_time_out
from grasp_pipeline.msg import *

from uuid import uuid4
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.grasp_client.grasp_sim_client_robotiq import *

class TestGraspController():
    def __init__(self,hand='robotiq'):
        rospy.init_node('test_grasp_controller')
        self.hand = hand
        self.curr_pos = None
        self.panda_planned_joint_trajectory = None
        self.robot_cartesian_goal_pose_pandaj8 = None
        self.joint_trajectory_to_goal = None
        self.robot_cartesian_goal_pose = None
        self.setup_lift_and_grasp_poses()

    def setup_lift_and_grasp_poses(self):
        self.home_pose = InteractiveMarkerPose()
        self.home_pose.pose.position.x = 0.5
        self.home_pose.pose.position.y = 0.0
        self.home_pose.pose.position.z = 0.9

        self.home_pose.pose.orientation.x = 0.62
        self.home_pose.pose.orientation.y = 0.62
        self.home_pose.pose.orientation.z = 0.33
        self.home_pose.pose.orientation.w = 0.334848

        self.lift_pose = InteractiveMarkerPose()
        self.lift_pose.pose.position.x = 0.6476439232691327
        self.lift_pose.pose.position.y = -0.009837140154184118
        self.lift_pose.pose.position.z = 0.507262644093148

        self.lift_pose.pose.orientation.x = 0.8476802233161043
        self.lift_pose.pose.orientation.y = 0.5240429764425666
        self.lift_pose.pose.orientation.z = -0.03652414525949581
        self.lift_pose.pose.orientation.w = 0.07404852904034345

        self.grasp_pose = InteractiveMarkerPose()
        self.grasp_pose.pose.position.x = 0.6615187681665692
        self.grasp_pose.pose.position.y = -0.04421595156309393
        self.grasp_pose.pose.position.z = 0.3212456481334106
        self.grasp_pose.pose.orientation.x = 0.7901923091273372
        self.grasp_pose.pose.orientation.y = 0.607271084365329
        self.grasp_pose.pose.orientation.z = -0.04383754600659636
        self.grasp_pose.pose.orientation.w = 0.06997295370272645

    def transform_pose_to_palm_link(self, pose, from_link='panda_link8'):
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rospy.sleep(0.5)
        if self.hand == 'robotiq':
            transform = tfBuffer.lookup_transform(from_link, 'palm_link_robotiq', rospy.Time())
        else:
            transform = tfBuffer.lookup_transform(from_link, 'palm_link_hithand', rospy.Time())
        trans_tf_mat = tft.translation_matrix([
            transform.transform.translation.x, transform.transform.translation.y,
            transform.transform.translation.z
        ])
        rot_tf_mat = tft.quaternion_matrix([
            transform.transform.rotation.x, transform.transform.rotation.y,
            transform.transform.rotation.z, transform.transform.rotation.w
        ])
        palm_T_link8 = np.dot(trans_tf_mat, rot_tf_mat)

        trans_pose_mat = tft.translation_matrix([
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        ])
        rot_pose_mat = tft.quaternion_matrix([
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
            pose.pose.orientation.w
        ])
        link8_T_world = np.dot(trans_pose_mat, rot_pose_mat)

        palm_T_world = np.dot(link8_T_world, palm_T_link8)

        palm_pose_stamped = PoseStamped()
        palm_pose_stamped.header.frame_id = 'world'
        palm_pose_stamped.pose.position.x = palm_T_world[0, 3]
        palm_pose_stamped.pose.position.y = palm_T_world[1, 3]
        palm_pose_stamped.pose.position.z = palm_T_world[2, 3]

        quat = tft.quaternion_from_matrix(palm_T_world)

        palm_pose_stamped.pose.orientation.x = quat[0]
        palm_pose_stamped.pose.orientation.y = quat[1]
        palm_pose_stamped.pose.orientation.z = quat[2]
        palm_pose_stamped.pose.orientation.w = quat[3]

        return palm_pose_stamped

    def get_goal_pose_from_marker(self):
        while True:
            marker_pose = rospy.wait_for_message(
                '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update',
                InteractiveMarkerUpdate, 10)
            if len(marker_pose.poses) != 0:
                break
        palm_T_world = self.transform_pose_to_palm_link(marker_pose.poses[0])
        self.robot_cartesian_goal_pose = palm_T_world

    def plan_joint_trajectory_to_goal(self):
        # wait_for_service('plan_arm_trajectory')
        try:
            moveit_cartesian_pose_planner = rospy.ServiceProxy('plan_arm_trajectory',
                                                               PlanArmTrajectory)
            req = PlanArmTrajectoryRequest()
            
            req.palm_goal_pose_world = self.robot_cartesian_goal_pose
            
            res = moveit_cartesian_pose_planner(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service plan_arm_trajectory call failed: %s' % e)
        rospy.logdebug('Service plan_arm_trajectory is executed.')
        self.joint_trajectory_to_goal = res.trajectory
        return res.success

    def execute_joint_trajectory_to_goal(self):
        execute_joint_trajectory = rospy.ServiceProxy('execute_joint_trajectory',
                                                      ExecuteJointTrajectory)
        req = ExecuteJointTrajectoryRequest()
        req.smoothen_trajectory = True
        req.joint_trajectory = self.joint_trajectory_to_goal
        res = execute_joint_trajectory(req)
        return res.success

    def move_to_marker_position(self):
        self.get_goal_pose_from_marker()
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def move_to_lift_position(self):
        palm_T_world = self.transform_pose_to_palm_link(self.lift_pose)
        self.robot_cartesian_goal_pose = palm_T_world
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def move_to_grasp_position(self):
        palm_T_world = self.transform_pose_to_palm_link(self.grasp_pose)
        self.robot_cartesian_goal_pose = palm_T_world
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def move_to_home_position(self):
        palm_T_world = self.transform_pose_to_palm_link(self.home_pose)
        self.robot_cartesian_goal_pose = palm_T_world
        result = self.plan_joint_trajectory_to_goal()
        # try replanning 5 times
        if result == False:
            for i in range(5):
                result = self.plan_joint_trajectory_to_goal()
                if result:
                    break
        self.execute_joint_trajectory_to_goal()

    def spawn_object(self):
        update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
        pose = [0.7, 0, 0, 0, 0, 0]
        update_gazebo_object('mustard_bottle', pose, '/home/david/gazebo-objects/objects_gazebo/ycb/006_mustard_bottle/006_mustard_bottle.sdf', 'sdf')
    
    def update_object_metadata(self, object_metadata):
        """ Update the metainformation about the object to be grasped
        """
        self.object_metadata = object_metadata
    
    def update_gazebo_object_client(self):
        """Gazebo management client, spawns new object
        """
        wait_for_service('update_gazebo_object')
        # object_pose_array = get_pose_array_from_stamped(self.object_metadata["mesh_frame_pose"])
        object_pose_array = [0.7, 0, 0, 0, 0, 0]
        try:
            update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
            req = UpdateObjectGazeboRequest()
            req.object_name = self.object_metadata["name"]
            req.object_model_file = self.object_metadata["model_file"]
            req.object_pose_array = object_pose_array
            req.model_type = 'sdf'
            res = update_gazebo_object(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_gazebo_object call failed: %s' % e)
        rospy.logdebug('Service update_gazebo_object is executed %s.' % str(res.success))
        return res.success
    
    def grasp_object(self):
        if self.hand == 'robotiq':
            grasp_control_robotiq = rospy.ServiceProxy('/robotiq_3f_gripper/close_hand', Move)
            res = grasp_control_robotiq()
        else:
            grasp_control_hithand = rospy.ServiceProxy('/grasp_control_hithand', GraspControl)
            req = GraspControlRequest()
            res = grasp_control_hithand(req)

    def reset_hithand_joints_client(self):
        """ Server call to reset the hithand joints.
        """
        wait_for_service('reset_hithand_joints')
        try:
            reset_hithand = rospy.ServiceProxy('reset_hithand_joints', SetBool)
            res = reset_hithand(SetBoolRequest(data=True))
        except rospy.ServiceException, e:
            rospy.logerr('Service reset_hithand_joints call failed: %s' % e)
        rospy.logdebug('Service reset_hithand_joints is executed.')
        
    def robotiq_reset_hand(self):
        try:
            reset_robotiq_joints = rospy.ServiceProxy('/robotiq_3f_gripper/open_hand', Move)
            res = reset_robotiq_joints()
        except rospy.ServiceException, e:
            rospy.logerr('Service reset_robotiq_joints call failed: %s' % e)
        rospy.logdebug('Service reset_robotiq_joints is executed.')
    
    def delete_hand(self):
        wait_for_service('delete_gazebo_hand')
        try:
            delete_gazebo_hand = rospy.ServiceProxy('delete_gazebo_hand', DeleteHandGazebo)
            res = delete_gazebo_hand()
        except rospy.ServiceException, e:
            rospy.logerr('Service delete_gazebo_hand call failed: %s' % e)
        rospy.logdebug('Service delete_gazebo_hand is executed %s.' % str(res.success))
        return res.success

    def execute_joint_trajectory_client(self, smoothen_trajectory=True, speed='fast'):
        """ Service call to smoothen and execute a joint trajectory.
        """
        wait_for_service('execute_joint_trajectory')
        try:
            if self.panda_planned_joint_trajectory == None:
                rospy.logerr(
                    'No joint trajectory has been computed. Call plan_joint_trajectory server first'
                )
            elif len(self.panda_planned_joint_trajectory.points):
                execute_joint_trajectory = rospy.ServiceProxy('execute_joint_trajectory',
                                                              ExecuteJointTrajectory)
                req = ExecuteJointTrajectoryRequest()
                req.smoothen_trajectory = smoothen_trajectory
                req.joint_trajectory = self.panda_planned_joint_trajectory
                req.trajectory_speed = speed
                res = execute_joint_trajectory(req)
                return True
            else:
                return False
                rospy.logerr('The joint trajectory in planned_panda_joint_trajectory was empty.')
        except rospy.ServiceException, e:
            rospy.logerr('Service execute_joint_trajectory call failed: %s' % e)
        rospy.logdebug('Service execute_joint_trajectory is executed.')

    def robotiq_status_callback(self, msg):
        self.curr_pos = {'finger1':msg.gPOC,'finger2':msg.gPOB,'middle':msg.gPOA}
        rospy.loginfo('Got the positions: ' + str(self.curr_pos))
        # self.curr_vel = np.abs(np.array(msg.velocity))
        self.received_curr_pos = True

    def sub_robotiq_state(self):
        joint_states_sub = rospy.Subscriber(
            '/robotiq3f/state', Robotiq3FGripperRobotInput, self.robotiq_status_callback, tcp_nodelay=True
        )

    def get_robotiq_pos(self):
        return self.curr_pos

    def plan_reset_trajectory_client(self):
        wait_for_service('plan_reset_trajectory')
        try:
            plan_reset_trajectory = rospy.ServiceProxy('plan_reset_trajectory',
                                                       PlanResetTrajectory)
            res = plan_reset_trajectory(PlanResetTrajectoryRequest())
            self.panda_planned_joint_trajectory = res.trajectory
        except rospy.ServiceException, e:
            rospy.logerr('Service plan_reset_trajectory call failed: %s' % e)
        rospy.logdebug('Service plan_reset_trajectory is executed.')
        return res.success

    def reset_hand_and_panda(self):
        """ Reset panda and hand to their home positions
        """
        if self.hand == 'robotiq':
            self.robotiq_reset_hand()
        else:
            self.reset_hithand_joints_client()

        reset_plan_exists = self.plan_reset_trajectory_client()
        if reset_plan_exists:
            self.execute_joint_trajectory_client()
        self.delete_hand()

def main(hand):
    t = TestGraspController(hand)
    t.sub_robotiq_state()
    t.get_robotiq_pos()
    t.reset_hand_and_panda()
    # raw_input('Press enter to continue: ')
    rospy.sleep(1)
    t.get_robotiq_pos()
    t.spawn_object()
    # t.move_to_marker_position()
    t.move_to_grasp_position()
    t.get_robotiq_pos()
    t.grasp_object()
    t.get_robotiq_pos()
    t.move_to_lift_position()
    t.move_to_home_position()

def main_data_gen(hand):
    # poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
    #      [0.5, 0.0, 0.2, 0, 0, -1.571]]
    poses = [[0.5, 0.0, 0.2, 0, 0, 0]]

    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Intput a grasp mode for the robotiq
    robotiq_grasp_mode = raw_input('Choose a grasp mode for the robotiq: ')

    # Create grasp client and metadata handler
    grasp_data_rec = VisualGraspDataHandler(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
    grasp_sampler = GraspSampler()
    grasp_controller = GraspController(grasp_mode = robotiq_grasp_mode)
    object_spawner = ObjectSpawner(is_rec_sess=True)
    grasp_sampler.get_transform_pose_func(func=grasp_data_rec.transform_pose )
    grasp_controller.define_tf_buffer(grasp_data_rec.tf_buffer)
    rospy.sleep(1)

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object()

        # update object metadata for all classes that require it
        for class_inst in [grasp_data_rec, grasp_sampler, object_spawner, grasp_controller]:
            class_inst.update_object_metadata(object_metadata)
        
        # Loop over all 4 poses:
        for k, pose in enumerate(poses):
            print('trying pose '+str(k), pose )
            # start timer
            object_cycle_start = time.time()
            start = object_cycle_start

            # Create dirs
            grasp_data_rec.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

            # Reset panda and hithand
            grasp_controller.reset_robotiq_and_panda()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            object_spawner.spawn_object(pose_type="init", pose_arr=pose)

            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_data_rec.save_visual_data_and_segment_object()

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_sampler.get_valid_preshape_for_all_points(object_segment_response = grasp_data_rec.object_segment_response)
            j = 0
            
            while grasp_sampler.grasps_available:
                print('grasp loop '+str(j))
                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_data_rec.create_dirs_new_grasp_trial(is_new_pose_or_object=False)
                    grasp_sampler.get_object_segment_response(grasp_data_rec.object_segment_response)

                    # Reset panda and hithand
                    grasp_controller.reset_robotiq_and_panda()

                    # Spawn object in same position
                    object_spawner.spawn_object(pose_type="same")

                grasp_controller.get_is_grasps_available(grasp_sampler.grasps_available)
                grasp_controller.get_heuristic_preshapes(grasp_sampler.heuristic_preshapes)
                grasp_controller.get_num_preshapes(grasp_sampler.num_preshapes)
                grasp_controller.get_save_only_depth_and_color_func(grasp_data_rec.save_only_depth_and_color)
                grasp_controller.get_pos_idxs(
                    grasp_sampler.top_idxs,
                    grasp_sampler.side1_idxs,
                    grasp_sampler.side2_idxs)

                # Grasp and lift object
                grasp_arm_plan = grasp_controller.grasp_and_lift_object()

                # Save all grasp data including post grasp images
                grasp_data_rec.get_palm_poses(grasp_controller.palm_poses)
                grasp_data_rec.get_post_grasp_data(
                    grasp_controller.chosen_is_top_grasp,
                    grasp_controller.grasp_label,
                    grasp_controller.object_metadata,
                    grasp_controller.hand_joint_states
                )
                grasp_data_rec.save_visual_data_and_record_grasp()

                # measure time
                print("One cycle took: " + str(time.time() - start))
                j += 1
                # only for test:
                if j >= 1 :
                    grasp_sampler.grasps_available = False
            # Finally write the time to file it took to test all poses
            grasp_data_rec.log_object_cycle_time(time.time() - object_cycle_start)

if __name__ == '__main__':
    # main('robotiq')
    main_data_gen('robotiq')