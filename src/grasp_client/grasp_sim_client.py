#!/usr/bin/env python
import rospy
import datetime
import time
from geometry_msgs.msg import PoseStamped
import tf
import tf.transformations as tft
import tf2_ros
import tf2_geometry_msgs
from grasp_pipeline.srv import *
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import sys
sys.path.append('..')
from utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped, plot_voxel
from align_object_frame import align_object
import numpy as np
from std_srvs.srv import SetBool, SetBoolRequest
import os
from multiprocessing import Process
import copy


class GraspClient():
    """ This class is a wrapper around all the individual functionality involved in grasping experiments.
    """
    def __init__(self, grasp_data_recording_path):
        rospy.init_node('grasp_client')
        self.object_datasets_folder = rospy.get_param('object_datasets_folder')
        self.grasp_data_recording_path = grasp_data_recording_path
        self.create_grasp_folder_structure(self.grasp_data_recording_path)
        # Save metainformation on object to be grasped in these vars
        self.object_metadata = dict()  # This dict holds info about object name, pose, meshpath
        self.palm_poses = dict()  # Saves the palm poses at different stages of the experiment
        self.hand_joint_states = dict()  # Saves the joint states at different stages
        self._setup_workspace_boundaries()

        self.tf_listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tf_buffer)

        self.depth_img = None
        self.color_img = None
        self.pcd = None

        # These variables get changed dynamically during execution to store relevant data under correct folder
        self.color_img_save_path = None
        self.depth_img_save_path = None
        self.scene_pcd_save_path = '/home/vm/scene.pcd'
        self.object_pcd_save_path = '/home/vm/object.pcd'
        self.object_pcd_record_path = ''

        self.heuristic_preshapes = None  # This variable stores all the information on multiple heuristically sampled grasping pre shapes
        # The chosen variables store one specific preshape (palm_pose, hithand_joint_state, is_top_grasp)
        self.chosen_is_top_grasp = None

        self.panda_planned_joint_trajectory = None
        self.num_of_replanning_attempts = 2
        self.num_poses = 50
        self.plan_without_approach_pose = False

        self.grasps_available = True
        self.grasp_types = ["side1", "side2", "top"]
        self.previous_grasp_type = None
        self.chosen_grasp_type = 'unspecified'

        self.object_lift_height = 0.2  # Lift the object 20 cm
        self.success_tolerance_lift_height = 0.05
        self.object_segment_response = None
        self.grasp_label = None

        self.trigger_cond = False
        self.hithand_reset_position = [
            0, 0.0872665, 0.0872665, 0.0872665, 0, 0.0872665, 0.0872665, 0.0872665, 0, 0.0872665,
            0.0872665, 0.0872665, 0, 0.0872665, 0.0872665, 0.0872665, -0.26, 0.0872665, 0.0872665,
            0.0872665
        ]

        # For voxel server
        self.voxel_grid_dim = np.array([26, 26, 26])
        self.voxel_grid_dim_full = np.array([32, 32, 32])
        self.voxel_translation_dim = (self.voxel_grid_dim_full - self.voxel_grid_dim) // 2

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
    def parallel_execute_functions(self, functions):
        """ Run functions in parallel
        """
        processes = [Process(target=function) for function in functions]
        for proc in processes:
            proc.start()
            print("Process started")
        for proc in processes:
            proc.join()

    def create_grasp_folder_structure(self, base_path):
        rec_sess_path = base_path + 'grasp_data/recording_sessions'
        if os.path.exists(rec_sess_path):
            self.sess_id_num = int(sorted(os.listdir(rec_sess_path))[-1].split('_')[-1]) + 1
            if os.listdir(rec_sess_path + '/recording_session_' +
                          str(self.sess_id_num - 1).zfill(4)):
                self.grasp_id_num = int(
                    sorted(
                        os.listdir(rec_sess_path + '/recording_session_' +
                                   str(self.sess_id_num - 1).zfill(4)))[-1].split('_')[-1])
            else:
                self.grasp_id_num = 0
        else:  # if the path did not exist yet, this is the first recording
            self.sess_id_num = 1
            self.grasp_id_num = 0
            os.makedirs(rec_sess_path)
            rospy.loginfo('This is the first recording, no prior recordings found.')

        self.sess_id_str = str(self.sess_id_num).zfill(4)
        self.grasp_id_str = str(self.grasp_id_num).zfill(6)
        rospy.loginfo('Session id: ' + self.sess_id_str)
        rospy.loginfo('Grasp id: ' + self.grasp_id_str)
        self.curr_rec_sess_path = rec_sess_path + '/recording_session_' + self.sess_id_str
        os.mkdir(self.curr_rec_sess_path)

    def update_object_metadata(self, object_metadata):
        """ Update the metainformation about the object to be grasped
        """
        self.object_metadata = object_metadata

    def create_dirs_new_grasp_trial(self):
        """ This should be called anytime before a new grasp trial is attempted as it will create the necessary folder structure.
        """
        self.grasp_id_num += 1
        self.grasp_id_str = str(self.grasp_id_num).zfill(6)

        rospy.loginfo('Grasp id: ' + self.grasp_id_str)

        self.curr_grasp_trial_path = self.curr_rec_sess_path + '/grasp_' + self.grasp_id_str
        if os.path.exists(self.curr_grasp_trial_path):
            rospy.logerr("Path for grasp trial already exists, something is wrong.")
        os.mkdir(self.curr_grasp_trial_path)
        os.mkdir(self.curr_grasp_trial_path + '/during_grasp')
        os.mkdir(self.curr_grasp_trial_path + '/post_grasp')
        os.mkdir(self.curr_grasp_trial_path + '/pre_grasp')

    def _setup_workspace_boundaries(self):
        """ Sets the boundaries in which an object can be spawned and placed.
        Gets called 
        """
        self.spawn_object_x_min, self.spawn_object_x_max = 0.25, 0.65
        self.spawn_object_y_min, self.spawn_object_y_max = -0.2, 0.2
        self.table_height = 0

    def generate_random_object_pose_for_experiment(self):
        """Generates a random x,y position and z orientation within object_spawn boundaries for grasping experiments.
        """
        rand_x = np.random.uniform(self.spawn_object_x_min, self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min, self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [
            self.object_metadata["spawn_angle_roll"], 0, rand_z_orientation, rand_x, rand_y,
            self.object_metadata["spawn_height_z"]
        ]
        rospy.loginfo('Generated random object pose:')
        rospy.loginfo(object_pose)
        object_pose_stamped = get_pose_stamped_from_array(object_pose)
        self.object_metadata["sim_pose"] = object_pose_stamped

    def choose_specific_grasp_preshape(self, grasp_type):
        """ This chooses one specific grasp preshape from the preshapes in self.heuristic_preshapes.
        Grasp type can be side1, side2, or top.
        """
        if self.heuristic_preshapes == None:
            rospy.logerr(
                "generate_hithand_preshape_service needs to be called before calling this in order to generate the needed preshapes!"
            )
            raise Exception
        # If this is not the first execution:
        if not (len(self.top_idxs) + len(self.side1_idxs) +
                len(self.side2_idxs)) == self.num_preshapes:
            self.previous_grasp_type = self.chosen_grasp_type

        # If unspecified randomly select a grasp type:
        if grasp_type == 'unspecified':
            grasp_type = self.grasp_types[np.random.randint(0, len(self.grasp_types))]

        available_grasp_types = copy.deepcopy(self.grasp_types)

        # Check if there still exist grasps of this type
        if not len(self.side1_idxs):
            available_grasp_types.remove('side1')
        if not len(self.side2_idxs):
            available_grasp_types.remove('side2')
        if not len(self.top_idxs):
            available_grasp_types.remove('top')

        # Check if the desired grasp type is available
        if not len(available_grasp_types):
            print('No grasp type is available anymore')
            self.grasps_available = False
            return False
        elif grasp_type in available_grasp_types:
            print('Grasp type is available. Chosen grasp type: ' + str(grasp_type))
        else:
            grasp_type = available_grasp_types[np.random.randint(0, len(available_grasp_types))]

        if grasp_type == 'side1':
            grasp_idx = self.side1_idxs[np.random.randint(0, len(self.side1_idxs))]
        elif grasp_type == 'side2':
            grasp_idx = self.side2_idxs[np.random.randint(0, len(self.side2_idxs))]
        elif grasp_type == 'top':
            grasp_idx = self.top_idxs[np.random.randint(0, len(self.top_idxs))]

        # Store the chosen grasp pose and appraoch pose as well as joint state in corresponding dicts
        self.palm_poses["desired_pre"] = self.heuristic_preshapes.palm_goal_pose_world[grasp_idx]
        self.palm_poses["approach"] = self.heuristic_preshapes.palm_approach_pose_world[grasp_idx]
        self.hand_joint_states["desired_pre"] = self.heuristic_preshapes.hithand_joint_state[
            grasp_idx]

        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]
        self.chosen_grasp_idx = grasp_idx

        self.chosen_grasp_type = grasp_type
        print("Chosen grasp type is: " + str(self.chosen_grasp_type))
        # If this IS the first execution initialize previous grasp_type with current:
        if (len(self.top_idxs) + len(self.side1_idxs) +
                len(self.side2_idxs)) == self.num_preshapes:
            self.previous_grasp_type = self.chosen_grasp_type

    def verify_hithand_needs_reset(self):
        curr_pos = rospy.wait_for_message('/hithand/joint_states', JointState).position
        pos_diff = np.abs(np.array(self.hithand_reset_position) - curr_pos)
        # If at least one joint is more than 1e-3 away from where it's supposed to be, say hithand needs reset
        if pos_diff[pos_diff > 8e-3].size == 0:
            return False
        else:
            return True

    # ++++++++ PART II: Second part consist of all clients that interact with different nodes/services ++++++++++++
    def control_hithand_config_client(self, go_home=False, close_hand=False):
        wait_for_service('control_hithand_config')
        try:
            req = ControlHithandRequest()
            control_hithand_config = rospy.ServiceProxy('control_hithand_config', ControlHithand)
            if go_home:
                req.go_home = True
            elif close_hand:
                req.close_hand = True
            else:
                req.hithand_target_joint_state = self.hand_joint_states["desired_pre"]
                res = control_hithand_config(req)
            # Buht how is the logic here for data gathering? Execute all of the samples and record responses right?
        except rospy.ServiceException as e:
            rospy.loginfo('Service control_hithand_config call failed: %s' % e)
        rospy.loginfo('Service control_allegro_config is executed %s.' % str(res))

    def execute_joint_trajectory_client(self, smoothen_trajectory=True):
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
                res = execute_joint_trajectory(req)
                return True
            else:
                return False
                rospy.loginfo('The joint trajectory in planned_panda_joint_trajectory was empty.')
        except rospy.ServiceException, e:
            rospy.loginfo('Service execute_joint_trajectory call failed: %s' % e)
        rospy.loginfo('Service execute_joint_trajectory is executed.')

    def generate_hithand_preshape_client(self):
        """ Generates 
        """
        wait_for_service('generate_hithand_preshape')
        try:
            generate_hithand_preshape = rospy.ServiceProxy('generate_hithand_preshape',
                                                           GraspPreshape)
            req = GraspPreshapeRequest()
            # The req.object will usually be aligned, meaning the coordinate axis have been flipped as to be consistent with the world frame and correspondingly width, height, depth have been changed
            if self.object_segment_response is not None:
                req.object = self.object_segment_response.object
            else:
                raise Exception(
                    "self.object_segment_response.object attribute is None. Should hold information about grasp object. Call segemtation server first."
                )
            req.sample = True
            self.heuristic_preshapes = generate_hithand_preshape(req)
            self.num_preshapes = len(self.heuristic_preshapes.palm_goal_pose_world)
            self.side1_idxs = range(0, self.num_preshapes / 3)
            if self.heuristic_preshapes.is_top_grasp[self.num_preshapes / 3] == True:
                self.top_idxs = range(self.num_preshapes / 3, 2 * self.num_preshapes / 3)
                self.side2_idxs = range(2 * self.num_preshapes / 3, self.num_preshapes)
            else:
                self.side2_idxs = range(self.num_preshapes / 3, 2 * self.num_preshapes / 3)
                self.top_idxs = range(2 * self.num_preshapes / 3, self.num_preshapes)

            self.grasps_available = True

        except rospy.ServiceException, e:
            rospy.loginfo('Service generate_hithand_preshape call failed: %s' % e)
        rospy.loginfo('Service generate_hithand_preshape is executed.')

    def generate_voxel_from_pcd_client(self, show_voxel=False):
        """ Generates centered sparse voxel grid from segmented object pcd.
        """
        wait_for_service("generate_voxel_from_pcd")
        try:
            generate_voxel_from_pcd = rospy.ServiceProxy("generate_voxel_from_pcd",
                                                         GenVoxelFromPcd)

            # First compute the voxel size from the object dimensions
            object_max_dim = np.max([self.object_metadata["aligned_dim_whd"]])
            voxel_size = object_max_dim / self.voxel_grid_dim

            # Generate request
            req = GenVoxelFromPcdRequest()
            req.object_pcd_path = self.object_pcd_save_path
            req.voxel_dim = self.voxel_grid_dim
            req.voxel_translation_dim = self.voxel_translation_dim
            req.voxel_size = voxel_size

            # Get result
            res = generate_voxel_from_pcd(req)
            self.object_metadata["sparse_voxel_grid"] = res.voxel_grid

            # Show the voxel grid
            if show_voxel:
                voxel_grid = np.reshape(res.voxel_grid, [len(res.voxel_grid) / 3, 3])
                plot_voxel(voxel_grid, voxel_res=self.voxel_grid_dim_full)
        except rospy.ServiceException, e:
            rospy.loginfo('Service generate_voxel_from_pcd call failed: %s' % e)
        rospy.loginfo('Service generate_voxel_from_pcd is executed.')

    def get_hand_palm_pose_and_joint_state(self):
        """ Returns pose stamped and joint state
            1. palm pose
            2. hithand joint state
        """
        (palm_trans, palm_rot) = self.tf_listener.lookupTransform('world', 'palm_link_hithand',
                                                                  rospy.Time())
        palm_pose = PoseStamped()
        palm_pose.header.frame_id = 'world'
        palm_pose.pose.position.x = palm_trans[0]
        palm_pose.pose.position.y = palm_trans[1]
        palm_pose.pose.position.z = palm_trans[2]
        palm_pose.pose.orientation.x = palm_rot[0]
        palm_pose.pose.orientation.y = palm_rot[1]
        palm_pose.pose.orientation.z = palm_rot[2]
        palm_pose.pose.orientation.w = palm_rot[3]
        joint_state = rospy.wait_for_message("/hithand/joint_states", JointState, timeout=5)
        return palm_pose, joint_state

    def get_grasp_object_pose_client(self):
        """ Get the current pose (not stamped) of the grasp object from Gazebo.
        """
        wait_for_service('gazebo/get_model_state')
        try:
            get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
            req = GetModelStateRequest()
            req.model_name = self.object_metadata["name"]
            res = get_model_state(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service grasp_control_hithand call failed: %s' % e)
        rospy.loginfo('Service grasp_control_hithand is executed.')
        return res.pose

    def grasp_control_hithand_client(self):
        """ Call server to close hithand fingers and stop when joint velocities are close to zero.
        """
        wait_for_service('grasp_control_hithand')
        try:
            grasp_control_hithand = rospy.ServiceProxy('grasp_control_hithand', GraspControl)
            req = GraspControlRequest()
            res = grasp_control_hithand(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service grasp_control_hithand call failed: %s' % e)
        rospy.loginfo('Service grasp_control_hithand is executed.')

    def plan_arm_trajectory_client(
        self,
        place_goal_pose=None,
    ):
        wait_for_service('plan_arm_trajectory')
        try:
            moveit_cartesian_pose_planner = rospy.ServiceProxy('plan_arm_trajectory',
                                                               PlanArmTrajectory)
            req = PlanArmTrajectoryRequest()
            if place_goal_pose is not None:
                req.palm_goal_pose_world = place_goal_pose
            else:
                req.palm_goal_pose_world = self.palm_poses["desired_pre"]
            res = moveit_cartesian_pose_planner(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service plan_arm_trajectory call failed: %s' % e)
        rospy.loginfo('Service plan_arm_trajectory is executed %s.' % str(res.success))
        self.panda_planned_joint_trajectory = res.trajectory
        return res.success

    def plan_cartesian_path_trajectory_client(self):
        wait_for_service('plan_cartesian_path_trajectory')
        try:
            plan_cartesian_path_trajectory = rospy.ServiceProxy('plan_cartesian_path_trajectory',
                                                                PlanCartesianPathTrajectory)
            req = PlanCartesianPathTrajectoryRequest()
            # Change the reference frame of desired_pre and approach pose to be
            req.palm_goal_pose_world = self.palm_poses["desired_pre"]
            req.palm_approach_pose_world = self.palm_poses["approach"]
            res = plan_cartesian_path_trajectory(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service plan_arm_trajectory call failed: %s' % e)
        rospy.loginfo('Service plan_arm_trajectory is executed %s.' % str(res.success))
        self.panda_planned_joint_trajectory = res.trajectory
        return res.success

    def plan_reset_trajectory_client(self):
        wait_for_service('plan_reset_trajectory')
        try:
            plan_reset_trajectory = rospy.ServiceProxy('plan_reset_trajectory',
                                                       PlanResetTrajectory)
            res = plan_reset_trajectory(PlanResetTrajectoryRequest())
            self.panda_planned_joint_trajectory = res.trajectory
        except rospy.ServiceException, e:
            rospy.loginfo('Service plan_reset_trajectory call failed: %s' % e)
        rospy.loginfo('Service plan_reset_trajectory is executed.')

    def record_grasp_data_client(self):
        wait_for_service('record_grasp_data')
        try:
            # Currently all poses are in the world frame. Transfer the desired pose into object frame

            record_grasp_data = rospy.ServiceProxy('record_grasp_data', RecordGraspDataSim)
            req = RecordGraspDataSimRequest()
            req.object_name = self.object_metadata["name"]
            req.time_stamp = datetime.datetime.now().isoformat()
            req.is_top_grasp = self.chosen_is_top_grasp
            req.grasp_success_label = self.grasp_label
            req.object_size_aligned = self.object_metadata["aligned_dim_whd"]
            req.object_size_unaligned = self.object_metadata["seg_dim_whd"]
            req.sparse_voxel_grid = self.object_metadata["sparse_voxel_grid"]
            req.object_world_sim_pose = self.object_metadata["sim_pose"]
            req.object_world_seg_unaligned_pose = self.object_metadata["seg_pose"]
            req.object_world_aligned_pose = self.object_metadata["aligned_pose"]
            req.desired_preshape_palm_world_pose = self.palm_poses["desired_pre"]
            req.palm_in_object_aligned_frame_pose = self.palm_poses["palm_in_object_aligned_frame"]
            req.true_preshape_palm_world_pose = self.palm_poses["true_pre"]
            req.closed_palm_world_pose = self.palm_poses["closed"]
            req.lifted_palm_world_pose = self.palm_poses["lifted"]
            req.desired_preshape_hithand_joint_state = self.hand_joint_states["desired_pre"]
            req.true_preshape_hithand_joint_state = self.hand_joint_states["true_pre"]
            req.closed_hithand_joint_state = self.hand_joint_states["closed"]
            req.lifted_hithand_joint_state = self.hand_joint_states["lifted"]

            res = record_grasp_data(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service record_grasp_data call failed: %s' % e)
        rospy.loginfo('Service record_grasp_data is executed.')

    def reset_hithand_from_topic(self):
        pub = rospy.Publisher("/start_hithand_reset", Bool, latch=True, queue_size=1)
        self.trigger_cond = not self.trigger_cond
        pub.publish(Bool(data=self.trigger_cond))

    def reset_hithand_joints_client(self):
        """ Server call to reset the hithand joints.
        """
        wait_for_service('reset_hithand_joints')
        try:
            reset_hithand = rospy.ServiceProxy('reset_hithand_joints', SetBool)
            res = reset_hithand(SetBoolRequest(data=True))
        except rospy.ServiceException, e:
            rospy.loginfo('Service reset_hithand_joints call failed: %s' % e)
        rospy.loginfo('Service reset_hithand_joints is executed.')

    def save_visual_data_client(self):
        wait_for_service('save_visual_data')
        try:
            save_visual_data = rospy.ServiceProxy('save_visual_data', SaveVisualData)
            req = SaveVisualDataRequest()
            req.color_img_save_path = self.color_img_save_path
            req.depth_img_save_path = self.depth_img_save_path
            req.scene_pcd_save_path = self.scene_pcd_save_path
            res = save_visual_data(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service save_visual_data call failed: %s' % e)
        rospy.loginfo('Service save_visual_data is executed %s' % res.success)

    def segment_object_client(self, align_object_world=True):
        wait_for_service('segment_object')
        try:
            segment_object = rospy.ServiceProxy('segment_object', SegmentGraspObject)
            req = SegmentGraspObjectRequest()
            req.scene_pcd_path = self.scene_pcd_save_path
            req.object_pcd_path = self.object_pcd_save_path
            req.object_pcd_record_path = self.object_pcd_record_path
            self.object_segment_response = segment_object(req)
            self.object_metadata["seg_pose"] = PoseStamped(
                header=Header(frame_id='world'), pose=self.object_segment_response.object.pose)
            self.object_metadata["seg_dim_whd"] = [
                self.object_segment_response.object.width,
                self.object_segment_response.object.height,
                self.object_segment_response.object.depth
            ]
            if align_object_world:
                # TODO: Move this to a function, possibly transform dim whd within alignment function
                self.object_segment_response.object = align_object(
                    self.object_segment_response.object, self.tf_listener)
                self.object_metadata["aligned_pose"] = PoseStamped(
                    header=Header(frame_id='world'), pose=self.object_segment_response.object.pose)

                print("whd before alignment: ")
                print(self.object_metadata["seg_dim_whd"])

                self.update_object_pose_aligned_client()
                print("whd from alignment: ")
                print([
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ])

                rospy.sleep(0.2)
                bb_extent = np.ones(4)
                bb_extent[:3] = np.array(self.object_metadata["seg_dim_whd"])
                (_, quat) = self.tf_listener.lookupTransform('object_pose_aligned', 'object_pose',
                                                             rospy.Time())
                aligned_T_pose = tft.quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])
                bb_extent_aligned = np.abs(aligned_T_pose.dot(bb_extent))
                self.object_metadata["aligned_dim_whd"] = bb_extent_aligned
                self.object_segment_response.object.width = bb_extent_aligned[0]
                self.object_segment_response.object.height = bb_extent_aligned[1]
                self.object_segment_response.object.depth = bb_extent_aligned[2]
                print("whd Vincent: ")
                print([
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ])

        except rospy.ServiceException, e:
            rospy.loginfo('Service segment_object call failed: %s' % e)
        rospy.loginfo('Service segment_object is executed.')

    def update_moveit_scene_client(self):
        wait_for_service('update_moveit_scene')
        try:
            update_moveit_scene = rospy.ServiceProxy('update_moveit_scene', ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            req = ManageMoveitSceneRequest()
            req.object_mesh_path = self.object_metadata["collision_mesh_path"]
            req.object_pose_world = self.object_metadata["sim_pose"]
            self.update_scene_response = update_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_moveit_scene call failed: %s' % e)
        rospy.loginfo(
            'Service update_moveit_scene is executed %s.' % str(self.update_scene_response))

    def update_gazebo_object_client(self):
        """Gazebo management client, deletes previous object and spawns new object
        """
        wait_for_service('update_gazebo_object')
        object_pose_array = get_pose_array_from_stamped(self.object_metadata["sim_pose"])
        try:
            update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
            req = UpdateObjectGazeboRequest()
            req.object_name = self.object_metadata["name"]
            req.object_model_file = self.object_metadata["model_file"]
            req.object_pose_array = object_pose_array
            req.model_type = 'sdf'
            res = update_gazebo_object(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' % str(res.success))
        return res.success

    def update_object_pose_aligned_client(self):
        wait_for_service("update_grasp_object_pose")
        try:
            update_grasp_object_pose = rospy.ServiceProxy('update_grasp_object_pose',
                                                          UpdateObjectPose)
            req = UpdateObjectPoseRequest()
            req.object_pose_world = self.object_metadata["aligned_pose"]
            res = update_grasp_object_pose(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_grasp_object_pose is executed %s.' % str(res.success))

    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def label_grasp(self):
        object_pose = self.get_grasp_object_pose_client()
        object_pos_delta_z = np.abs(object_pose.position.z -
                                    self.object_metadata["sim_pose"].pose.position.z)
        if object_pos_delta_z > (self.object_lift_height - self.success_tolerance_lift_height):
            self.grasp_label = 1
        else:
            self.grasp_label = 0

        rospy.loginfo("The grasp label is: " + str(self.grasp_label))

    def remove_unreachable_pose(self):
        if self.chosen_grasp_type == 'side1':
            self.side1_idxs.remove(self.chosen_grasp_idx)
        elif self.chosen_grasp_type == 'side2':
            self.side2_idxs.remove(self.chosen_grasp_idx)
        elif self.chosen_grasp_type == 'top':
            self.top_idxs.remove(self.chosen_grasp_idx)
        else:
            raise ValueError

    def reset_hithand_and_panda(self):
        """ Reset panda and hithand to their home positions
        """
        self.plan_reset_trajectory_client()
        self.reset_hithand_from_topic()
        self.execute_joint_trajectory_client()
        # Introduce a backup if reset from topic is failing
        if self.verify_hithand_needs_reset():
            self.reset_hithand_joints_client()
            self.trigger_cond = not self.trigger_cond

    def spawn_object(self, generate_random_pose):
        # Generate a random valid object pose
        if generate_random_pose:
            self.generate_random_object_pose_for_experiment()
        elif self.object_metadata["sim_pose"] == None:
            rospy.logerr(
                "Object pose has not been initialized yet and generate_random_pose is false. Either set true or update object pose."
            )
        # Update gazebo object, delete old object and spawn new one
        self.update_gazebo_object_client()
        # Now wait for 2 seconds for object to rest and update actual object position
        if self.object_metadata["dataset"] == 'kit':
            wait_time = 2
        else:
            wait_time = 0.2
        rospy.sleep(wait_time)
        object_pose = self.get_grasp_object_pose_client()
        # Update the sim_pose with the actual pose of the object after it came to rest
        self.object_metadata["sim_pose"] = PoseStamped(header=Header(frame_id='world'),
                                                       pose=object_pose)
        # Update moveit scene object
        self.update_moveit_scene_client()

    def set_visual_data_save_paths(self, grasp_phase):
        if grasp_phase == 'pre':
            folder_name = '/pre_grasp/'
            self.object_pcd_record_path = self.curr_grasp_trial_path + folder_name + self.object_metadata[
                "name"] + '.pcd'
        elif grasp_phase == 'during':
            folder_name = '/during_grasp/'
        elif grasp_phase == 'post':
            folder_name = '/post_grasp/'
        else:
            rospy.logerr('Given grasp_phase is not valid. Must be pre, during or post.')

        self.depth_img_save_path = self.curr_grasp_trial_path + folder_name + self.object_metadata[
            "name"] + '_depth.png'
        self.color_img_save_path = self.curr_grasp_trial_path + folder_name + self.object_metadata[
            "name"] + '_color.jpg'

    def save_visual_data_and_record_grasp(self):
        self.set_visual_data_save_paths(grasp_phase='post')
        self.save_visual_data_client()
        self.generate_voxel_from_pcd_client()
        self.record_grasp_data_client()

    def save_only_depth_and_color(self, grasp_phase):
        """ Saves only depth and color by setting scene_pcd_save_path to None. Resets scene_pcd_save_path afterwards.
        """
        self.set_visual_data_save_paths(grasp_phase=grasp_phase)
        pcd_save_path_temp = self.scene_pcd_save_path
        self.scene_pcd_save_path = ''
        self.save_visual_data_client()
        self.scene_pcd_save_path = pcd_save_path_temp

    def save_visual_data_and_segment_object(self):
        self.set_visual_data_save_paths(grasp_phase='pre')
        self.save_visual_data_client()
        self.segment_object_client()

    def store_poses_and_joint_states_in_instance_variable(self, grasp_phase):
        raise NotImplementedError

    def generate_hithand_preshape(self):
        """ Generate multiple grasp preshapes, which get stored in instance variable.
        """
        self.generate_hithand_preshape_client()

    def grasp_and_lift_object(self):
        # Control the hithand to it's preshape
        #self.control_hithand_config_client()
        i = 0
        # As long as there are viable poses
        while self.grasps_available:
            # Step 1 choose a specific grasp. In first iteration self.chosen_grasp_type is unspecific, e.g. function will randomly choose grasp type
            self.choose_specific_grasp_preshape(grasp_type=self.chosen_grasp_type)
            if not self.grasps_available:
                break
            # Step 2, if the previous grasp type is not same as current grasp type move to approach pose
            if self.previous_grasp_type != self.chosen_grasp_type or i == 0:
                approach_plan_exists = self.plan_arm_trajectory_client(
                    place_goal_pose=self.palm_poses["approach"])
                i = 0
                while not approach_plan_exists and i < 3:
                    i += 1
                    # add random 3D noise to approach pose
                    self.palm_poses["approach"].pose.position.x = np.random.uniform(-0.04, 0.04)
                    self.palm_poses["approach"].pose.position.y = np.random.uniform(-0.04, 0.04)
                    self.palm_poses["approach"].pose.position.z = np.random.uniform(-0.04, 0.04)
                    # try slightly shifted pose:
                    approach_plan_exists = self.plan_arm_trajectory_client(
                        place_goal_pose=self.palm_poses["approach"])
                    if approach_plan_exists:
                        print('Sampling approach pose helped.')
                # If a plan could be found, execute
                if approach_plan_exists:
                    self.execute_joint_trajectory_client()

            # Step 3, try to move to the desired palm position
            for i in range(2):
                desired_plan_exists = self.plan_arm_trajectory_client()
                if desired_plan_exists:
                    break
                else:
                    self.palm_poses["desired_pre"].pose.position.x += np.random.uniform(
                        -0.02, 0.02)
                    self.palm_poses["desired_pre"].pose.position.y += np.random.uniform(
                        -0.02, 0.02)
                    self.palm_poses["desired_pre"].pose.position.z += np.random.uniform(
                        -0.02, 0.02)
            i += 1
            # Step 4 if a plan exists execute it, otherwise delete unsuccessful pose and start from top:
            if desired_plan_exists:
                self.execute_joint_trajectory_client()
                break
            else:
                self.remove_unreachable_pose()

        # If the function did not already return, it means a valid plan has been found and will be executed
        # The pose to which the found plan leads is the pose which gets evaluated with respect to grasp success. Transform this pose to object_centric_frame
        assert self.palm_poses["desired_pre"].header.frame_id == 'world'
        tf2_pose = tf2_geometry_msgs.PoseStamped()
        tf2_pose.pose = self.palm_poses["desired_pre"].pose
        tf2_pose.header.frame_id = 'world'
        tf2_pose.header.stamp = rospy.Time.now()
        self.palm_poses["palm_in_object_aligned_frame"] = self.tf_buffer.transform(
            tf2_pose, 'object_pose_aligned', rospy.Duration(1))
        assert self.palm_poses[
            "palm_in_object_aligned_frame"].header.frame_id == 'object_pose_aligned'

        # Get the current actual joint position and palm pose
        self.palm_poses["true_pre"], self.hand_joint_states[
            "true_pre"] = self.get_hand_palm_pose_and_joint_state()

        letter = raw_input("Grasp object? Y/n: ")
        if letter == 'y' or letter == 'Y':
            # Close the hand
            self.grasp_control_hithand_client()

        # Get the current actual joint position and palm pose
        self.palm_poses["closed"], self.hand_joint_states[
            "closed"] = self.get_hand_palm_pose_and_joint_state()

        # Save visual data after hand is closed
        self.save_only_depth_and_color(grasp_phase='during')

        # Lift the object
        if letter == 'y' or letter == 'Y':
            lift_pose = self.palm_poses["desired_pre"]
            lift_pose.pose.position.z += self.object_lift_height
            execution_success = False
            start = time.time()
            while not execution_success:
                if time.time() - start > 60:
                    rospy.loginfo('Could not find a lift pose')
                    break
                self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
                execution_success = self.execute_joint_trajectory_client()
                lift_pose.pose.position.x += np.random.uniform(-0.05, 0.05)
                lift_pose.pose.position.y += np.random.uniform(-0.05, 0.05)
                lift_pose.pose.position.z += np.random.uniform(0, 0.1)

        # Get the joint position and palm pose after lifting
        self.palm_poses["lifted"], self.hand_joint_states[
            "lifted"] = self.get_hand_palm_pose_and_joint_state()

        # Evaluate success
        self.label_grasp()

        return True