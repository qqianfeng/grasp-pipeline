#!/usr/bin/env python
from __future__ import division
from grasp_pipeline.srv import *
from grasp_pipeline.utils.align_object_frame import align_object
from grasp_pipeline.utils import utils
from grasp_pipeline.utils.utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped, plot_voxel
from std_srvs.srv import SetBool, SetBoolRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import copy
import datetime
from multiprocessing import Process
import numpy as np
import cv2
import math
import open3d as o3d
import os
import rospy
import tf
import tf.transformations as tft
import tf2_geometry_msgs
import tf2_ros
import time
import sys
from moveit_commander import PlanningSceneInterface
sys.path.append('..')
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, SetBoolRequest

from grasp_pipeline.utils.utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped, plot_voxel
from grasp_pipeline.utils import utils
from grasp_pipeline.utils.open3d_draw_with_timeout import draw_with_time_out
from grasp_pipeline.utils.align_object_frame import align_object
from grasp_pipeline.srv import *
from grasp_pipeline.msg import *

from uuid import uuid4

camera_T_world_buffer = None
world_T_camera_buffer = None


class GraspClient():
    """ This class is a wrapper around all the individual functionality involved in grasping experiments.
    """
    def __init__(self, is_rec_sess, grasp_data_recording_path='', is_eval_sess=False):
        rospy.init_node('grasp_client', log_level=rospy.INFO)
        self.grasp_data_recording_path = grasp_data_recording_path
        if grasp_data_recording_path != '':
            self.create_grasp_folder_structure(self.grasp_data_recording_path)
        # Save metainformation on object to be grasped in these vars
        # This dict holds info about object name, pose, meshpath
        self.object_metadata = dict()
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
        self.base_path = os.path.join('/home/', os.getlogin())
        self.scene_pcd_save_path = os.path.join(self.base_path, 'scene.pcd')
        self.object_pcd_save_path = os.path.join(self.base_path, 'object.pcd')
        self.bps_object_path = os.path.join(self.base_path, 'pcd_enc.npy')
        self.object_pcd_record_path = ''

        # This variable stores all the information on multiple heuristically sampled grasping pre shapes
        self.heuristic_preshapes = None
        # The chosen variables store one specific preshape (palm_pose, hithand_joint_state, is_top_grasp)
        self.chosen_is_top_grasp = True  # Default True

        self.panda_planned_joint_trajectory = None
        self.num_of_replanning_attempts = 2
        self.plan_without_approach_pose = False

        self.grasps_available = True
        self.grasp_types = ["side1", "side2", "top"]
        self.previous_grasp_type = None
        self.chosen_grasp_type = 'unspecified'

        self.object_lift_height = 0.2  # Lift the object 20 cm
        self.success_tolerance_lift_height = 0.05
        self.object_segment_response = None
        self.grasp_label = None

        # special label made for multi obj data generation
        self.grasp_pose_collide_target_object = 0
        self.grasp_pose_collide_obstacle_objects = 0
        self.close_finger_collide_obstacle_objects = 0
        self.lift_motion_moved_obstacle_objects = 0

        # special label made for ffhnet evaluation
        self.collision_to_approach_pose = 0
        self.collision_to_grasp_pose = 0

        # For voxel server
        self.voxel_grid_dim = np.array([26, 26, 26])
        self.voxel_grid_dim_full = np.array([32, 32, 32])
        self.voxel_translation_dim = (self.voxel_grid_dim_full - self.voxel_grid_dim) // 2

        self.is_rec_sess = is_rec_sess
        self.is_eval_sess = is_eval_sess

        self.name_of_obstacle_objects_in_moveit_scene = set()
        self.object_name_to_moveit_name = dict()

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
    def log_object_cycle_time(self, cycle_time):
        file_path = os.path.join(self.base_path, 'grasp_timing.txt')
        with open(file_path, 'a') as timing_file:
            timing_file.writelines(self.object_metadata["name_rec_path"] + ': ' + str(cycle_time) +
                                   '\n')

    def log_num_grasps_removed(self, n_grasps_before, n_grasps_after, thresh):
        file_path = os.path.join(self.base_path, 'num_grasps_removed.txt')
        with open(file_path, 'a') as f:
            f.writelines(self.object_metadata["name"] + ', ' + str(thresh) + ', ' +
                         str(n_grasps_before) + ', ' + str(n_grasps_after) + '\n')

    def transform_pose(self, pose, from_frame, to_frame):
        assert pose.header.frame_id == from_frame

        tf2_pose = tf2_geometry_msgs.PoseStamped()
        tf2_pose.pose = pose.pose
        tf2_pose.header.frame_id = from_frame
        pose_stamped = self.tf_buffer.transform(tf2_pose, to_frame, rospy.Duration(5))

        return pose_stamped

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
        rec_sess_path = os.path.join(base_path, 'grasp_data/recording_sessions')
        if os.path.exists(rec_sess_path):
            self.sess_id_num = int(sorted(os.listdir(rec_sess_path))[-1].split('_')[-1]) + 1
            self.grasp_id_num = 0
            # if os.listdir(rec_sess_path + '/recording_session_' +
            #               str(self.sess_id_num - 1).zfill(4)):
            #     self.grasp_id_num = int(
            #         sorted(
            #             os.listdir(rec_sess_path + '/recording_session_' +
            #                        str(self.sess_id_num - 1).zfill(4)))[-1].split('_')[-1])
            # else:
            #     self.grasp_id_num = 0
        else:  # if the path did not exist yet, this is the first recording
            self.sess_id_num = 1
            self.grasp_id_num = 0
            os.makedirs(rec_sess_path)
            rospy.loginfo('This is the first recording, no prior recordings found.')

        self.sess_id_str = str(self.sess_id_num).zfill(4)
        self.grasp_id_str = str(self.grasp_id_num).zfill(4)
        rospy.loginfo('Session id: ' + self.sess_id_str)
        rospy.loginfo('Grasp id: ' + self.grasp_id_str)
        self.curr_rec_sess_path = rec_sess_path + \
            '/recording_session_' + self.sess_id_str
        os.mkdir(self.curr_rec_sess_path)

    def update_object_metadata(self, object_metadata):
        """ Update the metainformation about the object to be grasped
        """
        self.grasp_id_num = 0
        self.object_metadata = object_metadata

    def create_dirs_new_grasp_trial(self, is_new_pose_or_object=False):
        """ This should be called anytime before a new grasp trial is attempted as it will create the necessary folder structure.
        """
        # Check if directory for object name exists
        object_rec_path = os.path.join(self.curr_rec_sess_path,
                                       self.object_metadata["name_rec_path"])
        if not os.path.exists(object_rec_path):
            os.mkdir(object_rec_path)

        self.grasp_id_num += 1
        self.grasp_id_str = str(self.grasp_id_num).zfill(4)

        rospy.loginfo('Grasp id: ' + self.grasp_id_str)

        self.curr_grasp_trial_path = object_rec_path + '/grasp_' + self.grasp_id_str
        if os.path.exists(self.curr_grasp_trial_path):
            rospy.logerr("Path for grasp trial already exists, something is wrong.")
        os.mkdir(self.curr_grasp_trial_path)
        os.mkdir(self.curr_grasp_trial_path + '/during_grasp')
        os.mkdir(self.curr_grasp_trial_path + '/post_grasp')
        if is_new_pose_or_object:
            os.mkdir(self.curr_grasp_trial_path + '/pre_grasp')
            os.mkdir(self.curr_grasp_trial_path + '/single_grasp')

    def create_dirs_new_grasp_trial_multi_obj(self, pose_idx, is_new_pose_or_object=False):
        """ This should be called anytime before a new grasp trial is attempted as it will create the necessary folder structure.
        """
        # Check if directory for object name exists
        if not is_new_pose_or_object:
            object_rec_path = os.path.join(self.curr_rec_sess_path,
                                            self.object_metadata["name_rec_path"]+ '_pose_' + str(pose_idx))
            assert os.path.exists(object_rec_path)

            self.grasp_id_num += 1
            self.grasp_id_str = str(self.grasp_id_num).zfill(4)

            rospy.loginfo('Grasp id: ' + self.grasp_id_str)

            self.curr_grasp_trial_path = object_rec_path + '/grasp_' + self.grasp_id_str
            if os.path.exists(self.curr_grasp_trial_path):
                rospy.logerr("Path for grasp trial already exists, something is wrong.")
            os.mkdir(self.curr_grasp_trial_path)
            os.mkdir(self.curr_grasp_trial_path + '/during_grasp')
            os.mkdir(self.curr_grasp_trial_path + '/post_grasp')
        else:
            object_rec_path = os.path.join(self.curr_rec_sess_path,
                                            self.object_metadata["name_rec_path"]+ '_pose_' + str(pose_idx))
            assert not os.path.exists(object_rec_path)
            os.mkdir(object_rec_path)

            self.grasp_id_num = 1
            self.grasp_id_str = str(self.grasp_id_num).zfill(4)

            rospy.loginfo('Grasp id: ' + self.grasp_id_str)

            self.curr_grasp_trial_path = object_rec_path + '/grasp_' + self.grasp_id_str
            if os.path.exists(self.curr_grasp_trial_path):
                rospy.logerr("Path for grasp trial already exists, something is wrong.")
            os.mkdir(self.curr_grasp_trial_path)
            os.mkdir(self.curr_grasp_trial_path + '/during_grasp')
            os.mkdir(self.curr_grasp_trial_path + '/post_grasp')
            os.mkdir(self.curr_grasp_trial_path + '/pre_grasp')
            os.mkdir(self.curr_grasp_trial_path + '/single_grasp')

    def _setup_workspace_boundaries(self):
        """ Sets the boundaries in which an object can be spawned and placed.
        Gets called
        """
        self.spawn_object_x_min, self.spawn_object_x_max = 0.45, 0.65
        self.spawn_object_y_min, self.spawn_object_y_max = -0.2, 0.2

    def generate_random_object_pose_for_experiment(self):
        """Generates a random x,y position and z orientation within object_spawn boundaries for grasping experiments.
        """
        rand_x = np.random.uniform(self.spawn_object_x_min, self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min, self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [
            rand_x, rand_y, self.object_metadata["spawn_height_z"],
            self.object_metadata["spawn_angle_roll"], 0, rand_z_orientation
        ]
        rospy.logdebug('Generated random object pose:')
        rospy.logdebug(object_pose)
        object_pose_stamped = get_pose_stamped_from_array(object_pose)
        self.object_metadata["mesh_frame_pose"] = object_pose_stamped

    def choose_random_grasp_preshape(self):
        grasp_idx = np.random.randint(0, len(self.heuristic_preshapes.palm_goal_poses_world))
        # Store the chosen grasp pose and appraoch pose as well as joint state in corresponding dicts
        self.palm_poses["desired_pre"] = self.heuristic_preshapes.palm_goal_poses_world[grasp_idx]
        self.palm_poses["approach"] = self.heuristic_preshapes.palm_approach_poses_world[grasp_idx]
        self.hand_joint_states["desired_pre"] = self.heuristic_preshapes.hithand_joint_states[
            grasp_idx]

        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]
        self.chosen_grasp_idx = grasp_idx

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
            rospy.logerr('No grasp type is available anymore')
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
        self.palm_poses["desired_pre"] = self.heuristic_preshapes.palm_goal_poses_world[grasp_idx]
        self.palm_poses["approach"] = self.heuristic_preshapes.palm_approach_poses_world[grasp_idx]
        self.hand_joint_states["desired_pre"] = self.heuristic_preshapes.hithand_joint_states[
            grasp_idx]

        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]
        self.chosen_grasp_idx = grasp_idx

        self.chosen_grasp_type = grasp_type
        rospy.loginfo("Chosen grasp type is: " + str(self.chosen_grasp_type))
        # If this IS the first execution initialize previous grasp_type with current:
        if (len(self.top_idxs) + len(self.side1_idxs) +
                len(self.side2_idxs)) == self.num_preshapes:
            self.previous_grasp_type = self.chosen_grasp_type

    # ++++++++ PART II: Second part consist of all clients that interact with different nodes/services ++++++++++++
    def create_moveit_scene_client(self):
        # todo add multi objects
        wait_for_service('create_moveit_scene')
        try:
            req = ManageMoveitSceneRequest()
            create_moveit_scene = rospy.ServiceProxy('create_moveit_scene', ManageMoveitScene)
            req.object_names = [self.object_metadata["name"]]
            req.object_mesh_paths = [self.object_metadata["collision_mesh_path"]]
            req.object_pose_worlds = [self.object_metadata["mesh_frame_pose"]]
            create_scene_response = create_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service create_moveit_scene call failed: %s' % e)
        rospy.logdebug('Service create_moveit_scene is executed.')

    def clean_moveit_scene_client(self):
        wait_for_service('clean_moveit_scene')
        try:
            req = ManageMoveitSceneRequest()
            clean_moveit_scene = rospy.ServiceProxy('clean_moveit_scene', ManageMoveitScene)
            create_scene_response = clean_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service clean_moveit_scene call failed: %s' % e)
        rospy.logdebug('Service clean_moveit_scene is executed.')

    def control_hithand_config_client(self, joint_conf=None):
        wait_for_service('control_hithand_config')
        try:
            req = ControlHithandRequest()
            control_hithand_config = rospy.ServiceProxy('control_hithand_config', ControlHithand)
            if joint_conf == None:
                jc = self.hand_joint_states["desired_pre"]
                print("##### THUMB 0 ANGLE: " + str(jc.position[16]))
            else:
                if len(joint_conf.position) == 15:
                    jc = utils.full_joint_conf_from_vae_joint_conf(joint_conf)
                elif len(joint_conf.position) == 20:
                    jc = joint_conf
                else:
                    raise Exception('Given joint state has wrong dimension, must be 15 or 20')
            req.hithand_target_joint_state = jc
            res = control_hithand_config(req)
        except rospy.ServiceException as e:
            rospy.logerr('Service control_hithand_config call failed: %s' % e)
        rospy.logdebug('Service control_allegro_config is executed.')

    def check_pose_validity_utah_client(self, grasp_pose):
        wait_for_service('check_pose_validity_utah')
        try:
            check_pose_validity_utah = rospy.ServiceProxy('check_pose_validity_utah',
                                                          CheckPoseValidity)
            req = CheckPoseValidityRequest()
            req.object = self.object_segment_response.object
            req.pose = grasp_pose
            res = check_pose_validity_utah(req)
        except rospy.ServiceException as e:
            rospy.logerr('Service check_pose_validity_utah call failed: %s' % e)
        rospy.logdebug('Service check_pose_validity_utah is executed.')
        return res.is_valid

    def encode_pcd_with_bps_client(self):
        """ Encodes a pcd from disk (assumed static location) with bps_torch and saves the result to disk,
        from where the infer_grasp server can load it to sample grasps.
        """
        wait_for_service('encode_pcd_with_bps')
        try:
            encode_pcd_with_bps = rospy.ServiceProxy('encode_pcd_with_bps', SetBool)
            req = SetBoolRequest(data=True)
            res = encode_pcd_with_bps(req)
        except rospy.ServiceException as e:
            rospy.logerr('Service encode_pcd_with_bps call failed: %s' % e)
        rospy.logdebug('Service encode_pcd_with_bps is executed.')

    def evaluate_and_filter_grasp_poses_client(self, palm_poses, joint_confs, thresh):
        """Filter out all the grasp poses for which the FFHEvaluator predicts a success probability of less than thresh

        Args:
            palm_poses ([type]): [description]
            joint_confs ([type]): [description]
            thresh (float): Minimum probability of success. Between 0 and 1
        """
        wait_for_service('evaluate_and_filter_grasp_poses')
        try:
            evaluate_and_filter_grasp_poses = rospy.ServiceProxy('evaluate_and_filter_grasp_poses',
                                                                 EvaluateAndFilterGraspPoses)
            req = EvaluateAndFilterGraspPosesRequest()
            req.thresh = thresh
            req.palm_poses = palm_poses
            req.joint_confs = joint_confs
            res = evaluate_and_filter_grasp_poses(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service evaluate_and_filter_grasp_poses call failed: %s' % e)
        rospy.logdebug('Service evaluate_and_filter_grasp_poses.')
        return res.palm_poses, res.joint_confs

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

    def filter_palm_goal_poses_client(self,palm_poses=False):
        wait_for_service('filter_palm_goal_poses')
        try:
            filter_palm_goal_poses = rospy.ServiceProxy('filter_palm_goal_poses', FilterPalmPoses)
            req = FilterPalmPosesRequest()
            if palm_poses is False:
                req.palm_goal_poses_world = self.heuristic_preshapes.palm_goal_poses_world
            else:
                req.palm_goal_poses_world = palm_poses
            res = filter_palm_goal_poses(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service filter_palm_goal_poses call failed: %s' % e)
        rospy.logdebug('Service filter_palm_goal_poses is executed.')
        print("filtered so many:", len(res.prune_idxs)/len(req.palm_goal_poses_world))
        return res.prune_idxs, res.no_ik_idxs, res.collision_idxs

    def generate_hithand_preshape_client(self):
        """Generate preshape that is sampled from the each point cloud.
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
            self.num_preshapes = len(self.heuristic_preshapes.palm_goal_poses_world)
            # Generate list with indexes for different grasp types
            raise NotImplementedError
            self.side1_idxs = range(0, self.num_preshapes / 3)
            if self.heuristic_preshapes.is_top_grasp[self.num_preshapes / 3] == True:
                self.top_idxs = range(self.num_preshapes / 3, 2 * self.num_preshapes / 3)
                self.side2_idxs = range(2 * self.num_preshapes / 3, self.num_preshapes)
            else:
                self.side2_idxs = range(self.num_preshapes / 3, 2 * self.num_preshapes / 3)
                self.top_idxs = range(2 * self.num_preshapes / 3, self.num_preshapes)

            self.grasps_available = True

        except rospy.ServiceException, e:
            rospy.logerr('Service generate_hithand_preshape call failed: %s' % e)
        rospy.logdebug('Service generate_hithand_preshape is executed.')

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
            rospy.logerr('Service generate_voxel_from_pcd call failed: %s' % e)
        rospy.logdebug('Service generate_voxel_from_pcd is executed.')

    def get_hand_palm_pose_and_joint_state(self):
        """ Returns pose stamped and joint state
            1. palm pose
            2. hithand joint state
        """
        trans = self.tf_buffer.lookup_transform('world',
                                                'palm_link_hithand',
                                                rospy.Time(0),
                                                timeout=rospy.Duration(10))

        palm_pose = PoseStamped()
        palm_pose.header.frame_id = 'world'
        palm_pose.pose.position.x = trans.transform.translation.x
        palm_pose.pose.position.y = trans.transform.translation.y
        palm_pose.pose.position.z = trans.transform.translation.z
        palm_pose.pose.orientation.x = trans.transform.rotation.x
        palm_pose.pose.orientation.y = trans.transform.rotation.y
        palm_pose.pose.orientation.z = trans.transform.rotation.z
        palm_pose.pose.orientation.w = trans.transform.rotation.w
        joint_state = rospy.wait_for_message("/hithand/joint_states", JointState, timeout=5)
        return palm_pose, joint_state

    def get_grasp_object_pose_client(self, obj_name=''):
        """ Get the current pose (not stamped) of the grasp object from Gazebo.
        """
        wait_for_service('gazebo/get_model_state')
        try:
            get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
            req = GetModelStateRequest()
            if len(obj_name) == 0:
                req.model_name = self.object_metadata["name"]
            else:
                req.model_name = obj_name
            res = get_model_state(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service grasp_control_hithand call failed: %s' % e)
        rospy.logdebug('Service grasp_control_hithand is executed.')
        return res.pose

    def get_preshape_for_all_points_client(self):
        """ Generates
        """
        wait_for_service('get_preshape_for_all_points')
        try:
            get_preshape_for_all_points = rospy.ServiceProxy('get_preshape_for_all_points',
                                                             GraspPreshape)
            req = GraspPreshapeRequest()
            if self.object_segment_response is None:
                raise Exception("self.object_segment_response.object attribute is None.")
            req.object = self.object_segment_response.object

            res = get_preshape_for_all_points(req)

            self.heuristic_preshapes = res
            self.num_preshapes = len(res.palm_goal_poses_world)
            self.top_idxs = [i for i, x in enumerate(res.face_ids) if x == 'top']
            self.side1_idxs = [i for i, x in enumerate(res.face_ids) if x == 'side1']
            self.side2_idxs = [i for i, x in enumerate(res.face_ids) if x == 'side2']
            self.grasps_available = True

        except rospy.ServiceException, e:
            rospy.logerr('Service get_preshape_for_all_points call failed: %s' % e)
        rospy.logdebug('Service get_preshape_for_all_points is executed.')

    def grasp_control_hithand_client(self):
        """ Call server to close hithand fingers and stop when joint velocities are close to zero.
        """
        wait_for_service('grasp_control_hithand')
        try:
            grasp_control_hithand = rospy.ServiceProxy('grasp_control_hithand', GraspControl)
            req = GraspControlRequest()
            res = grasp_control_hithand(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service grasp_control_hithand call failed: %s' % e)
        rospy.logdebug('Service grasp_control_hithand is executed.')

    def infer_grasp_poses_client(self, n_poses, bps_object):
        """Infers grasps by sampling randomly in the latent space and decodes them to full pose via VAE. Later it will include some sort of refinement.
        """
        wait_for_service('infer_grasp_poses')
        try:
            infer_grasp_poses = rospy.ServiceProxy('infer_grasp_poses', InferGraspPoses)
            req = InferGraspPosesRequest()
            req.n_poses = n_poses
            req.bps_object = np.squeeze(bps_object)
            res = infer_grasp_poses(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service infer_grasp_poses call fialed: %s' % e)
        rospy.logdebug('Service infer_grasp_poses is executed')
        return res.palm_poses, res.joint_confs

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
            rospy.logerr('Service plan_arm_trajectory call failed: %s' % e)
        rospy.logdebug('Service plan_arm_trajectory is executed.')
        self.panda_planned_joint_trajectory = res.trajectory
        return res.success

    def plan_cartesian_path_trajectory_client(self, place_goal_pose=None):
        """Given place_goal_pose for evaluation. If not given, it's used for data generation.
        """
        wait_for_service('plan_cartesian_path_trajectory')
        try:
            plan_cartesian_path_trajectory = rospy.ServiceProxy('plan_cartesian_path_trajectory',
                                                                PlanCartesianPathTrajectory)
            req = PlanCartesianPathTrajectoryRequest()
            # Change the reference frame of desired_pre and approach pose to be
            if place_goal_pose is not None:
                palm_approach_pose_world = self.approach_pose_from_palm_pose(place_goal_pose)
                req.palm_approach_pose_world = self.add_position_noise(palm_approach_pose_world)
                req.palm_goal_pose_world = place_goal_pose
            else:
                # self.plam_poses value are saved in data generation pipeline
                req.palm_goal_pose_world = self.palm_poses["desired_pre"]
                req.palm_approach_pose_world = self.palm_poses["approach"]
            res = plan_cartesian_path_trajectory(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service plan_arm_trajectory call failed: %s' % e)
        rospy.logdebug('Service plan_arm_trajectory is executed %s.' % str(res.success))
        self.panda_planned_joint_trajectory = res.trajectory
        return res.success, res.fraction

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

    def record_collision_data_client(self):
        """ self.heuristic_preshapes stores all grasp poses. Self.prune_idxs contains idxs of poses in collision. Store
        these poses too, but convert to true object mesh frame first
        """
        wait_for_service('record_collision_data')
        try:
            # First select only the hithand joint states and heuristic preshapes which are in collision, as indicated by self.prune_idxs
            palm_poses_collision = [
                self.heuristic_preshapes.palm_goal_poses_world[i] for i in self.no_ik_idxs
            ]
            joint_states_collision = [
                self.heuristic_preshapes.hithand_joint_states[i] for i in self.no_ik_idxs
            ]

            # Then transform the poses from world frame to object mesh frame
            palm_goal_poses_mesh_frame = []
            for pose in palm_poses_collision:
                pose_mesh_frame = self.transform_pose(pose, 'world', 'object_mesh_frame')
                palm_goal_poses_mesh_frame.append(pose_mesh_frame)

            # Get service proxy
            record_collision_data = rospy.ServiceProxy('record_collision_data',
                                                       RecordCollisionData)

            # Build request only send the joint states and palm goal poses which are in collision
            req = RecordCollisionDataRequest()
            req.failure_type = 'no_ik'
            req.object_name = self.object_metadata["name_rec_path"]
            req.object_world_poses = len(
                self.no_ik_idxs) * [self.object_metadata["mesh_frame_pose"]]
            req.preshapes_palm_mesh_frame_poses = palm_goal_poses_mesh_frame
            req.preshape_hithand_joint_states = joint_states_collision

            res = record_collision_data(req)

        except rospy.ServiceException, e:
            rospy.logerr('Service record_collision_data call failed: %s' % e)
        rospy.logdebug('Service record_collision_data is executed.')

        wait_for_service('record_collision_data')
        try:
            # First select only the hithand joint states and heuristic preshapes which are in collision, as indicated by self.prune_idxs
            palm_poses_collision = [
                self.heuristic_preshapes.palm_goal_poses_world[i] for i in self.collision_idxs
            ]
            joint_states_collision = [
                self.heuristic_preshapes.hithand_joint_states[i] for i in self.collision_idxs
            ]

            # Then transform the poses from world frame to object mesh frame
            palm_goal_poses_mesh_frame = []
            for pose in palm_poses_collision:
                pose_mesh_frame = self.transform_pose(pose, 'world', 'object_mesh_frame')
                palm_goal_poses_mesh_frame.append(pose_mesh_frame)

            # Get service proxy
            record_collision_data = rospy.ServiceProxy('record_collision_data',
                                                       RecordCollisionData)

            # Build request only send the joint states and palm goal poses which are in collision
            req = RecordCollisionDataRequest()
            req.failure_type = 'collision'
            req.object_name = self.object_metadata["name_rec_path"]
            req.object_world_poses = len(
                self.collision_idxs) * [self.object_metadata["mesh_frame_pose"]]
            req.preshapes_palm_mesh_frame_poses = palm_goal_poses_mesh_frame
            req.preshape_hithand_joint_states = joint_states_collision

            res = record_collision_data(req)

        except rospy.ServiceException, e:
            rospy.logerr('Service record_collision_data call failed: %s' % e)
        rospy.logdebug('Service record_collision_data is executed.')

    def record_grasp_trial_data_client(self):
        """ self.heuristic_preshapes stores all grasp poses. Self.prune_idxs contains idxs of poses in collision. Store
        these poses too, but convert to true object mesh frame first
        """
        wait_for_service('record_grasp_trial_data')
        try:
            # First transform the poses from world frame to object mesh frame
            desired_pose_mesh_frame = self.transform_pose(self.palm_poses["desired_pre"], 'world',
                                                          'object_mesh_frame')
            true_pose_mesh_frame = self.transform_pose(self.palm_poses["true_pre"], 'world',
                                                       'object_mesh_frame')
            # Get service proxy
            record_grasp_trial_data = rospy.ServiceProxy('record_grasp_trial_data',
                                                         RecordGraspTrialData)

            # Build request
            req = RecordGraspTrialDataRequest()
            req.object_name = self.object_metadata["name_rec_path"]
            req.time_stamp = datetime.datetime.now().isoformat()
            req.is_top_grasp = self.chosen_is_top_grasp
            req.grasp_success_label = self.grasp_label
            req.collision_to_approach_pose = self.collision_to_approach_pose
            req.collision_to_grasp_pose = self.collision_to_grasp_pose

            req.object_mesh_frame_world = self.object_metadata["mesh_frame_pose"]
            req.desired_preshape_palm_mesh_frame = desired_pose_mesh_frame
            req.true_preshape_palm_mesh_frame = true_pose_mesh_frame
            req.desired_joint_state = self.hand_joint_states["desired_pre"]
            req.true_joint_state = self.hand_joint_states["true_pre"]
            req.closed_joint_state = self.hand_joint_states["closed"]
            req.lifted_joint_state = self.hand_joint_states["lifted"]

            # Call service
            res = record_grasp_trial_data(req)

        except rospy.ServiceException, e:
            rospy.logerr('Service record_grasp_trial_data call failed: %s' % e)
        rospy.logdebug('Service record_grasp_trial_data is executed.')

    def record_grasp_trial_data_multi_obj_client(self,objects=False):
        """ self.heuristic_preshapes stores all grasp poses. Self.prune_idxs contains idxs of poses in collision. Store
        these poses too, but convert to true object mesh frame first

        label:
        - grasp pose collide target_object -> this means either grasp collides with target object or motion planning is bad (can we get rid of bad motion planning)
        - grasp pose collide other objects ->  this means either grasp collides with other objects or motion planning is bad
        - close_finger_collide_other_objects -> this could cause by target object moved and pushes other objects
        - grasp_success
        - grasp_failure
        - grasp lift moved other objects -> objects were moved by lift motion or by previous any motion. Not a problem of grasping but motion planning.
                                            however this can happen that even the grasp pose is good, wired lift motion made it a failed grasp. Can consider removing them.

        """
        wait_for_service('record_grasp_trial_multi_obj_data')
        try:
            # First transform the poses from world frame to object mesh frame
            desired_pose_mesh_frame = self.transform_pose(self.palm_poses["desired_pre"], 'world',
                                                          'object_mesh_frame')
            true_pose_mesh_frame = self.transform_pose(self.palm_poses["true_pre"], 'world',
                                                       'object_mesh_frame')
            # Get service proxy
            record_grasp_trial_multi_obj_data = rospy.ServiceProxy('record_grasp_trial_multi_obj_data',
                                                         RecordGraspTrialMultiObjData)

            # Build request
            req = RecordGraspTrialMultiObjDataRequest()
            req.object_name = self.object_metadata["name_rec_path"]
            if objects is not False:
                print("objects[0]['name']:",objects[0]['name'])
                print("objects[1]['name']:",objects[1]['name'])
                print("objects[2]['name']:",objects[2]['name'])
                req.obstacle1_name = objects[0]['name']
                req.obstacle2_name = objects[1]['name']
                req.obstacle3_name = objects[2]['name']
            req.time_stamp = datetime.datetime.now().isoformat()
            req.is_top_grasp = self.chosen_is_top_grasp

            # labels for each grasp trial
            req.grasp_success_label = self.grasp_label
            req.grasp_pose_collide_target_object = self.grasp_pose_collide_target_object
            req.grasp_pose_collide_obstacle_objects = self.grasp_pose_collide_obstacle_objects
            req.close_finger_collide_obstacle_objects = self.close_finger_collide_obstacle_objects
            req.lift_motion_moved_obstacle_objects = self.lift_motion_moved_obstacle_objects

            req.object_mesh_frame_world = self.object_metadata["mesh_frame_pose"]
            if objects is not False:
                req.obstacle1_mesh_frame_world = objects[0]['mesh_frame_pose']
                req.obstacle2_mesh_frame_world = objects[1]['mesh_frame_pose']
                req.obstacle3_mesh_frame_world = objects[2]['mesh_frame_pose']
            # desired_pose_mesh_frame is the palm pose loaded for training.
            req.desired_preshape_palm_mesh_frame = desired_pose_mesh_frame
            req.true_preshape_palm_mesh_frame = true_pose_mesh_frame
            req.desired_joint_state = self.hand_joint_states["desired_pre"]
            req.true_joint_state = self.hand_joint_states["true_pre"]
            req.closed_joint_state = self.hand_joint_states["closed"]
            req.lifted_joint_state = self.hand_joint_states["lifted"]

            # Call service
            res = record_grasp_trial_multi_obj_data(req)

        except rospy.ServiceException, e:
            rospy.logerr('Service record_grasp_trial_multi_obj_data call failed: %s' % e)
        rospy.logdebug('Service record_grasp_trial_multi_obj_data is executed.')

    # This seems never used!!!
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
            req.object_world_sim_pose = self.object_metadata["mesh_frame_pose"]
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
            rospy.logerr('Service record_grasp_data call failed: %s' % e)
        rospy.logdebug('Service record_grasp_data is executed.')

    def record_sim_grasp_data_utah_client(self, grasp_id, object_name, grasp_config_obj, is_top,
                                          label):
        wait_for_service('record_grasp_data_utah')
        try:
            record_sim_grasp_data_utah = rospy.ServiceProxy("record_grasp_data_utah", SimGraspData)
            req = SimGraspDataRequest()
            req.grasp_id = grasp_id
            req.object_name = str(self.object_metadata["name_rec_path"])
            req.grasp_config_obj = list(grasp_config_obj)
            req.top_grasp = bool(is_top)
            req.sparse_voxel = list(self.object_metadata["sparse_voxel_grid"])
            req.dim_w_h_d = list(self.object_metadata["aligned_dim_whd_utah"])

            res = record_sim_grasp_data_utah(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service sim_grasp_data_utah call failed: %s' % e)
        rospy.logdebug('Service sim_grasp_data_utah is executed.')

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
            rospy.logerr('Service reset_hithand_joints call failed: %s' % e)
        rospy.logdebug('Service reset_hithand_joints is executed.')

    def save_visual_data_client(self, save_pcd=True):
        wait_for_service('save_visual_data')
        try:
            save_visual_data = rospy.ServiceProxy('save_visual_data', SaveVisualData)
            req = SaveVisualDataRequest()
            req.color_img_save_path = self.color_img_save_path
            req.depth_img_save_path = self.depth_img_save_path
            if save_pcd == True:
                req.scene_pcd_save_path = self.scene_pcd_save_path
            res = save_visual_data(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service save_visual_data call failed: %s' % e)
        rospy.logdebug('Service save_visual_data is executed %s' % res.success)

    def segment_object_client(self, align_object_world=True, down_sample_pcd=True, need_to_transfer_pcd_to_world_frame=False):
        wait_for_service('segment_object')
        try:
            segment_object = rospy.ServiceProxy('segment_object', SegmentGraspObject)
            req = SegmentGraspObjectRequest()
            req.down_sample_pcd = down_sample_pcd
            req.need_to_transfer_pcd_to_world_frame = need_to_transfer_pcd_to_world_frame
            req.scene_pcd_path = self.scene_pcd_save_path
            req.object_pcd_path = self.object_pcd_save_path
            req.object_pcd_record_path = self.object_pcd_record_path
            self.object_segment_response = segment_object(req)
            self.object_metadata["seg_pose"] = PoseStamped(
                header=Header(frame_id='world'), pose=self.object_segment_response.object.pose)
            # whd: Width Height Depth
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

                rospy.logdebug("whd before alignment: ")
                rospy.logdebug(self.object_metadata["seg_dim_whd"])

                self.update_object_pose_aligned_client()
                self.object_metadata["aligned_dim_whd_utah"] = [
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ]
                rospy.logdebug("whd from alignment: ")
                rospy.logdebug(self.object_metadata["aligned_dim_whd_utah"])
                rospy.sleep(0.2)
                if not self.is_rec_sess:
                    rospy.sleep(2)
                bb_extent = np.ones(4)
                bb_extent[:3] = np.array(self.object_metadata["seg_dim_whd"])
                trans = self.tf_buffer.lookup_transform('object_pose_aligned',
                                                        'object_pose',
                                                        rospy.Time(0),
                                                        timeout=rospy.Duration(5))
                quat = trans.transform.rotation
                aligned_T_pose = tft.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
                bb_extent_aligned = np.abs(aligned_T_pose.dot(bb_extent))
                self.object_metadata["aligned_dim_whd"] = bb_extent_aligned[:3]
                self.object_segment_response.object.width = bb_extent_aligned[0]
                self.object_segment_response.object.height = bb_extent_aligned[1]
                self.object_segment_response.object.depth = bb_extent_aligned[2]
                rospy.logdebug("whd: ")
                rospy.logdebug([
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ])

        except rospy.ServiceException, e:
            rospy.logerr('Service segment_object call failed: %s' % e)
        rospy.logdebug('Service segment_object is executed.')

    def update_moveit_scene_client(self):
        wait_for_service('update_moveit_scene')
        try:
            update_moveit_scene = rospy.ServiceProxy('update_moveit_scene', ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            req = ManageMoveitSceneRequest()
            req.object_names = [self.object_metadata["name"]]
            req.object_mesh_paths = [self.object_metadata["collision_mesh_path"]]
            req.object_pose_worlds = [self.object_metadata["mesh_frame_pose"]]
            self.update_scene_response = update_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_moveit_scene call failed: %s' % e)
        rospy.logdebug(
            'Service update_moveit_scene is executed %s.' % str(self.update_scene_response))

    def update_gazebo_object_client(self):
        """Gazebo management client, spawns new object
        """
        wait_for_service('update_gazebo_object')
        object_pose_array = get_pose_array_from_stamped(self.object_metadata["mesh_frame_pose"])
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

    #####################
    ## Test spawn hand ##
    #####################
    def get_base_link_hithand_pose(self):
        trans = self.tf_buffer.lookup_transform('world',
                                                'base_link_hithand',
                                                rospy.Time(0),
                                                timeout=rospy.Duration(10))
        palm_pose = PoseStamped()
        palm_pose.header.frame_id = 'world'
        palm_pose.pose.position.x = trans.transform.translation.x
        palm_pose.pose.position.y = trans.transform.translation.y
        palm_pose.pose.position.z = trans.transform.translation.z
        palm_pose.pose.orientation.x = trans.transform.rotation.x
        palm_pose.pose.orientation.y = trans.transform.rotation.y
        palm_pose.pose.orientation.z = trans.transform.rotation.z
        palm_pose.pose.orientation.w = trans.transform.rotation.w
        palm_pose_in_flange = utils.get_pose_array_from_stamped(palm_pose)
        return palm_pose_in_flange

    def spawn_hand(self, pose_arr):
        """Gazebo management client, spawns new object

        Args:
            pose_arr (list): _description_. Example: [0.5, 0.0, 0.2, 0, 0, 0]
        """
        wait_for_service('update_gazebo_hand')
        # TODO: Clean up
        palm_pose_in_flange = np.array([[ 2.68490602e-01,  1.43867476e-01, -9.52478318e-01,  2.00000000e-02],
                                        [ 9.00297098e-04,  9.88746286e-01,  1.49599368e-01,  0.00000000e+00],
                                        [ 9.63281883e-01, -4.10235378e-02,  2.65339562e-01,  6.00000000e-02],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        trans = self.tf_buffer.lookup_transform('base_link_hithand',
                                                'palm_link_hithand',
                                                rospy.Time(0),
                                                timeout=rospy.Duration(10))
        palm_pose = PoseStamped()
        palm_pose.header.frame_id = 'world'
        palm_pose.pose.position.x = trans.transform.translation.x
        palm_pose.pose.position.y = trans.transform.translation.y
        palm_pose.pose.position.z = trans.transform.translation.z
        palm_pose.pose.orientation.x = trans.transform.rotation.x
        palm_pose.pose.orientation.y = trans.transform.rotation.y
        palm_pose.pose.orientation.z = trans.transform.rotation.z
        palm_pose.pose.orientation.w = trans.transform.rotation.w
        hand_palm_pose_world = utils.hom_matrix_from_6D_pose(pose_arr[:3],pose_arr[3:])
        palm_pose_in_flange2 = utils.hom_matrix_from_pose_stamped(palm_pose)

        pose_arr_palm_hand = np.matmul(hand_palm_pose_world, np.linalg.inv(palm_pose_in_flange2))
        pose_stamped_palm_hand = utils.pose_stamped_from_hom_matrix(pose_arr_palm_hand,'world')
        pose_arr = utils.get_pose_array_from_stamped(pose_stamped_palm_hand)
        print("spawn hand at", pose_arr)

        try:
            update_gazebo_hand = rospy.ServiceProxy('update_gazebo_hand', UpdateHandGazebo)
            req = UpdateHandGazeboRequest()
            req.object_name = 'hand'

            req.object_model_file = rospy.get_param('hand_urdf_path')
            req.object_pose_array = pose_arr
            req.model_type = 'urdf'
            res = update_gazebo_hand(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_gazebo_hand call failed: %s' % e)
        rospy.logdebug('Service update_gazebo_hand is executed %s.' % str(res.success))
        return res.success

    def delete_hand(self):
        wait_for_service('delete_gazebo_hand')
        try:
            delete_gazebo_hand = rospy.ServiceProxy('delete_gazebo_hand', DeleteHandGazebo)
            res = delete_gazebo_hand()
        except rospy.ServiceException, e:
            rospy.logerr('Service delete_gazebo_hand call failed: %s' % e)
        rospy.logdebug('Service delete_gazebo_hand is executed %s.' % str(res.success))
        return res.success

    def create_hand_moveit_scene_(self):
        # todo add multi objects
        wait_for_service('create_moveit_scene')
        try:
            req = ManageMoveitSceneRequest()
            create_moveit_scene = rospy.ServiceProxy('create_moveit_scene', ManageMoveitScene)
            req.object_names = ['hand']
            req.object_mesh_paths = [self.object_metadata["collision_mesh_path"]]
            req.object_pose_worlds = [self.object_metadata["mesh_frame_pose"]]
            create_scene_response = create_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service create_moveit_scene call failed: %s' % e)
        rospy.logdebug('Service create_moveit_scene is executed.')

    def clean_moveit_scene_client(self):
        wait_for_service('clean_moveit_scene')
        # TODO: you need to specify the path for req: req.object_mesh_paths in order to remove the model in moveit scene.
        try:
            req = ManageMoveitSceneRequest()
            clean_moveit_scene = rospy.ServiceProxy('clean_moveit_scene', ManageMoveitScene)
            create_scene_response = clean_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service clean_moveit_scene call failed: %s' % e)
        rospy.logdebug('Service clean_moveit_scene is executed.')

    #####################################################
    ## below are codes for multiple objects generation ##
    #####################################################

    def spawn_obstacle_objects(self, objects, moveit=True):
        wait_for_service('create_new_scene')
        create_new_scene = rospy.ServiceProxy('create_new_scene', CreateNewScene)
        req = CreateNewSceneRequest()
        for grasp_object in objects:
            single_object = ObjectToBeSpawned()
            object_pose_array = get_pose_array_from_stamped(grasp_object["mesh_frame_pose"])
            single_object.object_name = grasp_object["name"]
            single_object.object_model_file = grasp_object["model_file"]
            single_object.object_pose_array = object_pose_array
            single_object.model_type = 'sdf'
            req.objects_in_new_scene.append(single_object)
        try:
            res = create_new_scene(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service create_new_scene call failed %s' % e)
            return False
        rospy.loginfo('Service create_new_scene is executed %s.' % str(res.success))

        for grasp_object in objects:
            pose = self.get_object_pose(grasp_object)
            if pose:
                # Update the sim_pose with the actual pose of the object after it came to rest
                grasp_object["mesh_frame_pose"] = PoseStamped(header=Header(frame_id='world'),
                                                                    pose=pose)
                if moveit:
                    uuid_str = str(uuid4())
                    self.add_to_moveit_scene(uuid_str, grasp_object)
                    self.object_name_to_moveit_name[grasp_object['name']] = uuid_str

        return res.success

    def remove_obstacle_objects(self, objects,moveit=True):
        if moveit:
            self.remove_obstacle_objects_from_moveit_scene()
        wait_for_service('clear_scene')
        clear_scene = rospy.ServiceProxy('clear_scene', ClearScene)
        req = ClearSceneRequest()
        req.confirm = len(objects) > 0
        try:
            res = clear_scene(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service clear_scene call failed %s' % e)
            exit()
        rospy.logdebug('Service clear_scene is executed %s.' % str(res.success))
        return res.success

    def reset_obstacle_objects(self, objects):
        if not self.reset_scene(len(objects) > 0):
            print 'reset_obstacle_objects failed'
            exit()

    def get_object_pose(self, object_metadata):
        """ Get the current pose (not stamped) of the grasp object from Gazebo.
        """
        wait_for_service('gazebo/get_model_state')
        try:
            get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
            req = GetModelStateRequest()
            req.model_name = object_metadata["name"]
            res = get_model_state(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service gazebo/get_model_state call failed: %s' % e)
            return None
        rospy.logdebug('Service gazebo/get_model_state is executed.')
        return res.pose

    def add_to_moveit_scene(self, name, object_metadata):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        scene.add_mesh(name, object_metadata['mesh_frame_pose'], object_metadata["collision_mesh_path"])
        rospy.sleep(0.5)
        self.name_of_obstacle_objects_in_moveit_scene.add(name)

    def remove_obstacle_objects_from_moveit_scene(self):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        # TODO: for first time, nothing to remove but there are stuff in the moveit scene.
        for name in self.name_of_obstacle_objects_in_moveit_scene:
            scene.remove_world_object(name)
            print('MOVEIT remove:',name,'time:',time.time()-time1)
            rospy.sleep(0.5)
        self.name_of_obstacle_objects_in_moveit_scene.clear()

    def remove_from_moveit_scene(self, object_name):
        scene = PlanningSceneInterface()
        rospy.sleep(0.5)
        name = self.object_name_to_moveit_name.pop(object_name, object_name)
        time1 = time.time()
        scene.remove_world_object(name)
        print('MOVEIT remove:',name,'time:',time.time()-time1)
        rospy.sleep(0.5)
        if name in self.name_of_obstacle_objects_in_moveit_scene:
            self.name_of_obstacle_objects_in_moveit_scene.discard(name)

    def reset_scene(self, confirm):
        """Reset all object poses to original pose which is saved in snapshot.
        """
        wait_for_service('reset_scene')
        reset_scene = rospy.ServiceProxy('reset_scene', ResetScene)
        req = ResetSceneRequest()
        req.confirm = confirm
        try:
            res = reset_scene(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service reset_scene call failed: %s' % e)
            return None
        rospy.logdebug('Service reset_scene is executed.')
        return res.success

    #####################################################
    ## above are codes for multiple objects generation ##
    #####################################################

    def update_grasp_palm_pose_client(self, palm_pose):
        wait_for_service("update_grasp_palm_pose")
        try:
            update_grasp_palm_pose = rospy.ServiceProxy('update_grasp_palm_pose', UpdatePalmPose)
            req = UpdatePalmPoseRequest()
            req.palm_pose = palm_pose
            res = update_grasp_palm_pose(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_grasp_palm_pose call failed: %s' % e)
        rospy.logdebug('Service update_grasp_palm_pose is executed.')

    def update_object_mesh_frame_data_gen_client(self, object_mesh_frame_pose):
        wait_for_service("update_object_mesh_frame_data_gen")
        try:
            update_object_mesh_frame_data_gen = rospy.ServiceProxy(
                'update_object_mesh_frame_data_gen', UpdateObjectPose)
            req = UpdateObjectPoseRequest()
            req.object_pose_world = object_mesh_frame_pose
            res = update_object_mesh_frame_data_gen(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_object_mesh_frame_data_gen call failed: %s' % e)
        rospy.logdebug('Service update_object_mesh_frame_data_gen is executed.')

    def update_object_pose_aligned_client(self):
        wait_for_service("update_grasp_object_pose")
        try:
            update_grasp_object_pose = rospy.ServiceProxy('update_grasp_object_pose',
                                                          UpdateObjectPose)
            req = UpdateObjectPoseRequest()
            req.object_pose_world = self.object_metadata["aligned_pose"]
            res = update_grasp_object_pose(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_gazebo_object call failed: %s' % e)
        rospy.logdebug('Service update_grasp_object_pose is executed %s.' % str(res.success))

    def update_object_mesh_frame_pose_client(self):
        wait_for_service("update_object_mesh_frame_pose")
        try:
            update_object_mesh_frame_pose = rospy.ServiceProxy('update_object_mesh_frame_pose',
                                                               UpdateObjectPose)
            req = UpdateObjectPoseRequest()
            req.object_pose_world = self.object_metadata["mesh_frame_pose"]
            res = update_object_mesh_frame_pose(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service update_object_mesh_frame_pose call failed: %s' % e)
        rospy.logdebug('Service update_object_mesh_frame_pose is executed.')

    def visualize_grasp_pose_list_client(self, grasp_poses):
        wait_for_service("visualize_grasp_pose_list")
        try:
            visualize_grasp_pose_list = rospy.ServiceProxy('visualize_grasp_pose_list',
                                                           VisualizeGraspPoseList)
            req = VisualizeGraspPoseListRequest()
            req.grasp_pose_list = grasp_poses
            res = visualize_grasp_pose_list(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service visualize_grasp_poses_list failed: %s' % e)
        rospy.logdebug('Service visualize_grasp_poses_list is executed.')

    # =============================================================================================================
    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def add_position_noise(self, pose):
        pose.pose.position.x += np.random.uniform(-0.02, 0.02)
        pose.pose.position.y += np.random.uniform(-0.02, 0.02)
        pose.pose.position.z += np.random.uniform(0, 0.3)
        return pose

    @staticmethod
    def approach_pose_from_palm_pose(palm_pose):
        """Compute an appraoch pose from a desired palm pose, by subtracting a vector parallel to x-direction
        of palm pose (x-direction points towards object, therefore we can get further away from the object with this.)

        Args:
            palm_pose (PoseStamped): The desired pose of the palm

        Returns:
            approach_pose (PoseStamped): An approach pose further away from the object
        """
        dist_factor = 0.1

        # Extract info from palm pose
        t = palm_pose.pose.position
        q = palm_pose.pose.orientation
        R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        x_dir = R[:3, 0]
        curr_pos = [t.x, t.y, t.z]

        # Compute new position
        approach_pos = curr_pos - 0.1 * x_dir

        # TODO check if add some z offset, will this improve the path planning?

        # build approach pose
        approach_pose = copy.deepcopy(palm_pose)
        approach_pose.pose.position.x = approach_pos[0]
        approach_pose.pose.position.y = approach_pos[1]
        approach_pose.pose.position.z = approach_pos[2]
        return approach_pose

    def check_pose_validity_utah(self, grasp_pose):
        return self.check_pose_validity_utah_client(grasp_pose)

    def encode_pcd_with_bps(self):
        self.encode_pcd_with_bps_client()

    def evaluate_and_remove_grasps(self, palm_poses, joint_confs, thresh, visualize_poses=True):
        n_before = len(joint_confs)
        palm_poses_f, joint_confs_f = self.evaluate_and_filter_grasp_poses_client(
            palm_poses, joint_confs, thresh)
        if visualize_poses:
            self.visualize_grasp_pose_list_client(palm_poses_f)
        n_after = len(joint_confs_f)
        rospy.logdebug("Remaining grasps after filtering: %.2f" % (n_after / n_before))
        rospy.logdebug("This means %d grasps were removed." % (n_before - n_after))
        self.log_num_grasps_removed(n_before, n_after, thresh)
        return palm_poses_f, joint_confs_f

    def infer_grasp_poses(self, n_poses, visualize_poses=False, bps_object=None):
        if bps_object == None:
            bps_object = np.load(self.bps_object_path)
        palm_poses, joint_confs = self.infer_grasp_poses_client(n_poses=n_poses,
                                                                bps_object=bps_object)
        # TODO why this visualization is not working
        if visualize_poses:
            self.visualize_grasp_pose_list_client(palm_poses)
        return palm_poses, joint_confs

    def label_grasp(self):
        object_pose = self.get_grasp_object_pose_client()
        object_pos_delta_z = np.abs(object_pose.position.z -
                                    self.object_metadata["mesh_frame_pose"].pose.position.z)
        if object_pos_delta_z > (self.object_lift_height - self.success_tolerance_lift_height):
            self.grasp_label = 1
        else:
            rospy.logdebug("object_pos_delta_z: %f" % object_pos_delta_z)
            self.grasp_label = 0

        rospy.loginfo("The grasp label is: " + str(self.grasp_label))

    def remove_grasp_pose(self):
        if self.chosen_grasp_type == 'side1' and self.chosen_grasp_idx in self.side1_idxs:
            self.side1_idxs.remove(self.chosen_grasp_idx)
        elif self.chosen_grasp_type == 'side2' and self.chosen_grasp_idx in self.side2_idxs:
            self.side2_idxs.remove(self.chosen_grasp_idx)
        elif self.chosen_grasp_type == 'top' and self.chosen_grasp_idx in self.top_idxs:
            self.top_idxs.remove(self.chosen_grasp_idx)

        if len(self.top_idxs + self.side1_idxs + self.side2_idxs) == 0:
            self.grasps_available = False

    def reset_hithand_and_panda(self):
        """ Reset panda and hithand to their home positions
        """
        self.reset_hithand_joints_client()
        reset_plan_exists = self.plan_reset_trajectory_client()
        if reset_plan_exists:
            self.execute_joint_trajectory_client()
        self.delete_hand()

    def spawn_object(self, pose_type, pose_arr=None):
        # Generate a random valid object pose
        if pose_type == "random":
            self.generate_random_object_pose_for_experiment()

        elif pose_type == "init":
            # set the roll angle
            pose_arr[3] = self.object_metadata["spawn_angle_roll"]  # 0
            pose_arr[2] = self.object_metadata["spawn_height_z"]  # 0.05

            self.object_metadata["mesh_frame_pose"] = get_pose_stamped_from_array(pose_arr)

        #print "Spawning object here:", pose_arr

        # Update gazebo object, delete old object and spawn new one
        self.update_gazebo_object_client()

        # Now wait for 2 seconds for object to rest and update actual object position
        if pose_type == "init" or pose_type == "random":
            if self.is_rec_sess:
                rospy.sleep(3)
            object_pose = self.get_grasp_object_pose_client()

            # Update the sim_pose with the actual pose of the object after it came to rest
            self.object_metadata["mesh_frame_pose"] = PoseStamped(header=Header(frame_id='world'),
                                                                  pose=object_pose)

        # Update moveit scene object
        if not self.is_eval_sess:
            self.update_moveit_scene_client()

        # Update the true mesh pose
        self.update_object_mesh_frame_pose_client()

    #####################################################
    ## below are codes for multiple objects generation ##
    #####################################################

    def set_to_random_pose(self, object_metadata):
        """Generates a random x,y position and z orientation within object_spawn boundaries for grasping experiments.
        """
        rand_x = np.random.uniform(self.spawn_object_x_min, self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min, self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [
            rand_x, rand_y, object_metadata["spawn_height_z"], object_metadata["spawn_angle_roll"],
            0, rand_z_orientation
        ]
        rospy.logdebug('Generated random object pose:')
        rospy.logdebug(object_pose)
        object_pose_stamped = get_pose_stamped_from_array(object_pose)
        object_metadata["mesh_frame_pose"] = object_pose_stamped
        return object_metadata


    def save_visual_data(self, down_sample_pcd=True, object_pcd_record_path=''):
        """Does what it says.

        Args:
            down_sample_pcd (bool, optional): If this is True the pcd will be down sampled. It is
            necessary to down_sample during data gen, because for each point of the pcd one pose will be computed.
            During inference it should not be down sampled. Defaults to True.
            object_pcd_record_path (str, optional): [description]. Defaults to ''.
        """
        if down_sample_pcd == True:
            rospy.logdebug(
                "Point cloud will be down sampled AND transformed to WORLD frame. This is not correct for testing grasp sampler!"
            )
        else:
            rospy.logdebug(
                "Point cloud will not be down sampled BUT transformed to OBJECT CENTROID frame, which is parallel to camera frame. This is necessary for testing grasp sampler."
            )
        self.object_pcd_record_path = object_pcd_record_path
        self.set_visual_data_save_paths(grasp_phase='pre')
        self.save_visual_data_client()

    def segment_object_as_point_cloud(self, ROI):
        world_T_camera = _get_camera_to_world_transformation()

        # Get camera data
        color_image = cv2.imread(self.color_img_save_path)
        depth_path = self.depth_img_save_path
        depth_path = depth_path[:-4] + '.npy'
        depth_image = np.load(depth_path)

        # Create mask
        mask = np.zeros((color_image.shape[0], color_image.shape[1]), np.uint8)

        # GrabCut arrays
        bgdModel = np.zeros((1, 65), np.float64)
        fgbModel = np.zeros((1, 65), np.float64)

        # Run GrabCut
        init_rect = ROI
        cv2.grabCut(color_image, mask, init_rect, bgdModel, fgbModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        masked_image = color_image * mask2[:, :, np.newaxis]

        # Set area outside of the segmentation mask to zero
        depth_image *= mask2

        # Remove data with large depth offset from segmented object's median
        median = np.median(depth_image[depth_image > 0.000001])
        depth_image = np.where(abs(depth_image - median) < 0.1, depth_image, 0)

        # Load depth image as o3d.Image
        depth_image_o3d = o3d.geometry.Image(depth_image)

        # Generate point cloud from depth image
        pinhole_camera_intrinsic = _get_camera_intrinsics()
        object_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, pinhole_camera_intrinsic)

        # We keep single object pcd in center and camera orientation for FFHGenerator
        # object_pcd.transform(world_T_camera)
        # np.save("/home/vm/single_")
        object_pcd.translate((-1) * object_pcd.get_center())

        pcd_save_path = self.object_pcd_save_path
        o3d.io.write_point_cloud(pcd_save_path, object_pcd)
        draw_with_time_out(object_pcd, 3)

    def post_process_object_point_cloud(self):
        temp_var = self.scene_pcd_save_path
        self.scene_pcd_save_path = self.object_pcd_save_path
        self.segment_object_client(down_sample_pcd=False)
        self.scene_pcd_save_path = temp_var

    def select_ROIs(self, obstacle_objects):
        ROIs = []
        names = []
        for _ in range(len(obstacle_objects) + 1):
            self.save_visual_data(down_sample_pcd=False)
            color_image = cv2.imread(self.color_img_save_path)
            ROI = _select_ROI(color_image, _ == len(obstacle_objects))
            ROIs.append(ROI)
            name = self._get_name_of_objcet_in_ROI(ROI, obstacle_objects)
            names.append(name)
            self.change_model_visibility(name, False)

        self.make_all_visiable(obstacle_objects)

        user_input = raw_input('Are the objects in ROI detected correctly? [Y/n]')
        are_names_detected_correctly = user_input in ['Y', 'y', '']
        if not are_names_detected_correctly:
            ROIs = []
            names = []
        return ROIs, names

    def make_all_visiable(self, obstacle_objects):
        self.change_model_visibility(self.object_metadata['name'], True)
        for obj in obstacle_objects:
            self.change_model_visibility(obj['name'], True)


    def change_model_visibility(self, model_name, visible):
        wait_for_service("update_object_mesh_frame_pose")
        try:
            service_change_model_visibility = rospy.ServiceProxy('change_model_visibility',
                                                               ChangeModelVisibility)
            req = ChangeModelVisibilityRequest()
            req.model_name = model_name
            req.visible = visible
            res = service_change_model_visibility(req)
        except rospy.ServiceException, e:
            rospy.logerr('Service change_model_visibility call failed: %s' % e)
        rospy.logdebug('Service change_model_visibility is executed.')

    def _get_name_of_objcet_in_ROI(self, ROI, obstacle_objects):
        candidate_names = set()
        objects_inside_ROI = []
        object_positions = dict()

        candidate_names.add(self.object_metadata['name'])
        for obj in obstacle_objects:
            candidate_names.add(obj['name'])

        for name in candidate_names:
            pose = self.get_grasp_object_pose_client(obj_name=name)
            pose_numpy = np.array([pose.position.x, pose.position.y, pose.position.z])
            if np.allclose(pose_numpy, np.zeros_like(pose_numpy)):
                continue
            object_positions[name] = pose_numpy
            x, y = _project_point_in_world_onto_image_plane(
                pose.position.x,
                pose.position.y,
                pose.position.z,
                _get_camera_intrinsics()
            )
            if _is_point_inside_ROI(ROI, x, y):
                objects_inside_ROI.append(name)

        if len(objects_inside_ROI) == 0:
            raise RuntimeError('Nothing inside ROI!')

        if len(objects_inside_ROI) == 1:
            return objects_inside_ROI[0]

        camera_position = np.array([0.48, -0.846602, 0.360986])
        min_distance = None
        closest_object = None
        for name in object_positions.keys():
            object_position = object_positions[name]
            distance = np.linalg.norm(object_position - camera_position)
            if min_distance is None or distance < min_distance:
                closest_object = name
                min_distance = distance
        return closest_object

    def remove_ground_plane_and_robot(self, scene_pcd_path=None):
        """ Segmentation to remove ground plane.

        Args:
            scene_pcd_path (_type_, optional): _description_. Defaults to None.
        """
        if scene_pcd_path is None:
            scene_pcd_path = self.scene_pcd_save_path
        scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

        # segment the panda base from point cloud
        points = np.asarray(scene_pcd.points)  # shape [x,3]
        colors = np.asarray(scene_pcd.colors)

        # currently the mask cropping is removed
        mask1 = points[:, 0] > 0.1
        mask2 = points[:, 2] < 10
        mask = np.logical_and(mask1,mask2)
        del scene_pcd
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points[mask])
        scene_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        _, inliers = scene_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
        plane_removed_pcd = scene_pcd.select_down_sample(inliers, invert=True)
        o3d.io.write_point_cloud(rospy.get_param('multi_object_pcd_path'), plane_removed_pcd)

    def remove_ground_plane(self, scene_pcd_path=None):
        if scene_pcd_path is None:
            scene_pcd_path = self.scene_pcd_save_path
        scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

        _, inliers = scene_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
        plane_removed_pcd = scene_pcd.select_down_sample(inliers, invert=True)
        o3d.io.write_point_cloud(rospy.get_param('multi_object_pcd_path'), plane_removed_pcd)

    #####################################################
    ## above are codes for multiple objects generation ##
    #####################################################

    def transform_pcd_from_world_to_grasp(self, grasp,vis=False):
        world_multi_obj_pcd = o3d.io.read_point_cloud(rospy.get_param('multi_object_pcd_path'))
        world_T_grasp_pose = self.transform_pose(grasp, 'object_centroid_vae', 'world')
        grasp_pose_T_world = np.linalg.inv(world_T_grasp_pose)
        world_multi_obj_pcd.transform(grasp_pose_T_world)
        if vis:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1)
            print("original pcd")
            o3d.visualization.draw_geometries([pcd, origin])
            # TODO: wrong world_T_mesh in visualization
            origin.transform(world_T_grasp_pose)
            o3d.visualization.draw_geometries([pcd, origin])
            visualization.show_dataloader_grasp_with_pcd_in_world_frame(bps_path, obj_name, world_T_mesh, world_T_grasp_pose, palm_pose_hom
                        , pcd)
        o3d.io.write_point_cloud(rospy.get_param('multi_object_pcd_path'))

    def set_visual_data_save_paths(self, grasp_phase):
        if self.is_rec_sess:
            if grasp_phase not in ['single','pre', 'during', 'post']:
                rospy.logerr('Given grasp_phase is not valid. Must be pre, during or post.')

            self.depth_img_save_path = os.path.join(self.curr_grasp_trial_path,
                                                    grasp_phase + '_grasp',
                                                    self.object_metadata["name"] + '_depth.png')
            self.color_img_save_path = os.path.join(self.curr_grasp_trial_path,
                                                    grasp_phase + '_grasp',
                                                    self.object_metadata["name"] + '_color.jpg')
        else:
            self.depth_img_save_path = os.path.join(self.base_path, 'depth.png')
            self.color_img_save_path = os.path.join(self.base_path, 'color.jpg')

    def save_visual_data_and_record_grasp(self,objects=False):
        """
        Args:
            objects (bool, optional): Obstacle objects
        """
        # This function is used for data generation only.
        self.set_visual_data_save_paths(grasp_phase='post')
        self.save_visual_data_client(save_pcd=False)
        # self.generate_voxel_from_pcd_client()

        # if generate for single object, by default objects are false.
        if objects is False:
            self.record_grasp_trial_data_client()
        # if generate for multi objects
        else:
            self.record_grasp_trial_data_multi_obj_client(objects)

    def save_only_depth_and_color(self, grasp_phase):
        """ Saves only depth and color by setting scene_pcd_save_path to None. Resets scene_pcd_save_path afterwards.
        """
        self.set_visual_data_save_paths(grasp_phase=grasp_phase)
        self.save_visual_data_client(save_pcd=False)

    def save_visual_data_and_segment_object(self, down_sample_pcd=True, object_pcd_record_path=''):
        """Does what it says.

        Args:
            down_sample_pcd (bool, optional): If this is True the pcd will be down sampled. It is
            necessary to down_sample during data gen, because for each point of the pcd one pose will be computed.
            During inference it should not be down sampled. Defaults to True.
            object_pcd_record_path (str, optional): [description]. Defaults to ''.
        """
        if down_sample_pcd == True:
            rospy.logdebug(
                "Point cloud will be down sampled AND transformed to WORLD frame. This is not correct for testing grasp sampler!"
            )
        else:
            rospy.logdebug(
                "Point cloud will not be down sampled BUT transformed to OBJECT CENTROID frame, which is parallel to camera frame. This is necessary for testing grasp sampler."
            )
        self.object_pcd_record_path = object_pcd_record_path
        self.set_visual_data_save_paths(grasp_phase='pre')
        self.save_visual_data_client()
        self.segment_object_client(down_sample_pcd=down_sample_pcd)

    def set_path_and_save_visual_data(self,grasp_phase,object_pcd_record_path=''):
        """ only for data generation with multiple objects.
        grasp_phase: single, pre, during, post.
        single: only with target object in the scene, for generating mask.
        pre: all objects before grasping
        during: during the grasp execution
        post: after the grasp execution
        """
        self.object_pcd_record_path = object_pcd_record_path
        self.set_visual_data_save_paths(grasp_phase)
        self.save_visual_data_client()

    def filter_preshapes(self):
        total, no_ik, collision = self.filter_palm_goal_poses_client()

        self.prune_idxs = list(total)
        self.no_ik_idxs = list(no_ik)
        self.collision_idxs = list(collision)

        # Modify to choose what pose to prune
        self.top_idxs = [x for x in self.top_idxs if x not in self.prune_idxs]
        self.side1_idxs = [x for x in self.side1_idxs if x not in self.prune_idxs]
        self.side2_idxs = [x for x in self.side2_idxs if x not in self.prune_idxs]
        # self.top_idxs = [x for x in self.top_idxs if x in self.collision_idxs]
        # self.side1_idxs = [x for x in self.side1_idxs if x in self.collision_idxs]
        # self.side2_idxs = [x for x in self.side2_idxs if x in self.collision_idxs]


        if len(self.top_idxs) + len(self.side1_idxs) + len(self.side2_idxs) == 0:
            self.grasps_available = False

    def generate_hithand_preshape(self):
        """ Generate multiple grasp preshapes, which get stored in instance variable.
        """
        self.generate_hithand_preshape_client()

    def generate_valid_hithand_preshapes(self):
        """ First generates preshpes from the hithand preshape server and then prunes out all preshapes which are either in collision or have no IK solution.
        """
        self.generate_hithand_preshape_client()
        self.filter_preshapes()

    def get_valid_preshape_for_all_points(self):
        """ First generates preshpes from the hithand preshape server and then prunes out all preshapes which are either in collision or have no IK solution.
        """
        # Only record ones which
        self.get_preshape_for_all_points_client()
        self.filter_preshapes()
        if self.prune_idxs:
            self.record_collision_data_client()

    ### Functions to check object status ###
    def get_obstacle_objects_poses(self, obstacle_objects, threshold=0.01):
        """
        Args:
            obstacle_objects (list): _description_
            threshold (float, optional): Defaults to 0.01.

        Returns:
            obstacle_obj_poses (dict): {name_1: pose_1, name_2:pose2, ...}
        """
        obstacle_obj_poses = {}
        for idx, _ in enumerate(obstacle_objects):
            current_pose = self.get_grasp_object_pose_client(obj_name=obstacle_objects[idx]['name'])
            obstacle_obj_poses[obstacle_objects[idx]['name']] = current_pose
        return obstacle_obj_poses

    @staticmethod
    def check_if_object_moved(pose_1, pose_2, threshold=0.01):
        """Given two poses of certain object, this function tells if two poses are closer than certain
        threshold or not. In this way we can tell if certain object is static or being moved during grasping.
        """
        dist_x = abs(pose_1.position.x - pose_2.position.x)
        dist_y = abs(pose_1.position.y - pose_2.position.y)
        dist_z = abs(pose_1.position.z - pose_2.position.z)
        dist = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        if dist > threshold:
            return True
        else:
            return False

    def check_if_target_object_moved(self, previous_pose):
        current_pose = self.get_grasp_object_pose_client()
        self.check_if_object_moved(previous_pose, current_pose)

    def check_if_any_obstacle_object_moved(self, obstacle_obj_poses_1, obstacle_obj_poses_2):
        """True if any of object is moved. False if all objects are not moved.
        """
        for key in obstacle_obj_poses_1:
            is_moved = self.check_if_object_moved(obstacle_obj_poses_1[key],obstacle_obj_poses_2[key])
            if is_moved:
                return True
        return False

    @staticmethod
    def get_poses_distance(pose_1, pose_2):
        delta_x = abs(pose_1.pose.position.x - pose_2.pose.position.x)
        delta_y = abs(pose_1.pose.position.y - pose_2.pose.position.y)
        delta_z = abs(pose_1.pose.position.z - pose_2.pose.position.z)
        distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        return distance

    ################################################

    def grasp_and_lift_object(self, obstacle_objects):
        """ Used in data generation. For multi object generation.
        """
        # Record all object poses before grasp experiments
        target_obj_pose = self.get_grasp_object_pose_client()
        obstacle_obj_poses = self.get_obstacle_objects_poses(obstacle_objects)

        # Control the hithand to it's preshape
        i = 0
        # As long as there are viable poses
        if not self.grasps_available:
            rospy.logerr("No grasps are available")
            return False

        desired_plan_exists = False
        while self.grasps_available:
            # While loop to execute one feasible grasp, if one grasp pose is not feasible, go for next one, until no grasp pose left anymore
            i += 1

            # Step 1 choose a specific grasp. In first iteration self.chosen_grasp_type is unspecific, e.g. function will randomly choose grasp type
            self.choose_specific_grasp_preshape(grasp_type=self.chosen_grasp_type)

            if not self.grasps_available:
                break

            # Step 1.5 spawn hand to check collision
            palm_pose_world_arr = get_pose_array_from_stamped(self.palm_poses["desired_pre"])
            self.spawn_hand(palm_pose_world_arr)
            # Check if any object is being moved, if so, skip this experiment
            is_target_obj_moved = self.check_if_target_object_moved(target_obj_pose)
            obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
            are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
            raw_input('hand ok?')
            self.delete_hand()

            # Now once it failed once, we remove this grasp pose.
            if is_target_obj_moved or are_obstacle_obj_moved:
                rospy.logerr("target_object_moved: %s or obstacle_object_mmoved: %s" % (is_target_obj_moved, are_obstacle_obj_moved))
                self.remove_grasp_pose()


            # Step 2, if the previous grasp type is not same as current grasp type move to approach pose
            if self.previous_grasp_type != self.chosen_grasp_type or i == 1:
                approach_plan_exists = self.plan_arm_trajectory_client(self.palm_poses["approach"])
                # If a plan could be found, execute
                if approach_plan_exists:
                    self.execute_joint_trajectory_client(speed='mid')

            # Check if any object is being moved, if so, skip this experiment
            is_target_obj_moved = self.check_if_target_object_moved(target_obj_pose)
            obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
            are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
            # TODO: it's better for each grasp pose, try more times with diff. approach pose to avoid wired trajectory.
            # Now once it failed once, we remove this grasp pose.
            if is_target_obj_moved or are_obstacle_obj_moved:
                rospy.logerr("target_object_moved: %s or obstacle_object_mmoved: %s" % (is_target_obj_moved, are_obstacle_obj_moved))
                self.remove_grasp_pose()

            # Step 3, try to move to the desired palm position
            desired_plan_exists = self.plan_arm_trajectory_client()

            # Step 4 if a plan exists execute it, otherwise delete unsuccessful pose and start from top:
            if desired_plan_exists:
                self.execute_joint_trajectory_client(speed='mid')
                break
            else:
                self.remove_grasp_pose()

        # Check if any object is being moved
        is_target_obj_moved = self.check_if_target_object_moved(target_obj_pose)
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.grasp_pose_collide_target_object = 1 if is_target_obj_moved else 0
        self.grasp_pose_collide_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The grasp_pose_collide_target_object label: %s" % self.grasp_pose_collide_target_object)
        rospy.loginfo("The grasp_pose_collide_obstacle_objects label: %s" % self.grasp_pose_collide_obstacle_objects)

        # Get the current actual joint position and palm pose
        self.palm_poses["true_pre"], self.hand_joint_states[
            "true_pre"] = self.get_hand_palm_pose_and_joint_state()

        #letter = raw_input("Grasp object? Y/n: ")
        letter = 'y'
        if letter == 'y' or letter == 'Y':
            # Close the hand
            if desired_plan_exists:
                # Go into preshape
                self.control_hithand_config_client()
                # self.grasp_control_hithand_client()

        # Check if any obstacle obj being moved during finger close
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.close_finger_collide_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The close_finger_collide_obstacle_objects label: %s" % self.close_finger_collide_obstacle_objects)

        # Get the current actual joint position and palm pose
        self.palm_poses["closed"], self.hand_joint_states[
            "closed"] = self.get_hand_palm_pose_and_joint_state()

        # Check if robot reach the target grasp pose.
        pos_error = self.get_poses_distance(self.palm_poses["desired_pre"],self.palm_poses["true_pre"])
        if pos_error > 0.01:
            rospy.logerr("Cannot reach goal pose with error: %f m" % pos_error)
        else:
            rospy.logdebug("pos_error to target pose %f" % pos_error)

        # Save visual data after hand is closed
        self.save_only_depth_and_color(grasp_phase='during')

        # Lift the object
        if letter == 'y' or letter == 'Y':
            lift_pose = copy.deepcopy(self.palm_poses["desired_pre"])
            lift_pose.pose.position.z += self.object_lift_height
            start = time.time()
            execution_success = False
            while not execution_success:
                if time.time() - start > 60:
                    rospy.logerr('Could not find a lift pose')
                    break
                plan_exists = self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
                if plan_exists:
                    execution_success = self.execute_joint_trajectory_client(speed='slow')
                lift_pose.pose.position.x += np.random.uniform(-0.05, 0.05)
                lift_pose.pose.position.y += np.random.uniform(-0.05, 0.05)
                lift_pose.pose.position.z += np.random.uniform(0, 0.1)

        # Check if any obj moved during lift
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.lift_motion_moved_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The lift_motion_moved_obstacle_objects label: %s" % self.lift_motion_moved_obstacle_objects)

        # Get the joint position and palm pose after lifting
        self.palm_poses["lifted"], self.hand_joint_states[
            "lifted"] = self.get_hand_palm_pose_and_joint_state()

        # Evaluate success
        self.label_grasp()

        # Finally remove the executed grasp from the list
        self.remove_grasp_pose()

        return execution_success

    def grasp_from_inferred_pose(self, pose_obj_frame, joint_conf):
        """ Used in FFHNet evaluataion. Try to reach the pose and joint conf and attempt grasp given grasps from FFHNet.

        Args:
            pose_obj_frame (PoseStamped): 6D pose of the hand wrist in the object centroid vae frame.
            joint_conf (JointState): The desired joint position.
        """
        # transform the pose_obj_frame to world_frame
        palm_pose_world = self.transform_pose(pose_obj_frame, 'object_centroid_vae', 'world')

        object_pose = self.get_grasp_object_pose_client()

        # save the desired pre of the palm and joints (given to function call) in an ins
        # tance variable
        self.palm_poses['desired_pre'] = palm_pose_world
        self.hand_joint_states['desired_pre'] = joint_conf

        # Update the palm pose for visualization in RVIZ
        self.update_grasp_palm_pose_client(palm_pose_world)

        # Compute an approach pose and try to reach it. Add object mesh to moveit to avoid hitting it with the approach plan. Delete it after
        if not self.is_eval_sess:
            self.create_moveit_scene_client()
        approach_pose = self.approach_pose_from_palm_pose(palm_pose_world)
        approach_plan_exists = self.plan_arm_trajectory_client(approach_pose)
        if not approach_plan_exists:
            count = 0
            while not approach_plan_exists and count < 3:
                approach_pose = self.add_position_noise(approach_pose)
                approach_plan_exists = self.plan_arm_trajectory_client(approach_pose)
                count += 1

        if approach_plan_exists:
            if not self.is_eval_sess:
                self.clean_moveit_scene_client()
        else:
            self.grasp_label = -1
            rospy.logerr("no traj found to approach pose")
            return False
        # Execute to approach pose
        self.execute_joint_trajectory_client(speed='mid')

        # Detect object pose to check if collision happened
        if self.check_if_target_object_moved(object_pose):
            self.collision_to_approach_pose = 1
            rospy.logerr("Object moved during way to approach pose")
        else:
            self.collision_to_approach_pose = 0

        # Try to find a plan to the final destination
        plan_exists = self.plan_arm_trajectory_client(palm_pose_world)

        # TODO: Move L Motion not working
        # self.plan_cartesian_path_trajectory_client(palm_pose_world)

        # Backup if no plan found
        if not plan_exists:
            plan_exists = self.plan_arm_trajectory_client(palm_pose_world)
            if not plan_exists:
                self.grasp_label = -1
                rospy.logerr("no traj found to final grasp pose")
                return False

        # Execute joint trajectory
        self.execute_joint_trajectory_client(speed='mid')

        # Detect object pose to check if collision happened
        if self.check_if_target_object_moved(object_pose):
            self.collision_to_grasp_pose = 1
            rospy.logerr("Object moved during way to final pose")
        else:
            self.collision_to_grasp_pose = 0

        # Get the current actual joint position and palm pose
        self.palm_poses["true_pre"], self.hand_joint_states[
            "true_pre"] = self.get_hand_palm_pose_and_joint_state()

        # Check if robot reach the target grasp pose.
        pos_error = self.get_poses_distance(self.palm_poses["desired_pre"],self.palm_poses["true_pre"])
        if pos_error > 0.01:
            rospy.logerr("Cannot reach goal pose with error: %f m" % pos_error)
        else:
            rospy.logdebug("pos_error to target pose %f" % pos_error)

        # Go into the joint conf:
        self.control_hithand_config_client(joint_conf=joint_conf)

        # Possibly apply some more control to apply more force

        # Get the current actual joint position and palm pose
        self.palm_poses["closed"], self.hand_joint_states[
            "closed"] = self.get_hand_palm_pose_and_joint_state()

        # Plan lift trajectory client
        lift_pose = copy.deepcopy(self.palm_poses["desired_pre"])
        lift_pose.pose.position.z += self.object_lift_height
        start = time.time()
        execution_success = False
        while not execution_success:
            if time.time() - start > 60:
                rospy.logerr('Could not find a lift pose')
                break
            plan_exists = self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
            if plan_exists:
                execution_success = self.execute_joint_trajectory_client(speed='slow')
            lift_pose.pose.position.x += np.random.uniform(-0.05, 0.05)
            lift_pose.pose.position.y += np.random.uniform(-0.05, 0.05)
            lift_pose.pose.position.z += np.random.uniform(0, 0.1)

        # Get the joint position and palm pose after lifting
        self.palm_poses["lifted"], self.hand_joint_states[
            "lifted"] = self.get_hand_palm_pose_and_joint_state()

        # Evaluate success
        self.label_grasp()

        # raw_input('Continue?')
        return True

    def grasp_from_inferred_pose_multi_obj(self, pose_obj_frame, joint_conf, obstacle_objects):
        """ Used in FFHNet evaluataion. Try to reach the pose and joint conf and attempt grasp given grasps from FFHNet.

        Args:
            pose_obj_frame (PoseStamped): 6D pose of the hand wrist in the object centroid vae frame.
            joint_conf (JointState): The desired joint position.
        """
        # transform the pose_obj_frame to world_frame
        palm_pose_world = self.transform_pose(pose_obj_frame, 'object_centroid_vae', 'world')

        target_obj_pose = self.get_grasp_object_pose_client()
        obstacle_obj_poses = self.get_obstacle_objects_poses(obstacle_objects)

        # save the desired pre of the palm and joints (given to function call) in an ins
        # tance variable
        self.palm_poses['desired_pre'] = palm_pose_world
        self.hand_joint_states['desired_pre'] = joint_conf

        # Update the palm pose for visualization in RVIZ
        self.update_grasp_palm_pose_client(palm_pose_world)

        # Compute an approach pose and try to reach it. Add object mesh to moveit to avoid hitting it with the approach plan. Delete it after
        if not self.is_eval_sess:
            self.create_moveit_scene_client()
        approach_pose = self.approach_pose_from_palm_pose(palm_pose_world)
        approach_plan_exists = self.plan_arm_trajectory_client(approach_pose)
        if not approach_plan_exists:
            count = 0
            while not approach_plan_exists and count < 3:
                approach_pose = self.add_position_noise(approach_pose)
                approach_plan_exists = self.plan_arm_trajectory_client(approach_pose)
                count += 1

        if approach_plan_exists:
            if not self.is_eval_sess:
                self.clean_moveit_scene_client()
        else:
            self.grasp_label = -1
            rospy.logerr("no traj found to approach pose")
            return False
        # Execute to approach pose
        self.execute_joint_trajectory_client(speed='mid')

        # Check if any object is being moved, if so, skip this experiment
        is_target_obj_moved = self.check_if_target_object_moved(target_obj_pose)
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        # TODO: it's better for each grasp pose, try more times with diff. approach pose to avoid wired trajectory.
        # Now once it failed once, we remove this grasp pose.
        if is_target_obj_moved or are_obstacle_obj_moved:
            rospy.logerr("Way to approach pose, target_object_moved: %s or obstacle_object_mmoved: %s" % (is_target_obj_moved, are_obstacle_obj_moved))
            return False

        # Try to find a plan to the final destination
        plan_exists = self.plan_arm_trajectory_client(palm_pose_world)

        # TODO: Move L Motion not working
        # self.plan_cartesian_path_trajectory_client(palm_pose_world)

        # Backup if no plan found
        if not plan_exists:
            plan_exists = self.plan_arm_trajectory_client(palm_pose_world)
            if not plan_exists:
                self.grasp_label = -1
                rospy.logerr("no traj found to final grasp pose")
                return False

        # Execute joint trajectory
        self.execute_joint_trajectory_client(speed='mid')

        # Check if any object is being moved
        is_target_obj_moved = self.check_if_target_object_moved(target_obj_pose)
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.grasp_pose_collide_target_object = 1 if is_target_obj_moved else 0
        self.grasp_pose_collide_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The grasp_pose_collide_target_object label: %s" % self.grasp_pose_collide_target_object)
        rospy.loginfo("The grasp_pose_collide_obstacle_objects label: %s" % self.grasp_pose_collide_obstacle_objects)

        # Get the current actual joint position and palm pose
        self.palm_poses["true_pre"], self.hand_joint_states[
            "true_pre"] = self.get_hand_palm_pose_and_joint_state()

        # Check if robot reach the target grasp pose.
        pos_error = self.get_poses_distance(self.palm_poses["desired_pre"],self.palm_poses["true_pre"])
        if pos_error > 0.01:
            rospy.logerr("Cannot reach goal pose with error: %f m" % pos_error)
        else:
            rospy.logdebug("pos_error to target pose %f" % pos_error)

        # Go into the joint conf:
        self.control_hithand_config_client(joint_conf=joint_conf)

        # Check if any obstacle obj being moved during finger close
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.close_finger_collide_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The close_finger_collide_obstacle_objects label: %s" % self.close_finger_collide_obstacle_objects)

        # Possibly apply some more control to apply more force

        # Get the current actual joint position and palm pose
        self.palm_poses["closed"], self.hand_joint_states[
            "closed"] = self.get_hand_palm_pose_and_joint_state()

        # Plan lift trajectory client
        lift_pose = copy.deepcopy(self.palm_poses["desired_pre"])
        lift_pose.pose.position.z += self.object_lift_height
        start = time.time()
        execution_success = False
        while not execution_success:
            if time.time() - start > 60:
                rospy.logerr('Could not find a lift pose')
                break
            plan_exists = self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
            if plan_exists:
                execution_success = self.execute_joint_trajectory_client(speed='slow')
            lift_pose.pose.position.x += np.random.uniform(-0.05, 0.05)
            lift_pose.pose.position.y += np.random.uniform(-0.05, 0.05)
            lift_pose.pose.position.z += np.random.uniform(0, 0.1)

        # Check if any obj moved during lift
        obstacle_obj_poses_tmp = self.get_obstacle_objects_poses(obstacle_objects)
        are_obstacle_obj_moved = self.check_if_any_obstacle_object_moved(obstacle_obj_poses,obstacle_obj_poses_tmp)
        self.lift_motion_moved_obstacle_objects = 1 if are_obstacle_obj_moved else 0
        rospy.loginfo("The lift_motion_moved_obstacle_objects label: %s" % self.lift_motion_moved_obstacle_objects)

        # Get the joint position and palm pose after lifting
        self.palm_poses["lifted"], self.hand_joint_states[
            "lifted"] = self.get_hand_palm_pose_and_joint_state()

        # Evaluate success
        self.label_grasp()

        # raw_input('Continue?')
        return True
#####################################################
## below are codes for multiple objects generation ##
#####################################################

def _select_ROI(image, close_window=True):
    while True:
        cv2.namedWindow("Seg", cv2.WND_PROP_FULLSCREEN)
        try:
            roi = cv2.selectROI('Seg', image, False, False)
        except:
            roi = [0]

        if not any(roi):
            print("No area selected. Press 'c' to abort or anything else to reselect")
            if cv2.waitKey(0) == ord('c'):
                exit()
        else:
            # user selected something
            break
    if close_window:
        cv2.destroyWindow("Seg")
    return roi


def _project_point_in_world_onto_image_plane(x, y, z, camera_intrinsics):
    intrinsic_matrix = camera_intrinsics.intrinsic_matrix
    camera_T_world = _get_world_to_camera_transformation()
    P = np.matmul(intrinsic_matrix, camera_T_world[:3,:])
    point_coordinate = np.matmul(P, np.array([x, y, z, 1]).reshape(-1, 1))
    x = int(point_coordinate[0] / point_coordinate[2])
    y = int(point_coordinate[1] / point_coordinate[2])

    return x, y

def _is_point_inside_ROI(ROI, x, y):
    return (ROI[0] < x < ROI[0] + ROI[2]) and (ROI[1] < y < ROI[1] + ROI[3])

def _get_camera_intrinsics():
    image_width = 1280
    image_height = 720
    horizontal_fov = math.radians(64)
    fx = 0.5 * image_width / math.tan(0.5 * horizontal_fov)
    fy = fx
    cx = image_width * 0.5
    cy = image_height * 0.5

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    return pinhole_camera_intrinsic

def _get_camera_to_world_transformation():
    global world_T_camera_buffer
    if world_T_camera_buffer is not None:
        return world_T_camera_buffer
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    scene_pcd_topic = rospy.get_param('scene_pcd_topic', default='/camera/depth/points')
    # as stated in grasp-pipeline/launch/grasp_pipeline_servers_real.launch, the pcd_topic for
    # realsense is either /camera/depth/points from simulation or the other one in real world
    if scene_pcd_topic == '/camera/depth/points' or scene_pcd_topic == '/camera/depth/color/points':
        pcd_frame = 'camera_depth_optical_frame'
    elif scene_pcd_topic == '/depth_registered_points':
        pcd_frame = 'camera_color_optical_frame'
    else:
        rospy.logerr(
            'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

    transform_camera_world = tf_buffer.lookup_transform(
        'world', pcd_frame, rospy.Time())
    q = transform_camera_world.transform.rotation
    r = transform_camera_world.transform.translation
    world_T_camera = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    world_T_camera[:, 3] = [r.x, r.y, r.z, 1]
    world_T_camera_buffer = world_T_camera
    return world_T_camera

def _get_world_to_camera_transformation():
    global camera_T_world_buffer
    if camera_T_world_buffer is not None:
        return camera_T_world_buffer
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    scene_pcd_topic = rospy.get_param('scene_pcd_topic', default='/camera/depth/points')
    # as stated in grasp-pipeline/launch/grasp_pipeline_servers_real.launch, the pcd_topic for
    # realsense is either /camera/depth/points from simulation or the other one in real world
    if scene_pcd_topic == '/camera/depth/points' or scene_pcd_topic == '/camera/depth/color/points':
        pcd_frame = 'camera_depth_optical_frame'
    elif scene_pcd_topic == '/depth_registered_points':
        pcd_frame = 'camera_color_optical_frame'
    else:
        rospy.logerr(
            'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

    transform_world_camera = tf_buffer.lookup_transform(
        pcd_frame, 'world', rospy.Time())
    q = transform_world_camera.transform.rotation
    r = transform_world_camera.transform.translation
    camera_T_world = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    camera_T_world[:, 3] = [r.x, r.y, r.z, 1]
    camera_T_world_buffer = camera_T_world
    return camera_T_world
