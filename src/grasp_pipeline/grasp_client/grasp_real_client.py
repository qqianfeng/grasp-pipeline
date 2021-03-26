#!/usr/bin/env python
import copy
import datetime
from multiprocessing import Process
import numpy as np
import open3d as o3d
import os
import rospy
import sys
import tf
import tf.transformations as tft
import tf2_geometry_msgs
import tf2_ros
import time

sys.path.append('..')
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, SetBoolRequest

from grasp_pipeline.utils.utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped, plot_voxel
from grasp_pipeline.utils import utils
from grasp_pipeline.utils.align_object_frame import align_object
from grasp_pipeline.srv import *


class GraspClient():
    """ This class is a wrapper around all the individual functionality involved in grasping experiments.
    """
    def __init__(self, is_rec_sess, grasp_data_recording_path=''):
        rospy.init_node('grasp_client')
        self.grasp_data_recording_path = grasp_data_recording_path
        if grasp_data_recording_path is not '':
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
        self.base_path = '/home/vm'
        self.scene_pcd_save_path = os.path.join(self.base_path, 'scene.pcd')
        self.object_pcd_save_path = os.path.join(self.base_path, 'object.pcd')
        self.bps_object_path = os.path.join(self.base_path, 'pcd_enc.npy')
        self.object_pcd_record_path = ''

        self.heuristic_preshapes = None  # This variable stores all the information on multiple heuristically sampled grasping pre shapes
        # The chosen variables store one specific preshape (palm_pose, hithand_joint_state, is_top_grasp)
        self.chosen_is_top_grasp = None

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

        # For voxel server
        self.voxel_grid_dim = np.array([26, 26, 26])
        self.voxel_grid_dim_full = np.array([32, 32, 32])
        self.voxel_translation_dim = (self.voxel_grid_dim_full - self.voxel_grid_dim) // 2

        self.is_rec_sess = is_rec_sess

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
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

    def update_object_metadata(self, object_metadata):
        """ Update the metainformation about the object to be grasped
        """
        self.grasp_id_num = 0
        self.object_metadata = object_metadata

    def _setup_workspace_boundaries(self):
        """ Sets the boundaries in which an object can be spawned and placed.
        Gets called 
        """
        self.spawn_object_x_min, self.spawn_object_x_max = 0.25, 0.65
        self.spawn_object_y_min, self.spawn_object_y_max = -0.2, 0.2

    def show_grasps_o3d_viewer(self, palm_poses):
        """Visualize the sampled grasp poses in open3d along with the object point cloud.

        Args:
            palm_poses (list of PoseStamped): Sampled poses.
        """
        frames = []
        for i in range(0, len(palm_poses)):
            palm_hom = utils.hom_matrix_from_pose_stamped(palm_poses[i])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01).transform(palm_hom)
            frames.append(frame)

        #visualize
        orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
        frames.append(orig)

        obj = o3d.io.read_point_cloud(self.object_pcd_save_path)
        frames.append(obj)
        o3d.visualization.draw_geometries(frames)

    # ++++++++ PART II: Second part consist of all clients that interact with different nodes/services ++++++++++++
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
            rospy.loginfo('Service control_hithand_config call failed: %s' % e)
        rospy.loginfo('Service control_allegro_config is executed.')

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
            rospy.loginfo('Service check_pose_validity_utah call failed: %s' % e)
        rospy.loginfo('Service check_pose_validity_utah is executed.')
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
            rospy.loginfo('Service encode_pcd_with_bps call failed: %s' % e)
        rospy.loginfo('Service encode_pcd_with_bps is executed.')

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
                rospy.loginfo('The joint trajectory in planned_panda_joint_trajectory was empty.')
        except rospy.ServiceException, e:
            rospy.loginfo('Service execute_joint_trajectory call failed: %s' % e)
        rospy.loginfo('Service execute_joint_trajectory is executed.')

    def filter_palm_goal_poses_client(self):
        wait_for_service('filter_palm_goal_poses')
        try:
            filter_palm_goal_poses = rospy.ServiceProxy('filter_palm_goal_poses', FilterPalmPoses)
            req = FilterPalmPosesRequest()
            req.palm_goal_poses_world = self.heuristic_preshapes.palm_goal_poses_world

            res = filter_palm_goal_poses(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service filter_palm_goal_poses call failed: %s' % e)
        rospy.loginfo('Service filter_palm_goal_poses is executed.')
        return res.prune_idxs

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
            rospy.loginfo('Service infer_grasp_poses call fialed: %s' % e)
        rospy.loginfo('Service infer_grasp_poses is executed')
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
            rospy.loginfo('Service plan_arm_trajectory call failed: %s' % e)
        rospy.loginfo('Service plan_arm_trajectory is executed.')
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
        return res.success

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
            rospy.loginfo('Service record_grasp_trial_data call failed: %s' % e)
        rospy.loginfo('Service record_grasp_trial_data is executed.')

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
            rospy.loginfo('Service record_grasp_data call failed: %s' % e)
        rospy.loginfo('Service record_grasp_data is executed.')

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
            rospy.loginfo('Service save_visual_data call failed: %s' % e)
        rospy.loginfo('Service save_visual_data is executed %s' % res.success)

    def segment_object_client(self, align_object_world=True, down_sample_pcd=True):
        wait_for_service('segment_object')
        try:
            segment_object = rospy.ServiceProxy('segment_object', SegmentGraspObject)
            req = SegmentGraspObjectRequest()
            req.down_sample_pcd = down_sample_pcd
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
                self.object_metadata["aligned_dim_whd_utah"] = [
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ]
                print("whd from alignment: ")
                print(self.object_metadata["aligned_dim_whd_utah"])
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
                print("whd Vincent: ")
                print([
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ])

        except rospy.ServiceException, e:
            rospy.loginfo('Service segment_object call failed: %s' % e)
        rospy.loginfo('Service segment_object is executed.')

    def update_grasp_palm_pose_client(self, palm_pose):
        wait_for_service("update_grasp_palm_pose")
        try:
            update_grasp_palm_pose = rospy.ServiceProxy('update_grasp_palm_pose', UpdatePalmPose)
            req = UpdatePalmPoseRequest()
            req.palm_pose = palm_pose
            res = update_grasp_palm_pose(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_grasp_palm_pose call failed: %s' % e)
        rospy.loginfo('Service update_grasp_palm_pose is executed.')

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

    def update_object_mesh_frame_pose_client(self):
        wait_for_service("update_object_mesh_frame_pose")
        try:
            update_object_mesh_frame_pose = rospy.ServiceProxy('update_object_mesh_frame_pose',
                                                               UpdateObjectPose)
            req = UpdateObjectPoseRequest()
            req.object_pose_world = self.object_metadata["mesh_frame_pose"]
            res = update_object_mesh_frame_pose(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_object_mesh_frame_pose call failed: %s' % e)
        rospy.loginfo('Service update_object_mesh_frame_pose is executed.')

    def visualize_grasp_pose_list_client(self, grasp_poses):
        wait_for_service("visualize_grasp_pose_list")
        try:
            visualize_grasp_pose_list = rospy.ServiceProxy('visualize_grasp_pose_list',
                                                           VisualizeGraspPoseList)
            req = VisualizeGraspPoseListRequest()
            req.grasp_pose_list = grasp_poses
            res = visualize_grasp_pose_list(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service visualize_grasp_poses_list failed: %s' % e)
        rospy.loginfo('Service visualize_grasp_poses_list is executed.')

    # =============================================================================================================
    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def check_pose_validity_utah(self, grasp_pose):
        return self.check_pose_validity_utah_client(grasp_pose)

    def encode_pcd_with_bps(self):
        self.encode_pcd_with_bps_client()

    def infer_grasp_poses(self, n_poses, visualize_poses=False, bps_object=None):
        if bps_object == None:
            bps_object = np.load(self.bps_object_path)
        palm_poses, joint_confs = self.infer_grasp_poses_client(n_poses=n_poses,
                                                                bps_object=bps_object)
        if visualize_poses:
            self.visualize_grasp_pose_list_client(palm_poses)
        return palm_poses, joint_confs

    def reset_hithand_and_panda(self):
        """ Reset panda and hithand to their home positions
        """
        self.reset_hithand_joints_client()
        reset_plan_exists = self.plan_reset_trajectory_client()
        if reset_plan_exists:
            self.execute_joint_trajectory_client()

    def set_visual_data_save_paths(self, grasp_phase):
        if self.is_rec_sess:
            if grasp_phase not in ['pre', 'during', 'post']:
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

    def save_visual_data_and_record_grasp(self):
        self.set_visual_data_save_paths(grasp_phase='post')
        self.save_visual_data_client(save_pcd=False)
        #self.generate_voxel_from_pcd_client()
        self.record_grasp_trial_data_client()

    def save_only_depth_and_color(self, grasp_phase):
        """ Saves only depth and color by setting scene_pcd_save_path to None. Resets scene_pcd_save_path afterwards.
        """
        self.set_visual_data_save_paths(grasp_phase=grasp_phase)
        self.save_visual_data_client(save_pcd=False)

    def save_visual_data_and_segment_object(self, down_sample_pcd=True, object_pcd_record_path=''):
        if down_sample_pcd == True:
            print(
                "Point cloud will be down_sampled AND transformed to WORLD frame. This is not correct for testing grasp sampler!"
            )
        else:
            print(
                "Point cloud will not be down sampled BUT transformed to OBJECT CENTROID frame, which is parallel to camera frame. This is necessary for testing grasp sampler."
            )
        #self.object_pcd_record_path = object_pcd_record_path
        #self.set_visual_data_save_paths(grasp_phase='pre')
        #self.save_visual_data_client()
        self.segment_object_client(down_sample_pcd=down_sample_pcd)

    def filter_preshapes(self):
        self.prune_idxs = list(self.filter_palm_goal_poses_client())

        print(self.prune_idxs)

        self.top_idxs = [x for x in self.top_idxs if x not in self.prune_idxs]
        self.side1_idxs = [x for x in self.side1_idxs if x not in self.prune_idxs]
        self.side2_idxs = [x for x in self.side2_idxs if x not in self.prune_idxs]

        if len(self.top_idxs) + len(self.side1_idxs) + len(self.side2_idxs) == 0:
            self.grasps_available = False

    def grasp_from_inferred_pose(self, pose_obj_frame, joint_conf):
        """Try to reach the pose and joint conf and attempt grasp.

        Args:
            pose_obj_frame (PoseStamped): 6D pose of the hand wrist in the object centroid vae frame.
            joint_conf (JointState): The desired joint position.
        """
        # transform the pose_obj_frame to world_frame
        palm_pose_world = self.transform_pose(pose_obj_frame, 'object_centroid_vae', 'world')

        # Update the palm pose for visualization in RVIZ
        self.update_grasp_palm_pose_client(palm_pose_world)

        return

        # Try to find a plan
        plan_exists = self.plan_arm_trajectory_client(palm_pose_world)

        # Backup if no plan found
        if not plan_exists:
            plan_exists = self.plan_arm_trajectory_client(palm_pose_world)
            if not plan_exists:
                return False

        # Execute joint trajectory
        self.execute_joint_trajectory_client(speed='mid')

        # Go into the joint conf:
        self.control_hithand_config_client(joint_conf=joint_conf)

        # Possibly apply some more control to apply more force

        # Plan lift trajectory client
        lift_pose = copy.deepcopy(palm_pose_world)
        lift_pose.pose.position.z += self.object_lift_height
        self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
