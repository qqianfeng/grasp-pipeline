#!/usr/bin/env python
import rospy
import datetime
from geometry_msgs.msg import PoseStamped
import tf
import tf.transformations as tft
from grasp_pipeline.srv import *
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import sys
sys.path.append('..')
from utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped
from align_object_frame import align_object
import numpy as np
from std_srvs.srv import SetBool, SetBoolRequest
import os
from multiprocessing import Process


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

        self.depth_img = None
        self.color_img = None
        self.pcd = None

        # These variables get changed dynamically during execution to store relevant data under correct folder
        self.color_img_save_path = None
        self.depth_img_save_path = None
        self.scene_pcd_save_path = '/home/vm/scene.pcd'
        self.object_pcd_save_path = '/home/vm/object.pcd'
        self.object_pcd_record_path = ''

        self.segmented_object_pcd = None
        self.segmented_object_width = None
        self.segmented_object_height = None
        self.segmented_object_depth = None

        self.heuristic_preshapes = None  # This variable stores all the information on multiple heuristically sampled grasping pre shapes
        # The chosen variables store one specific preshape (palm_pose, hithand_joint_state, is_top_grasp)
        self.chosen_palm_pose = None
        self.chosen_hithand_joint_state = None
        self.chosen_is_top_grasp = None

        self.panda_planned_joint_trajectory = None
        self.smooth_trajectories = True
        self.num_of_replanning_attempts = 5
        self.num_poses = 50

        self.object_lift_height = 0.2  # Lift the object 15 cm
        self.success_tolerance_lift_height = 0.05
        self.object_segment_response = None
        self.grasp_label = None

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
        self.spawn_object_x_min, self.spawn_object_x_max = 0.5, 0.8
        self.spawn_object_y_min, self.spawn_object_y_max = -0.3, 0.3
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
        """
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
        print('Chosen grasp_idx is: ' + str(grasp_idx))
        print('Chosen palm pose is: ')
        self.chosen_palm_pose = self.heuristic_preshapes.palm_goal_pose_world[grasp_idx]
        self.chosen_grasp_idx = grasp_idx
        print(self.chosen_palm_pose)
        self.chosen_hithand_joint_state = self.heuristic_preshapes.hithand_joint_state[grasp_idx]
        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]

        # Add selected pose and joint state as desired palm pose and joint state to dict
        self.palm_poses["desired_pre"] = self.chosen_palm_pose
        self.hand_joint_states["desired_pre"] = self.chosen_hithand_joint_state

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
                req.hithand_target_joint_state = self.chosen_hithand_joint_state
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
            else:
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
        except rospy.ServiceException, e:
            rospy.loginfo('Service generate_hithand_preshape call failed: %s' % e)
        rospy.loginfo('Service generate_hithand_preshape is executed.')

    def get_hand_palm_pose_and_joint_state(self):
        """ Returns a list with
            1. first 3 entries being the palm position and last 4 the orientation quaternion.
            2. hithand joint state
        """
        (palm_trans, palm_rot) = self.tf_listener.lookupTransform('world', 'palm_link_hithand',
                                                                  rospy.Time())
        palm_pose_list = [
            palm_trans[0], palm_trans[1], palm_trans[2], palm_rot[0], palm_rot[1], palm_rot[2],
            palm_rot[3]
        ]
        joint_state = rospy.wait_for_message("/hithand/joint_states", JointState, timeout=5)
        return palm_pose_list, joint_state

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
                req.palm_goal_pose_world = self.chosen_palm_pose
            res = moveit_cartesian_pose_planner(req)
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
        raise NotImplementedError  # NEED TO SAVE DESIRED PALM POSE IN OBJECT ALIGNED FRAME
        try:
            record_grasp_data = rospy.ServiceProxy('record_grasp_data', RecordGraspDataSim)
            res = record_grasp_data(RecordGraspDataSimRequest())
            res.object_name = self.object_metadata["name"]
            res.time_stamp = datetime.datetime.now().isoformat()
            res.is_top_grasp = self.chosen_is_top_grasp
            res.grasp_success_label = self.grasp_label
            res.object_size = self.object_metadata["aligned_dim_whd"]
            res.oibject_size_aligned = self.object_metadata["seg_dim_whd"]
            res.sparse_voxel_grid = EOFError  # NOT IMPLEMENTED
            res.object_world_sim_pose = self.object_metadata["sim_pose"]
            res.object_world_seg_pose = self.object_metadata["seg_pose"]
            res.object_world_aligned_pose = self.object_metadata["aligned_pose"]
            res.desired_preshape_palm_world_pose = self.palm_poses["desired_pre"]
            res.true_preshape_palm_world_pose = self.palm_poses["true_pre"]
            res.closed_preshape_palm_world_pose = self.palm_poses["closed"]
            res.lifted_preshape_palm_world_pose = self.palm_poses["lifted"]
            res.desired_preshape_hithand_joint_state = self.hand_joint_states["desired_pre"]
            res.true_preshape_hithand_joint_state = self.hand_joint_states["true_pre"]
            res.closed_hithand_joint_state = self.hand_joint_states["closed"]
            res.lifted_hithand_joint_state = self.hand_joint_states["lifted"]
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
                self.object_segment_response.object = align_object(
                    self.object_segment_response.object, self.tf_listener)
                self.object_metadata["aligned_pose"] = PoseStamped(
                    header=Header(frame_id='world'), pose=self.object_segment_response.object.pose)
                self.object_metadata["aligned_dim_whd"] = [
                    self.object_segment_response.object.width,
                    self.object_segment_response.object.height,
                    self.object_segment_response.object.depth
                ]

                self.update_object_pose_aligned_client()
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
        del self.heuristic_preshapes.palm_goal_pose_world[self.chosen_grasp_idx]
        del self.heuristic_preshapes.hithand_joint_state[self.chosen_grasp_idx]
        del self.heuristic_preshapes.is_top_grasp[self.chosen_grasp_idx]

    def reset_hithand_and_panda(self):
        """ Reset panda and hithand to their home positions
        """
        self.plan_reset_trajectory_client()
        # Parallel execute these:
        self.parallel_execute_functions([
            self.execute_joint_trajectory_client, self.reset_hithand_joints_client
        ])

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
        rospy.sleep(2)
        object_pose = self.get_grasp_object_pose_client()
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
        self.save_visual_data_client()
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

        # Generate a robot trajectory to move to desired pose
        moveit_plan_exists = self.plan_arm_trajectory_client()
        for j in range(self.num_poses):
            if moveit_plan_exists:
                break
            self.remove_unreachable_pose()
            if len(self.heuristic_preshapes.palm_goal_pose_world) == 0:
                rospy.loginfo(
                    'All poses have been removed, because they were either executed or are not feasible'
                )
                return
            self.choose_specific_grasp_preshape(grasp_type='unspecified')
            if not moveit_plan_exists:
                for i in range(self.num_of_replanning_attempts):
                    moveit_plan_exists = self.plan_arm_trajectory_client()
                    if moveit_plan_exists:
                        rospy.loginfo("Found valid moveit plan after " + str(i + 1) + " tries")
                        break
        # Possibly trajectory/pose needs an extra validity check or smth like that
        if not moveit_plan_exists:
            rospy.loginfo("No moveit plan could be found")
            return
        # Execute the generated joint trajectory
        self.execute_joint_trajectory_client(smoothen_trajectory=self.smooth_trajectories)

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
        lift_pose = self.chosen_palm_pose
        lift_pose.pose.position.z += self.object_lift_height
        self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
        self.execute_joint_trajectory_client(smoothen_trajectory=True)

        # Get the joint position and palm pose after lifting
        self.palm_poses["lifted"], self.hand_joint_states[
            "lifted"] = self.get_hand_palm_pose_and_joint_state()

        # Evaluate success
        self.label_grasp()