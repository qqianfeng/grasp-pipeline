#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState
import sys
sys.path.append('..')
from utils import wait_for_service, get_pose_stamped_from_array, get_pose_array_from_stamped
import numpy as np
from std_srvs.srv import SetBool, SetBoolRequest


class GraspClient():
    """ This class is a wrapper around all the individual functionality involved in grasping experiments.
    """
    def __init__(self):
        rospy.init_node('grasp_client')
        self.object_datasets_folder = rospy.get_param('object_datasets_folder')
        self.color_img_save_path = rospy.get_param('color_img_save_path')
        self.depth_img_save_path = rospy.get_param('depth_img_save_path')
        self.object_pcd_path = rospy.get_param('object_pcd_path')
        self.scene_pcd_path = rospy.get_param('scene_pcd_path')

        # Save metainformation on object to be grasped in these vars
        self.object_name = None
        self.object_mesh_path = None
        self.object_pose_stamped = None
        self._setup_workspace_boundaries()

        self.depth_img = None
        self.color_img = None
        self.pcd = None

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

        self.object_lift_height = 0.15  # Lift the object 15 cm

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
    def update_object_metadata(self, object_metadata):
        """ Update the metainformation about the object to be grasped
        """
        raise NotImplementedError

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
        object_pose = [0, 0, rand_z_orientation, rand_x, rand_y, self.table_height]
        rospy.loginfo('Generated random object pose:')
        rospy.loginfo(object_pose)
        object_pose_stamped = get_pose_stamped_from_array(object_pose)
        self.object_pose_stamped = object_pose_stamped

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
        self.chosen_palm_pose = self.heuristic_preshapes.palm_goal_pose_world[grasp_idx]
        self.chosen_hithand_joint_state = self.heuristic_preshapes.hithand_joint_state[grasp_idx]
        self.chosen_is_top_grasp = self.heuristic_preshapes.is_top_grasp[grasp_idx]

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
            execute_joint_trajectory = rospy.ServiceProxy('execute_joint_trajectory',
                                                          ExecuteJointTrajectory)
            req = ExecuteJointTrajectoryRequest()
            req.smoothen_trajectory = smoothen_trajectory
            req.joint_trajectory = self.panda_planned_joint_trajectory
            res = execute_joint_trajectory(req)
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
            req.sample = True
            self.heuristic_preshapes = generate_hithand_preshape(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service generate_hithand_preshape call failed: %s' % e)
        rospy.loginfo('Service generate_hithand_preshape is executed.')

    def grasp_control_hithand_client(self):
        """ Call server to close hithand fingers and stop when joint velocities are close to zero.
        """
        wait_for_service('gras_control_hithand')
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
        self.panda_planned_joint_trajectory = res.plan_traj
        return res.success

    def plan_reset_trajectory_client(self):
        wait_for_service('plan_reset_trajectory')
        try:
            plan_reset_trajectory = rospy.ServiceProxy('plan_reset_trajectory',
                                                       PlanResetTrajectory)
            res = plan_reset_trajectory(PlanResetTrajectoryRequest())
            self.panda_planned_joint_trajectory = res.plan_traj
        except rospy.ServiceException, e:
            rospy.loginfo('Service plan_reset_trajectory call failed: %s' % e)
        rospy.loginfo('Service plan_reset_trajectory is executed.')

    def record_grasp_data_client(self)
        wait_for_service('record_grasp_data')
        try:
            record_grasp_data = rospy.ServiceProxy('record_grasp_data',
                                                      RecordGraspDataSim)
            res = record_grasp_data(RecordGraspDataSimRequest())
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
            rospy.loginfo('Service save_visual_data call failed: %s' % e)
        rospy.loginfo('Service save_visual_data is executed.')

    def save_visual_data_client(self):
        wait_for_service('save_visual_data')
        try:
            save_visual_data = rospy.ServiceProxy('save_visual_data', SaveVisualData)
            req = SaveVisualDataRequest()
            req.color_img = self.color_img
            req.depth_img = self.depth_img
            req.scene_pcd = self.pcd
            req.color_img_save_path = self.color_img_save_path
            req.depth_img_save_path = self.depth_img_save_path
            req.scene_pcd_save_path = self.scene_pcd_path
            res = save_visual_data(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service save_visual_data call failed: %s' % e)
        rospy.loginfo('Service save_visual_data is executed.')

    def segment_object_client(self):
        wait_for_service('segment_object')
        try:
            segment_object = rospy.ServiceProxy('segment_object', SegmentGraspObject)
            req = SegmentGraspObjectRequest()
            req.scene_pcd_path = self.scene_pcd_path
            req.object_pcd_path = self.object_pcd_path
            res = segment_object(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service segment_object call failed: %s' % e)
        rospy.loginfo('Service segment_object is executed.')

    def update_moveit_scene_client(self):
        wait_for_service('update_moveit_scene')
        try:
            update_moveit_scene = rospy.ServiceProxy('update_moveit_scene', ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            req = ManageMoveitSceneRequest()
            req.object_mesh_path = self.spawned_object_mesh_path
            req.object_pose_world = self.object_pose_stamped
            self.update_scene_response = update_moveit_scene(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_moveit_scene call failed: %s' % e)
        rospy.loginfo(
            'Service update_moveit_scene is executed %s.' % str(self.update_scene_response))

    def update_gazebo_object_client(self):
        """Gazebo management client, deletes previous object and spawns new object
        """
        raise NotImplementedError
        wait_for_service('update_gazebo_object')
        object_pose_array = get_pose_array_from_stamped(self.object_pose_stamped)
        try:
            update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
            res = update_gazebo_object(object_name, object_pose_array, object_model_name,
                                       model_type, dataset)
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' % str(res))
        return res.success

    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def reset_hithand_and_panda(self):
        self.reset_hithand_joints_client()
        self.plan_reset_trajectory_client()
        self.execute_joint_trajectory_client(smoothen_trajectory=True)

    def spawn_object(self, generate_random_pose):
        # Generate a random valid object pose
        if generate_random_pose:
            self.generate_random_object_pose_for_experiment()
        elif self.object_pose_stamped == None:
            rospy.logerr(
                "Object pose has not been initialized yet and generate_random_pose is false. Either set true or update object pose."
            )
        # Update gazebo object, delete old object and spawn new one
        self.update_gazebo_object_client()
        # Update moveit scene object
        self.update_moveit_scene_client()

    def save_data_post_grasp(self):
        self.save_visual_data_client()
        self.record_grasp_data_client()

    def save_visual_data_and_segment_object(self):
        self.save_visual_data_client()
        self.segment_object_client()

    def generate_hithand_preshape(self):
        """ Generate multiple grasp preshapes, which get stored in instance variable.
        """
        self.generate_hithand_preshape_client()

    def grasp_and_lift_object(self):
        # Control the hithand to it's preshape
        self.control_hithand_config_client()

        # Generate a robot trajectory to move to desired pose
        moveit_plan_exists = self.plan_arm_trajectory_client()
        if not moveit_plan_exists:
            for i in range(self.num_of_replanning_attempts):
                moveit_plan_exists = self.plan_arm_trajectory_client()
                if moveit_plan_exists:
                    rospy.loginfo("Found valid moveit plan after %d tries." % i + 1)
                    break
        # Possibly trajectory/pose needs an extra validity check or smth like that

        # Execute the generated joint trajectory
        self.execute_joint_trajectory_client(smoothen_trajectory=self.smooth_trajectories)

        # Close the hand
        self.grasp_control_hithand_client()

        # Lift the object
        lift_pose = self.chosen_palm_pose
        lift_pose.pose.position.z += self.object_lift_height
        self.plan_arm_trajectory_client(place_goal_pose=lift_pose)
        self.execute_joint_trajectory_client(smoothen_trajectory=True)