#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from grasp_pipeline.srv import *
from sensor_msgs.msg import JointState

import numpy as np


class GraspClient():
    def __init__(self):
        rospy.init_node('grasp_client')
        self.object_datasets_folder = rospy.get_param('object_datasets_folder')
        self.color_img_save_path = rospy.get_param('color_img_save_path')
        self.depth_img_save_path = rospy.get_param('depth_img_save_path')
        self.object_pcd_path = rospy.get_param('object_pcd_path')
        self.scene_pcd_path = rospy.get_param('scene_pcd_path')

        # save the mesh path of the currently spawned model
        self.spawned_object_name = None
        self.spawned_object_mesh_path = None
        self.spawned_object_pose = None
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

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
    def _setup_workspace_boundaries(self):
        """Sets the boundaries in which an object can be spawned and placed.
        Gets called 
        """
        self.spawn_object_x_min, self.spawn_object_x_max = 0.5, 0.8
        self.spawn_object_y_min, self.spawn_object_y_max = -0.3, 0.3
        self.table_height = 0

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

    def generate_random_object_pose_for_experiment(self):
        """Generates a random x,y position and z orientation within object_spawn boundaries for grasping experiments.
        """
        rand_x = np.random.uniform(self.spawn_object_x_min, self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min, self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [0, 0, rand_z_orientation, rand_x, rand_y, self.table_height]
        rospy.loginfo('Generated random object pose:')
        rospy.loginfo(object_pose)
        object_pose_stamped = self.get_pose_stamped_from_array(object_pose)
        self.spawned_object_pose = object_pose_stamped

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
    def arm_moveit_cartesian_pose_planner_client(
        self,
        go_home=False,
        place_goal_pose=None,
    ):
        rospy.loginfo('Waiting for service arm_moveit_cartesian_pose_planner.')
        rospy.wait_for_service('arm_moveit_cartesian_pose_planner')
        rospy.loginfo('Calling service arm_moveit_cartesian_pose_planner.')
        try:
            moveit_cartesian_pose_planner = rospy.ServiceProxy('arm_moveit_cartesian_pose_planner',
                                                               PalmGoalPoseWorld)
            req = PalmGoalPoseWorldRequest()
            if go_home:
                req.go_home = True
            elif place_goal_pose is not None:
                req.palm_goal_pose_world = place_goal_pose
            else:
                req.palm_goal_pose_world = self.mount_desired_world.pose
                raise NotImplementedError
            planning_response = moveit_cartesian_pose_planner(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service arm_moveit_cartesian_pose_planner call failed: %s' % e)
        rospy.loginfo('Service arm_moveit_cartesian_pose_planner is executed %s.' %
                      str(planning_response.success))
        self.panda_planned_joint_trajectory = planning_response.plan_traj
        return planning_response.success

    def create_moveit_scene_client(self):
        rospy.loginfo('Waiting for service create_moveit_scene.')
        rospy.wait_for_service('create_moveit_scene')
        rospy.loginfo('Calling service create_moveit_scene.')
        try:
            create_moveit_scene = rospy.ServiceProxy('create_moveit_scene', ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            req = ManageMoveitSceneRequest()
            req.create_scene = True
            req.object_mesh_path = self.spawned_object_mesh_path
            req.object_pose_world = self.spawned_object_pose
            self.create_scene_response = create_moveit_scene(req)
            #print self.create_scene_response
        except rospy.ServiceException, e:
            rospy.loginfo('Service create_moveit_scene call failed: %s' % e)
        rospy.loginfo(
            'Service create_moveit_scene is executed %s.' % str(self.create_scene_response))

    def update_gazebo_object_client(self, object_name, object_model_name, model_type, dataset):
        """Gazebo management client, deletes previous object and spawns new object
        """
        rospy.loginfo('Waiting for service update_gazebo_object.')
        rospy.wait_for_service('update_gazebo_object')
        rospy.loginfo('Calling service update_gazebo_object.')
        object_pose_array = self.get_pose_array_from_stamped(self.spawned_object_pose)
        try:
            update_gazebo_object = rospy.ServiceProxy('update_gazebo_object', UpdateObjectGazebo)
            res = update_gazebo_object(object_name, object_pose_array, object_model_name,
                                       model_type, dataset)
            self.spawned_object_mesh_path = self.object_datasets_folder + '/' + dataset + \
                '/models/' + object_model_name + '/google_16k/nontextured.stl'
            self.spawned_object_name = object_name
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' % str(res))
        return res.success

    def control_hithand_config_client(self, go_home=False, close_hand=False):
        rospy.loginfo('Waiting for service control_hithand_config.')
        rospy.wait_for_service('control_hithand_config_server')
        rospy.loginfo('Calling service control_hithand_config.')
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

    def generate_hithand_preshape_client(self):
        """ Generates 
        """
        rospy.loginfo('Waiting for service generate_hithand_preshape.')
        rospy.wait_for_service('generate_hithand_preshape')
        rospy.loginfo('Calling service generate_hithand_preshape.')
        try:
            generate_hithand_preshape = rospy.ServiceProxy('generate_hithand_preshape',
                                                           GraspPreshape)
            req = GraspPreshapeRequest()
            req.sample = True
            self.heuristic_preshapes = generate_hithand_preshape(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service generate_hithand_preshape call failed: %s' % e)
        rospy.loginfo('Service generate_hithand_preshape is executed.')

    def save_visual_data_client(self):
        rospy.loginfo('Waiting for service save_visual_data.')
        rospy.wait_for_service('save_visual_data')
        rospy.loginfo('Calling service save_visual_data.')
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
        rospy.loginfo('Waiting for service segment_object.')
        rospy.wait_for_service('segment_object')
        rospy.loginfo('Calling service segment_object.')
        try:
            segment_object = rospy.ServiceProxy('segment_object', SegmentGraspObject)
            req = SegmentGraspObjectRequest()
            req.scene_pcd_path = self.scene_pcd_path
            req.object_pcd_path = self.object_pcd_path
            res = segment_object(req)
        except rospy.ServiceException, e:
            rospy.loginfo('Service segment_object call failed: %s' % e)
        rospy.loginfo('Service segment_object is executed.')

    def execute_joint_trajectory_client(self, smoothen_trajectory=True):
        """ Service call to smoothen and execute a joint trajectory.
        """
        rospy.loginfo('Waiting for service execute_joint_trajectory.')
        rospy.wait_for_service('execute_joint_trajectory')
        rospy.loginfo('Calling service execute_joint_trajectory.')
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

    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def spawn_object_in_gazebo_random_pose(self, object_name, object_model_name, model_type,
                                           dataset):
        # Generate a random valid object pose
        self.generate_random_object_pose_for_experiment()
        # Update gazebo object, delete old object and spawn new one
        self.update_gazebo_object_client(object_name, object_model_name, model_type, dataset)

    def save_visual_data_and_segment_object(self):
        self.save_visual_data_client()
        self.segment_object_client()

    def generate_hithand_preshape(self, grasp_type):
        """ First generate multiple grasp preshapes and then choose one specific one for grasp execution.
        """
        self.generate_hithand_preshape_client()
        self.choose_specific_grasp_preshape(grasp_type=grasp_type)

    def grasp_and_lift_object(self):
        # Spawn the object collision model in Moveit
        self.create_moveit_scene_client()

        # Control the hithand to it's preshape
        self.control_hithand_config_client()

        # Generate a robot trajectory to move to desired pose
        moveit_plan_exists = self.arm_moveit_cartesian_pose_planner_client()
        if not moveit_plan_exists:
            for i in range(self.num_of_replanning_attempts):
                moveit_plan_exists = self.arm_moveit_cartesian_pose_planner_client()
                if moveit_plan_exists:
                    rospy.loginfo("Found valid moveit plan after %d tries." % i + 1)
                    break
        # Possibly trajectory/pose needs an extra validity check or smth like that

        # Execute the generated joint trajectory
        self.execute_joint_trajectory_client(smoothen_trajectory=self.smooth_trajectories)