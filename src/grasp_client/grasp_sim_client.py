#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from grasp_pipeline.srv import *

import numpy as np


class GraspClient():
    def __init__(self, datasets_base_path):
        rospy.init_node('grasp_client')
        self.datasets_base_path = datasets_base_path
        # save the mesh path of the currently spawned model
        self.spawned_object_name = None
        self.spawned_object_mesh_path = None
        self.spawned_object_pose = None
        self._setup_workspace_boundaries()

    # +++++++ PART I: First part are all the "helper functions" w/o interface to any other nodes/services ++++++++++
    def _setup_workspace_boundaries(self):
        """Sets the boundaries in which an object can be spawned and placed.
        Gets called 
        """
        self.spawn_object_x_min, self.spawn_object_x_max = 0.5, 1
        self.spawn_object_y_min, self.spawn_object_y_max = -0.5, 0.5
        self.table_height = 0

    def get_pose_stamped_from_array(self, pose_array, frame_id='/world'):
        """Transforms an array pose into a ROS stamped pose.
        """
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        r, p, y = pose_array[:3]
        quaternion = tft.quaternion_from_euler(r, p, y)
        pose.pose.orientation = quaternion  # x,y,z,w quaternion
        pose.pose.position = pose_array[3:]  # x,y,z position
        return pose

    def get_pose_array_from_stamped(self, pose_stamped):
        """Transforms a stamped pose into a 6D pose array.
        """
        r, p, y = tft.euler_from_quaternion(pose_stamped.pose.orientation)
        x_p, y_p, z_p = pose_stamped.pose.position
        pose_array = [r, p, y, x_p, y_p, z_p]
        return pose_array

    def generate_random_object_pose_for_experiment(self):
        """Generates a random x,y position and z orientation within object_spawn boundaries for grasping experiments.
        """
        rand_x = np.random.uniform(self.spawn_object_x_min,
                                   self.spawn_object_x_max)
        rand_y = np.random.uniform(self.spawn_object_y_min,
                                   self.spawn_object_y_max)
        rand_z_orientation = np.random.uniform(0., 2 * np.pi)
        object_pose = [
            0, 0, rand_z_orientation, rand_x, rand_y, self.table_height
        ]
        rospy.loginfo('Generated random object pose:')
        rospy.loginfo(object_pose)
        object_pose_stamped = self.get_pose_stamped_from_array(object_pose)
        self.spawned_object_pose = object_pose_stamped
        return object_pose_stamped

    # ++++++++ PART II: Second part consist of all clients that interact with different nodes/services ++++++++++++
    def create_moveit_scene_client(self, object_pose):
        rospy.loginfo('Waiting for service create_moveit_scene.')
        rospy.wait_for_service('create_moveit_scene')
        rospy.loginfo('Calling service create_moveit_scene.')
        try:
            create_moveit_scene = rospy.ServiceProxy('create_moveit_scene',
                                                     ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            create_scene_request = ManageMoveitSceneRequest(
                create_scene=True,
                object_mesh_path=self.spawned_object_mesh_path,
                object_pose_world=object_pose)
            self.create_scene_response = create_moveit_scene(
                create_scene_request)
            #print self.create_scene_response
        except rospy.ServiceException, e:
            rospy.loginfo('Service create_moveit_scene call failed: %s' % e)
        rospy.loginfo('Service create_moveit_scene is executed %s.' %
                      str(self.create_scene_response))

    def update_gazebo_object_client(self, object_name, object_pose,
                                    object_model_name, model_type, dataset):
        """Gazebo management client, deletes previous object and spawns new object
        """
        rospy.loginfo('Waiting for service update_gazebo_object.')
        rospy.wait_for_service('update_gazebo_object')
        rospy.loginfo('Calling service update_gazebo_object.')
        object_pose_array = self.get_pose_array_from_stamped(object_pose)
        try:
            update_gazebo_object = rospy.ServiceProxy('update_gazebo_object',
                                                      UpdateObjectGazebo)
            res = update_gazebo_object(object_name, object_pose_array,
                                       object_model_name, model_type, dataset)
            self.spawned_object_mesh_path = self.datasets_base_path + '/' + dataset + \
                '/models/' + object_model_name + '/google_16k/nontextured.stl'
            self.spawned_object_name = object_name
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' %
                      str(res))
        return res.success

    def control_hithand_config_client(self, go_home=False, close_hand=False):
        rospy.loginfo('Waiting for service control_hithand_config.')
        rospy.wait_for_service('control_hithand_config_server')
        rospy.loginfo('Calling service control_hithand_config.')
        try:
            req = ControlHithandRequest()
            control_hithand_config = rospy.ServiceProxy(
                'control_hithand_config', ControlHithand)
            if go_home:
                req.go_home = True
            elif close_hand:
                req.close_hand = True
            else:
                req.hithand_target_joint_state = None
                raise NotImplementedError  # Here we set the desires hithand preshape which comes from the response of the hithand_preshape_client
            # Buht how is the logic here for data gathering? Execute all of the samples and record responses right?
        except rospy.ServiceException as e:
            rospy.loginfo('Service control_hithand_config call failed: %s' % e)
        rospy.loginfo('Service control_allegro_config is executed %s.' %
                      str(self.control_response))

    def generate_hithand_preshape_client(self):
        rospy.loginfo('Waiting for service generate_hithand_preshape.')
        rospy.wait_for_service('generate_hithand_preshape')
        rospy.loginfo('Calling service generate_hithand_preshape.')
        try:
            generate_hithand_preshape = rospy.ServiceProxy('generate_hithand_preshape', GraspPreshape)
            req = GraspPreshapeRequest()
            req.sample = True

    # ++++++++ PART III: The third part consists of all the main logic/orchestration of Parts I and II ++++++++++++
    def grasp_and_lift_object(self, object_pose_stamped):
        self.create_moveit_scene_client(object_pose_stamped)

        self.control_hithand_config_client()

        grasp_arm_plan = None
        return grasp_arm_plan
