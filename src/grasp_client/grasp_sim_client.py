#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from grasp_pipeline.srv import *


class GraspClient():
    def __init__(self, datasets_base_path):
        rospy.init_node('grasp_client')
        self.datasets_base_path = datasets_base_path
        # save the mesh path of the currently spawned model
        self.spawned_object_mesh_path = None

    def get_pose_stamped_from_array(self, pose_array, frame_id='/world'):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        quaternion = tft.quaternion_from_euler(
            pose_array[0], pose_array[1], pose_array[2])
        pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = quaternion
        pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = pose_array[
            3:]
        return pose_stamped

    def create_moveit_scene_client(self, object_pose):
        rospy.loginfo('Waiting for service create_moveit_scene.')
        rospy.wait_for_service('create_moveit_scene')
        rospy.loginfo('Calling service create_moveit_scene.')
        try:
            create_moveit_scene = rospy.ServiceProxy(
                'create_moveit_scene', ManageMoveitScene)
            # print(self.spawned_object_mesh_path)
            create_scene_request = ManageMoveitSceneRequest(
                create_scene=True, object_mesh_path=self.spawned_object_mesh_path, object_pose_world=object_pose)
            self.create_scene_response = create_moveit_scene(
                create_scene_request)
            #print self.create_scene_response
        except rospy.ServiceException, e:
            rospy.loginfo('Service create_moveit_scene call failed: %s' % e)
        rospy.loginfo('Service create_moveit_scene is executed %s.' %
                      str(self.create_scene_response))

    def update_gazebo_object_client(self, object_name, object_pose_array, object_model_name, model_type, dataset):
        '''
            Gazebo management client, deletes previous object and spawns new object
        '''
        rospy.loginfo('Waiting for service update_gazebo_object.')
        rospy.wait_for_service('update_gazebo_object')
        rospy.loginfo('Calling service update_gazebo_object.')
        try:
            update_gazebo_object = rospy.ServiceProxy(
                'update_gazebo_object', UpdateObjectGazebo)
            res = update_gazebo_object(
                object_name, object_pose_array, object_model_name, model_type, dataset)
            self.spawned_object_mesh_path = self.datasets_base_path + '/' + dataset + \
                '/models/' + object_model_name + '/google_16k/nontextured.stl'
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' %
                      str(res))
        return res.success

    def control_hithand_config_client(self, go_home=False, close_hand=False):
        rospy.loginfo('Waiting for service control_hithand_config.')
        rospy.wait_for_service('control_hithand_config')
        rospy.loginfo('Calling service control_hithand_config.')
        req = ControlHithandRequest()
        if go_home:
            req.go_home = True
        elif close_hand:
            req.close_hand = True
        else:
            req.hithand_target_joint_state =

    def plan_hithand_preshape_client

    def grasp_and_lift_object(self, object_pose_stamped):
        self.create_moveit_scene_client(object_pose_stamped)

        self.control_hithand_config_client()

        grasp_arm_plan = None
        return grasp_arm_plan
