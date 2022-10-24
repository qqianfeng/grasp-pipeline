#!/usr/bin/env python
# from urllib import response
import rospy
import numpy as np

from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState

from grasp_pipeline.srv import *


class GazeboSceneManager():

    def __init__(self):
        rospy.init_node('manage_gazebo_scene_server')
        rospy.loginfo('Node: manage_gazebo_scene_server')
        self.prev_object_model_name = None
        self.objcets_in_scene = set()
        self.scene_snapshot = None

    def delete_object(self, object_model_name):
        try:
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model(object_model_name)
        except rospy.ServiceException as e:
            print "Delete model service failed: %s" % e

    def delete_prev_object(self, object_model_name):
        self.delete_object(object_model_name)
        if self.prev_object_model_name is None:
            return
        self.delete_object(self.prev_object_model_name)

    def spawn_object(self, object_name, object_model_file, object_pose_array, model_type):
        rospy.wait_for_service('/gazebo/spawn_' + model_type + '_model')
        try:
            with open(object_model_file, 'r') as f:
                model_file = f.read()
            quaternion = quaternion_from_euler(object_pose_array[3], object_pose_array[4],
                                               object_pose_array[5])
            initial_pose = Pose()
            initial_pose.position.x = object_pose_array[0]
            initial_pose.position.y = object_pose_array[1]
            initial_pose.position.z = object_pose_array[2]
            initial_pose.orientation.x = quaternion[0]
            initial_pose.orientation.y = quaternion[1]
            initial_pose.orientation.z = quaternion[2]
            initial_pose.orientation.w = quaternion[3]
            rospy.loginfo('Spawning model: ' + object_name)
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_' + model_type + '_model', SpawnModel)
            spawn_model(object_name, model_file, '', initial_pose, 'world')
            self.prev_object_model_name = object_name
        except rospy.ServiceException as e:
            print "Service call failed: %s" % e

    def handle_update_gazebo_object(self, req):
        print("RECEIVED REQUEST")
        print(req)
        self.delete_prev_object(req.object_name)
        rospy.sleep(1)
        self.spawn_object(req.object_name, req.object_model_file, req.object_pose_array,
                          req.model_type)

        response = UpdateObjectGazeboResponse()
        response.success = True
        return response

    def create_update_gazebo_object_server(self):
        rospy.Service('update_gazebo_object', UpdateObjectGazebo, self.handle_update_gazebo_object)
        rospy.loginfo('Service update_gazebo_object:')
        rospy.loginfo('Ready to update the object the in the Gazebo scene')

    #####################################################
    ## below are codes for multiple objects generation ##
    #####################################################

    def create_server_create_new_scene(self):
        rospy.Service('create_new_scene', CreateNewScene, self.handle_create_new_scene)
        rospy.loginfo('Service create_new_scene:')
        rospy.loginfo('Ready to create new gazebo scene')

    def handle_create_new_scene(self, req):
        print("RECEIVED REQUEST")
        print(req)
        self.clear_scene()
        self.clear_scene_snapshot()
        response = CreateNewSceneResponse()

        self.spawn_multiple_objects(req.objects_in_new_scene)
        if len(self.objcets_in_scene) < len(req.objects_in_new_scene):
            # TODO how to avoid the failure of spawning new object
            response.success = False
            return response
        rospy.sleep(3)
        try:
            self.take_scene_snapshot()
        except rospy.ServiceException as e:
            print 'service call failed: %s' % e
            response.success = False
            return response

        response.success = True
        return response

    def create_server_reset_scene(self):
        rospy.Service('reset_scene', ResetScene, self.handel_reset_scene)
        rospy.loginfo('Service reset_scene:')
        rospy.loginfo('Ready to reset gazebo scene')

    def handel_reset_scene(self, req):
        print("RECEIVED REQUEST")
        print(req)
        response = ResetSceneResponse()

        if not req.confirm:
            response.success = False
            return response
        try:
            self.recover_scene_to_snapshot()
            response.success = True
        except Exception as e:
            print e
            response.success = False
        return response

    def spawn_multiple_objects(self, object_list):
        for object in object_list:
            self.spawn_object(object.object_name, object.object_model_file,
                              object.object_pose_array, object.model_type)
        self.objcets_in_scene.add(object)

    def clear_scene(self):
        for object in self.objcets_in_scene:
            self.delete_object(object.object_name)
            self.objcets_in_scene.discard(object)

    def take_scene_snapshot(self):
        scene = _get_stationary_scene()
        if scene:
            self.scene_snapshot = scene
        else:
            raise rospy.ServiceException(
                "Failed to take scene snapshot, some of the objects in the scene did not rest.")

    def recover_scene_to_snapshot(self):
        if not self.scene_snapshot:
            raise RuntimeError("snapshot not found")
        for model_state in self.scene_snapshot.model_states:
            self.set_model_state(model_state)

    def clear_scene_snapshot(self):
        self.scene_snapshot = None

    def set_model_state(self, model_state):
        service_name = '/gazebo/set_model_state'
        rospy.wait_for_service(service_name)
        service_proxy = rospy.ServiceProxy(service_name, SetModelState)
        try:
            service_proxy(model_state)
        except rospy.ServiceException as e:
            print "Service call failed: %s" % e


def _get_stationary_scene():
    num_retries_allowed = 5
    waiting_time_between_retries_in_seconds = 1
    while num_retries_allowed > 0:
        model_states = _get_model_states()
        if _check_all_are_stationary(model_states.twist):
            return Scene(model_states)
        else:
            num_retries_allowed -= 1
            rospy.sleep(waiting_time_between_retries_in_seconds)


def _get_model_states():
    topic = '/gazebo/model_states'
    return rospy.wait_for_message(topic, ModelStates)


def _check_all_are_stationary(twists):
    return all([_check_is_stationary(twist) for twist in twists])


def _check_is_stationary(twist):
    epsilon = 0.01
    return np.linalg.norm(np.array(twist.linear ), ord=2) < epsilon and \
           np.linalg.norm(np.array(twist.angular), ord=2) < epsilon


class ObjectForSpawn():

    def __init__(self, object_name, object_model_file, object_pose_array, model_type):
        self.object_name = object_name
        self.object_model_file = object_model_file
        self.object_pose_array = object_pose_array
        self.model_type = model_type


class Scene():

    def __init__(self, model_states):
        self.model_states = set()
        for each_name, each_pose, each_twist in zip(model_states.name, model_states.pose,
                                                    model_states.twist):
            model_state = ModelState()
            model_state.model_name = each_name
            model_state.pose = each_pose
            model_state.twist = each_twist
            self.model_states.add(model_state)

    #####################################################
    ## above are codes for multiple objects generation ##
    #####################################################


if __name__ == '__main__':
    manager = GazeboSceneManager()
    manager.create_update_gazebo_object_server()
    manager.create_server_create_new_scene()
    manager.create_server_reset_scene()
    rospy.spin()
