#!/usr/bin/env python
# from urllib import response
import rospy
import numpy as np

from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState

from grasp_pipeline.srv import *

import sys


class GazeboSceneManager():

    def __init__(self):
        rospy.init_node('manage_gazebo_scene_server')
        rospy.loginfo('Node: manage_gazebo_scene_server')
        self.prev_object_model_name = None
        self.objects_in_scene = set()
        self.scene_snapshot = None
        self.cache_grasp_object_spawn_info = None

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
            return False
        return True

    def handle_update_gazebo_object(self, req):
        print("update_gazebo_object: RECEIVED REQUEST")
        print(req)
        self.cache_grasp_object_spawn_info = req
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

    #############################################
    ## Test to spawn hand directly without arm ##
    #############################################

    def spawn_hand(self, object_name, hand_model_file, object_pose_array, model_type):
        rospy.wait_for_service('/gazebo/spawn_' + model_type + '_model')
        try:
            with open(hand_model_file, 'r') as f:
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
            # self.prev_object_model_name = object_name
        except rospy.ServiceException as e:
            print "Service call failed: %s" % e
            return False
        return True

    def delete_hand(self):
        self.delete_object(object_model_name='hand')

    def handle_update_gazebo_hand(self, req):
        print("update_gazebo_hand: RECEIVED REQUEST")
        print(req)
        self.spawn_hand(req.object_name, req.object_model_file, req.object_pose_array,
                          req.model_type)

        response = UpdateHandGazeboResponse()
        response.success = True
        return response

    def handle_delete_gazebo_hand(self, req):
        self.delete_hand()
        response = DeleteHandGazeboResponse()
        response.success = True
        return response

    def create_update_gazebo_hand_server(self):
        rospy.Service('update_gazebo_hand', UpdateHandGazebo, self.handle_update_gazebo_hand)
        rospy.loginfo('Service update_gazebo_hand:')
        rospy.loginfo('Ready to update the hand the in the Gazebo scene')

    def create_delete_gazebo_hand_server(self):
        rospy.Service('delete_gazebo_hand', DeleteHandGazebo, self.handle_delete_gazebo_hand)
        rospy.loginfo('Service delete_gazebo_hand:')
        rospy.loginfo('Ready to delete the hand the in the Gazebo scene')

    #####################################################
    ## below are codes for multiple objects generation ##
    #####################################################

    def spawn_object_do_not_modify_prev_object(self, object_name, object_model_file, object_pose_array, model_type):
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
        except rospy.ServiceException as e:
            print "Service call failed: %s" % e
            return False
        return True

    def create_server_create_new_scene(self):
        rospy.Service('create_new_scene', CreateNewScene, self.handle_create_new_scene)
        rospy.loginfo('Service create_new_scene:')
        rospy.loginfo('Ready to create new gazebo scene')

    def handle_create_new_scene(self, req):
        print("create_new_scene: RECEIVED REQUEST")
        print(req)
        self.clear_scene()
        self.clear_scene_snapshot()
        response = CreateNewSceneResponse()
        self.spawn_multiple_objects(req.objects_in_new_scene)
        if len(self.objects_in_scene) < len(req.objects_in_new_scene):
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
        self.scene_snapshot.attach_spawning_info(req.objects_in_new_scene)
        response.success = True
        return response

    def create_server_clear_scene(self):
        rospy.Service('clear_scene', ClearScene, self.handle_clear_scene)
        rospy.loginfo('Service create_new_scene:')
        rospy.loginfo('Ready to create new gazebo scene')

    def handle_clear_scene(self, req):
        print("clear_scene: RECEIVED REQUEST")
        print(req)
        self.clear_scene()
        self.clear_scene_snapshot()
        response = ClearSceneResponse()
        response.success = True
        return response

    def create_server_reset_scene(self):
        rospy.Service('reset_scene', ResetScene, self.handel_reset_scene)
        rospy.loginfo('Service reset_scene:')
        rospy.loginfo('Ready to reset gazebo scene')

    def handel_reset_scene(self, req):
        print("reset_scene: RECEIVED REQUEST")
        print(req)
        response = ResetSceneResponse()

        if not req.confirm:
            response.success = True
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
            if self.spawn_object_do_not_modify_prev_object(object.object_name, object.object_model_file,
                            object.object_pose_array, object.model_type):
                self.objects_in_scene.add(object)

    def clear_scene(self):
        for object in self.objects_in_scene.copy():
            self.delete_object(object.object_name)
            self.objects_in_scene.discard(object)

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

    def create_server_change_model_visibility(self):
        rospy.Service('change_model_visibility', ChangeModelVisibility, self.handle_change_model_visibility)
        rospy.loginfo('Service change_model_visibility:')
        rospy.loginfo('Ready to change model visibility')

    def handle_change_model_visibility(self, req):
        if req.visible:
            self.make_visible(req.model_name)
        else:
            self.make_invisible(req.model_name)
        response = ChangeModelVisibilityResponse()
        response.success = True
        return response

    def make_invisible(self, model_name):
        model_state = self.scene_snapshot.get_model_state_by_name(model_name)
        if model_state:
            self.scene_snapshot.model_states.discard(model_state)
            self.scene_snapshot.model_states_invisible_objects.add(model_state)
            self.delete_object(model_name)

    def make_visible(self, model_name):
        model_state = self.scene_snapshot.get_invisible_object_model_state_by_name(model_name)
        if model_state:
            self.scene_snapshot.model_states_invisible_objects.discard(model_state)
            self.scene_snapshot.model_states.add(model_state)

            if model_name == self.cache_grasp_object_spawn_info.object_name:
                # spawned with update_gazebo_object
                self.spawn_object_do_not_modify_prev_object(
                    self.cache_grasp_object_spawn_info.object_name,
                    self.cache_grasp_object_spawn_info.object_model_file,
                    self.cache_grasp_object_spawn_info.object_pose_array,
                    self.cache_grasp_object_spawn_info.model_type
                )
            else:
                # spawned with create_new_scene
                spawn_info = None
                for info in self.scene_snapshot.spawning_info:
                    if info.object_name == model_name:
                        spawn_info = info

                self.spawn_object_do_not_modify_prev_object(
                    spawn_info.object_name,
                    spawn_info.object_model_file,
                    spawn_info.object_pose_array,
                    spawn_info.model_type
                )
            self.set_model_state(model_state)

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
    return np.linalg.norm(np.array([twist.linear.x, twist.linear.y, twist.linear.z]), ord=2) < epsilon and \
           np.linalg.norm(np.array([twist.angular.x, twist.angular.y, twist.angular.z]), ord=2) < epsilon

class ObjectForSpawn():

    def __init__(self, object_name, object_model_file, object_pose_array, model_type):
        self.object_name = object_name
        self.object_model_file = object_model_file
        self.object_pose_array = object_pose_array
        self.model_type = model_type


class Scene():

    def __init__(self, model_states):
        self.model_states = set()
        self.model_states_invisible_objects = set()
        for each_name, each_pose, each_twist in zip(model_states.name, model_states.pose,
                                                    model_states.twist):
            model_state = ModelState()
            model_state.model_name = each_name
            model_state.pose = each_pose
            model_state.twist = each_twist
            self.model_states.add(model_state)

    def attach_spawning_info(self, spawning_info):
        self.spawning_info = spawning_info

    def get_model_state_by_name(self, name):
        for model_state in self.model_states:
            if model_state.model_name == name:
                return model_state
        return None

    def get_invisible_object_model_state_by_name(self, name):
        for model_state in self.model_states_invisible_objects:
            if model_state.model_name == name:
                return model_state
        return None


    #####################################################
    ## above are codes for multiple objects generation ##
    #####################################################


if __name__ == '__main__':
    manager = GazeboSceneManager()
    manager.create_update_gazebo_object_server()
    manager.create_server_create_new_scene()
    manager.create_server_reset_scene()
    manager.create_server_clear_scene()
    manager.create_server_change_model_visibility()

    manager.create_update_gazebo_hand_server()
    manager.create_delete_gazebo_hand_server()
    rospy.spin()
