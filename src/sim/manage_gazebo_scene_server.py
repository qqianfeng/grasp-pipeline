#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel

from grasp_pipeline.srv import *


class GazeboSceneManager():
    def __init__(self):
        rospy.init_node('manage_gazebo_scene_server')
        rospy.loginfo('Node: manage_gazebo_scene_server')
        self.prev_object_model_name = None

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
            quaternion = quaternion_from_euler(object_pose_array[0], object_pose_array[1],
                                               object_pose_array[2])
            initial_pose = Pose()
            initial_pose.position.x = object_pose_array[3]
            initial_pose.position.y = object_pose_array[4]
            initial_pose.position.z = object_pose_array[5]
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


if __name__ == '__main__':
    manager = GazeboSceneManager()
    manager.create_update_gazebo_object_server()
    rospy.spin()
