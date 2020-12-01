#!/usr/bin/env python
import rospy

from grasp_pipeline.srv import *


class GraspClient():
    def __init__(self):
        rospy.init_node('grasp_client')

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
        except rospy.ServiceException, e:
            rospy.loginfo('Service update_gazebo_object call failed: %s' % e)
        rospy.loginfo('Service update_gazebo_object is executed %s.' %
                      str(res))
        return res.success
