#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
from std_msgs.msg import Float64MultiArray


class ServerUnitTester():
    def __init__(self):
        rospy.loginfo('Running Unit Tests')
        self.test_count = 0

    def test_manage_gazebo_scene_server(self, object_name, object_model_name, dataset, model_type):
        self.test_count += 1
        rospy.loginfo(
            'Running test_manage_gazebo_scene_server, test number %d' % self.test_count)
        update_gazebo_object = rospy.ServiceProxy(
            'update_gazebo_object', UpdateObjectGazebo)
        pose = Float64MultiArray()
        pose.data = [0, 0, 0, 1, 0, 0]
        res = update_gazebo_object(
            object_name, pose, object_model_name, model_type, dataset)

        result = 'SUCCEDED' if res else 'FAILED'
        rospy.loginfo(result)


if __name__ == '__main__':
    # Define variables for testing
    object_name = 'mustard_bottle'
    object_model_name = '006_mustard_bottle'
    dataset = 'ycb'
    model_type = 'sdf'
    # Test
    sut = ServerUnitTester()
    sut.test_manage_gazebo_scene_server(
        object_name, object_model_name, dataset, model_type)
