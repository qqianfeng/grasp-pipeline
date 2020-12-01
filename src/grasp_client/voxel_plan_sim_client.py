#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient


if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    dc_client = GraspClient()

    # Specify the object to be grasped, its pose, dataset, type, name etc.
    object_name = 'mustard_bottle'
    object_pose_array = [0., 0., 0., 1, 0., 0.]
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'

    datasets_base_path = '/home/vm/obect_datasets'

    # Main grasping logic
    rospy.loginfo('Trying to grasp object: %s' % object_name)
    dc_client.update_gazebo_object_client(
        object_name, object_pose_array, object_model_name, model_type, dataset)
