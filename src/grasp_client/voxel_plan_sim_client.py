#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient


if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    datasets_base_path = '/home/vm/object_datasets'
    dc_client = GraspClient(datasets_base_path)

    # Specify the object to be grasped, its pose, dataset, type, name etc.
    object_name = 'mustard_bottle'
    object_pose_array = [0., 0., 0., 1., 0., 0.]
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'

    # +++++++ Main grasping logic +++++++++++
    rospy.loginfo('Trying to grasp object: %s' % object_name)

    # Update gazebo object, delete old object and spawn new one
    dc_client.update_gazebo_object_client(
        object_name, object_pose_array, object_model_name, model_type, dataset)

    # Grasp and lift object
    object_pose_stamped = dc_client.get_pose_stamped_from_array(
        object_pose_array)  # This should come from a random generator, WHY do this if object spawned in fixed  position previously?
    grasp_arm_plan = dc_client.grasp_and_lift_object(object_pose_stamped)
