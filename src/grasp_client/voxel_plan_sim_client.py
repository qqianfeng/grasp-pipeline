#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient

if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    datasets_base_path = '/home/vm/object_datasets'
    dc_client = GraspClient(datasets_base_path)

    # Specify the object to be grasped, its pose, dataset, type, name etc.
    object_name = 'mustard_bottle'
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'

    # +++++++ Main grasping logic +++++++++++
    rospy.loginfo('Trying to grasp object: %s' % object_name)

    # Generate a random valid object pose
    object_pose = dc_client.generate_random_object_pose_for_experiment()

    # Update gazebo object, delete old object and spawn new one
    dc_client.update_gazebo_object_client(object_name, object_pose,
                                          object_model_name, model_type,
                                          dataset)

    # Generate hithand preshape
    hithand_preshape = dc_client.plan_hithand_preshape_client()

    # Grasp and lift object
    grasp_arm_plan = dc_client.grasp_and_lift_object(
        object_pose
    )  # object_pose is given to function in order for moveit scene server to spawn it in the right place
