#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient

if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    dc_client = GraspClient()

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
    dc_client.update_gazebo_object_client(
        object_name, object_pose, object_model_name, model_type, dataset)

    # Call a service to segment the object point_cloud and store the result
    dc_client.segment_object_client()

    # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
    #  of the GraspClient such that it can be used in the next API to generate a plan for the arm and move the hithand to the desired preshape
    dc_client.generate_hithand_preshape_client()

    # The previous call generated and saved a variable with multiple grasp shapes, now one specific desired grasp preshape should be chosen. This preshape (characteristed by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
    dc_client.choose_specific_preshape(grasp_type='unspecified')
    # Grasp and lift object
    grasp_arm_plan = dc_client.grasp_and_lift_object(
        object_pose, grasp_type='unspecified'
    )  # object_pose is given to function in order for moveit scene server to spawn it in the right place, this is needed for planning
    # Grasp types can be either unspecified, top, or side
