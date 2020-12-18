#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient

if __name__ == '__main__':
    # Specify the object to be grasped, its pose, dataset, type, name etc.
    object_name = 'mustard_bottle'
    object_model_name = '006_mustard_bottle'
    model_type = 'sdf'
    dataset = 'ycb'

    # Create the generic GraspClient, wrapper around all sorts of functionalities
    dc_client = GraspClient()

    # +++++++ Main grasping logic +++++++++++
    rospy.loginfo('Trying to grasp object: %s' % object_name)

    # Spawn a new object in Gazebo in a random valid pose and delete the old object
    dc_client.spawn_object_in_gazebo_random_pose(object_name, object_model_name, model_type,
                                                 dataset)

    # First take a shot of the scene and store RGB, depth and point cloud to disk
    # Then segment the object point cloud from the rest of the scene
    dc_client.save_visual_data_and_segment_object()

    # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
    #  of the GraspClient such that it can be used in the next API to generate a plan for the arm and move the hithand to the desired preshape
    # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
    dc_client.generate_hithand_preshape(
        grasp_type='unspecified')  # Grasp types can be either unspecified, top, or side

    # Grasp and lift object
    grasp_arm_plan = dc_client.grasp_and_lift_object()
