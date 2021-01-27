#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient

object_name, object_model_name, model_type, dataset
res = update_gazebo_object(object_name, object_pose_array, object_model_name, model_type, dataset)
req.object_mesh_path = self.spawned_object_mesh_path


def choose_grasp_object():
    """ Chooses a specific object from all objects available and returns 
    """
    object_name = None
    object_mesh_path = None

    object_metadata = dict()
    object_metadata["object_name"] = object_name
    object_metadata["object_mesh_path"] = object_mesh_path

    rospy.loginfo('Trying to grasp object: %s' % object_name)

    return object_metadata


if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    grasp_client = GraspClient()
    num_grasps_per_pose = 5  # how many sampled grasp poses to evaluate for object in same position
    while True:
        # Reset panda and hithand
        grasp_client.reset_hithand_and_panda()

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = choose_grasp_object()
        grasp_client.update_object_metadata(object_metadata)

        # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
        grasp_client.spawn_object(generate_random_pose=True)

        # First take a shot of the scene and store RGB, depth and point cloud to disk
        # Then segment the object point cloud from the rest of the scene
        grasp_client.save_visual_data_and_segment_object()

        # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
        # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
        grasp_client.generate_hithand_preshape()

        for i in xrange(num_grasps_per_pose):
            if i != 0:  # Spawn object in same pose, just choose different grasp configuration. No need for segmentation etc as it won't be different
                grasp_client.reset_hithand_and_panda()

                grasp_client.spawn_object(generate_random_pose=False)

            # Grasp types can be either unspecified, top, or side
            grasp_type = 'unspecified'

            # From the sampled preshapes choose one specific for execution
            grasp_client.choose_specific_grasp_preshape(grasp_type=grasp_type)

            # Grasp and lift object
            grasp_arm_plan = grasp_client.grasp_and_lift_object()

            # Save all grasp data including post grasp images
            grasp_client.save_data_post_grasp()

            # Save grasping experiment data
