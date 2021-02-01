#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient
import os
import shutil
# object_name, object_model_name, model_type, dataset
# res = update_gazebo_object(object_name, object_pose_array, object_model_name, model_type, dataset)
# req.object_mesh_path = self.spawned_object_mesh_path

# self.color_img_save_path = rospy.get_param('color_img_save_path')
# self.depth_img_save_path = rospy.get_param('depth_img_save_path')
# self.object_pcd_path = rospy.get_param('object_pcd_path')
# self.scene_pcd_path = rospy.get_param('scene_pcd_path')


def choose_grasp_object():
    """ Chooses a specific object from all objects available and returns a dict with all relevant metadata.
    """
    object_name = None
    object_mesh_path = None
    dataset = None

    object_metadata = dict()
    object_metadata["name"] = object_name
    object_metadata["mesh_path"] = object_mesh_path
    object_metadata["dataset"] = dataset
    object_metadata["pose_world"] = None

    rospy.loginfo('Trying to grasp object: %s' % object_name)

    return object_metadata


if __name__ == '__main__':
    # Create the generic GraspClient, wrapper around all sorts of functionalities
    data_recording_path = '/home/vm/'
    shutil.rmtree('/home/vm/grasp_data')
    grasp_client = GraspClient(grasp_data_recording_path=data_recording_path)
    num_grasps_per_pose = 5  # how many sampled grasp poses to evaluate for object in same position
    while True:
        grasp_client.create_dirs_new_grasp_trial()
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

                grasp_client.save_only_depth_and_color(grasp_phase='pre')

            # Grasp types can be either unspecified, top, or side
            grasp_type = 'unspecified'

            # From the sampled preshapes choose one specific for execution
            grasp_client.choose_specific_grasp_preshape(grasp_type=grasp_type)

            # Grasp and lift object
            grasp_arm_plan = grasp_client.grasp_and_lift_object()

            # Save all grasp data including post grasp images
            grasp_client.save_data_post_grasp()
