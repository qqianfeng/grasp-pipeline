#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient
import os
import shutil
from object_names_in_datasets import YCB_OBJECTS, KIT_OBJECTS, BIGBIRD_OBJECTS


class MetaDataHandler():
    """ Simple class to help iterate through objects and 
    """
    def __init__(self, gazebo_objects_path='/home/vm/object_datasets/objects_gazebo'):
        self.datasets = [YCB_OBJECTS, KIT_OBJECTS, BIGBIRD_OBJECTS]
        self.datasets_name = ['ycb', 'kit', 'bigbird']
        self.object_ix = -1
        self.dataset_ix = 0
        self.gazebo_objects_path = gazebo_objects_path

    def choose_next_grasp_object(self):
        """ Iterates through all objects in all datasets and returns object_metadata. Gives a new object each time it is called.
        """
        # When this is called a new object is requested
        self.object_ix += 1

        # Check if we are past the last object of the dataset. If so take next dataset
        if self.object_ix == len(self.datasets[self.dataset_ix]):
            self.object_ix = 0
            self.dataset_ix += 1
            if self.dataset_ix == 3:
                self.dataset_ix = 0
        else:
            self.object_ix += 1

        # Set some relevant variables
        curr_dataset = self.datasets[self.dataset_ix]
        curr_dataset_name = self.datasets_name[self.dataset_ix]
        object_name = curr_dataset[self.object_ix]
        curr_object_path = os.path.join(self.gazebo_objects_path, curr_dataset_name, object_name)
        files = os.listdir(curr_object_path)
        collision_mesh = [s for s in files if "collision" in s]

        # Create the final metadata dict to return
        object_metadata = dict()
        object_metadata["name"] = object_name
        object_metadata["mesh_path"] = os.path.join(curr_object_path, collision_mesh)
        object_metadata["dataset"] = curr_dataset
        object_metadata["pose_world"] = None

        rospy.loginfo('Trying to grasp object: %s' % object_metadata["name"])

        return object_metadata


if __name__ == '__main__':
    # Define variables for nested for loops
    num_grasps_per_pose = 5  # how many sampled grasp poses to evaluate for object in same position
    num_poses_per_object = 5  # how many random poses should be tried for one object

    # Some relevant variables
    data_recording_path = '/home/vm/'
    gazebo_objects_path = '/home/vm/object_datasets/objects_gazebo'
    shutil.rmtree('/home/vm/grasp_data')

    # Create grasp client and metadata handler
    grasp_client = GraspClient(grasp_data_recording_path=data_recording_path)
    metadata_handler = MetaDataHandler(gazebo_objects_path=gazebo_objects_path)

    while True:
        grasp_client.create_dirs_new_grasp_trial()

        # Reset panda and hithand
        grasp_client.reset_hithand_and_panda()

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object()
        grasp_client.update_object_metadata(object_metadata)

        for _ in xrange(num_poses_per_object):
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
