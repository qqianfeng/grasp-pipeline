#!/usr/bin/env python
import rospy
from grasp_sim_client import GraspClient
import os
import time
import shutil
from object_names_in_datasets import YCB_OBJECTS, KIT_OBJECTS, BIGBIRD_OBJECTS

kit_no_roll_angle = [""]
YCB_PI_HALF_ROLL = ["008_pudding_box", "035_power_drill"]


class MetaDataHandler():
    """ Simple class to help iterate through objects and 
    """
    def __init__(self, gazebo_objects_path='/home/vm/object_datasets/objects_gazebo'):
        self.datasets = [YCB_OBJECTS, BIGBIRD_OBJECTS, KIT_OBJECTS]
        self.datasets_name = ['ycb', 'bigbird', 'kit']
        self.object_ix = -1
        self.dataset_ix = 0
        self.gazebo_objects_path = gazebo_objects_path

    def choose_next_grasp_object(self):
        """ Iterates through all objects in all datasets and returns object_metadata. Gives a new object each time it is called.
        """
        choose_success = False
        while (not choose_success):
            try:
                # When this is called a new object is requested
                self.object_ix += 1

                # Check if we are past the last object of the dataset. If so take next dataset
                if self.object_ix == len(self.datasets[self.dataset_ix]):
                    self.object_ix = 0
                    self.dataset_ix += 1
                    if self.dataset_ix == 3:
                        self.dataset_ix = 0

                # Set some relevant variables
                curr_dataset = self.datasets[self.dataset_ix]
                curr_dataset_name = self.datasets_name[self.dataset_ix]
                object_name = curr_dataset[self.object_ix]
                curr_object_path = os.path.join(self.gazebo_objects_path, curr_dataset_name,
                                                object_name)
                files = os.listdir(curr_object_path)
                collision_mesh = [s for s in files if "collision" in s][0]

                # Create the final metadata dict to return
                object_metadata = dict()
                object_metadata["name"] = object_name
                object_metadata["model_file"] = os.path.join(curr_object_path,
                                                             object_name + '.sdf')
                object_metadata["collision_mesh_path"] = os.path.join(
                    curr_object_path, collision_mesh)
                object_metadata["dataset"] = curr_dataset_name
                object_metadata["name_rec_path"] = curr_dataset_name + '_' + object_name
                object_metadata["mesh_frame_pose"] = None
                object_metadata["seg_pose"] = None
                object_metadata["aligned_pose"] = None
                object_metadata["seg_dim_whd"] = None
                object_metadata["aligned_dim_whd"] = None
                object_metadata["spawn_height_z"] = 0.2
                object_metadata["spawn_angle_roll"] = 0
                if curr_dataset_name == 'kit' or object_name in YCB_PI_HALF_ROLL:
                    object_metadata["spawn_angle_roll"] = 1.57079632679
                    if object_name in kit_no_roll_angle:
                        object_metadata["spawn_angle_roll"] = 0

                rospy.loginfo('Trying to grasp object: %s' % object_metadata["name"])
                choose_success = True
            except:
                self.object_ix += 1

        return object_metadata


num_objects = len(YCB_OBJECTS + BIGBIRD_OBJECTS + KIT_OBJECTS)

poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
         [0.5, 0.0, 0.2, 0, 0, -1.571]]

if __name__ == '__main__':
    # Some relevant variables
    data_recording_path = '/home/vm/'
    gazebo_objects_path = '/home/vm/gazebo-objects/objects_gazebo'

    # Remove these while testing
    shutil.rmtree('/home/vm/grasp_data', ignore_errors=True)

    # Create grasp client and metadata handler
    grasp_client = GraspClient(grasp_data_recording_path=data_recording_path)
    metadata_handler = MetaDataHandler(gazebo_objects_path=gazebo_objects_path)

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(num_objects):

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object()
        grasp_client.update_object_metadata(object_metadata)

        for pose in poses:
            start = time.time()

            # Create dirs
            grasp_client.create_dirs_new_grasp_trial()

            # Reset panda and hithand
            grasp_client.reset_hithand_and_panda()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            grasp_client.spawn_object(pose_type="init", pose_arr=pose)

            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.save_visual_data_and_segment_object()

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_client.get_valid_preshape_for_all_points()

            j = 0
            while grasp_client.grasps_available:
                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_client.create_dirs_new_grasp_trial()

                    # Reset panda and hithand
                    grasp_client.reset_hithand_and_panda()

                    # Spawn object in same position
                    grasp_client.spawn_object(pose_type="same")

                # Grasp and lift object
                grasp_arm_plan = grasp_client.grasp_and_lift_object()

                # Save all grasp data including post grasp images
                grasp_client.save_visual_data_and_record_grasp()

                # measure time
                print("One cycle took: " + str(time.time() - start))
                j += 1
