#!/usr/bin/env python
from pickletools import anyobject

import rospy
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
import os
import time
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils import utils
import random
import numpy as np
import h5py
from grasp_pipeline.utils.object_names_in_datasets import KIT_OBJECTS_DATA_GENERATED, OBJECTS_FOR_EVAL

KIT_OBJECTS_ALL =[]
for i in KIT_OBJECTS_DATA_GENERATED:
    KIT_OBJECTS_ALL.append('kit_' + i)

KIT_OBJECTS_TRAIN = [i for i in KIT_OBJECTS_ALL if i not in OBJECTS_FOR_EVAL ]

object_datasets_folder = rospy.get_param('object_datasets_folder')
gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')
metadata_handler = MetadataHandler(gazebo_objects_path)

heaps_per_item = 1

all_grasp_objects = []
for obj_full in KIT_OBJECTS_TRAIN:
    dset, obj_name = metadata_handler.split_full_name(obj_full)
    # get metadata on object
    metadata = metadata_handler.get_object_metadata(dset, obj_name)
    all_grasp_objects.append(metadata)

def find_random_obstacles(grasp_object, amount=3):

    objects = []
    object_names = set()
    num_total = len(all_grasp_objects)

    for _ in range(amount):
        obj = random.choice(all_grasp_objects)
        while obj['name'] == grasp_object['name'] or obj['name'] in object_names:
            obj = random.choice(all_grasp_objects)
        object_names.add(obj['name'])
        print(obj['name'])
        rospy.loginfo("obstacle object: %s"% obj['name'])
        objects.append(obj)
    return objects

def distribute_obstacle_objects_randomly(grasp_object_pose,
                                         obstacle_objects,
                                         min_center_to_center_distance=0.10):
    """Assign random location to each obstacle objects. Location is defined within a certain space.

    Args:
        grasp_object_pose (_type_):
        obstacle_objects (list): a list of obstacle objects
        min_center_to_center_distance (float, optional): distance between object center. Describe the clutterness of the scene. Defaults to 0.1.

    Returns:
        objects (list): a list of chosen obstacle objects
    """
    existing_object_positions = [np.array(grasp_object_pose)[:3]]
    for idx, obj in enumerate(obstacle_objects):
        obstacle_objects[idx] = grasp_client.set_to_random_pose(obj)
        position = np.array([
            obstacle_objects[idx]['mesh_frame_pose'].pose.position.x,
            obstacle_objects[idx]['mesh_frame_pose'].pose.position.y,
            obstacle_objects[idx]['mesh_frame_pose'].pose.position.z
        ])
        while not all([
                np.linalg.norm(position[:2] - existing_position[:2]) >
                min_center_to_center_distance for existing_position in existing_object_positions
        ]):
            obstacle_objects[idx] = grasp_client.set_to_random_pose(obj)
            position = np.array([
                obstacle_objects[idx]['mesh_frame_pose'].pose.position.x,
                obstacle_objects[idx]['mesh_frame_pose'].pose.position.y,
                obstacle_objects[idx]['mesh_frame_pose'].pose.position.z
            ])
        existing_object_positions.append(position)
    return obstacle_objects

if __name__ == '__main__':
    ####################  Init #################
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Create grasp client and metadata handler
    grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
    ############################################

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):
        if i < 10:
            continue
        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object(case="postprocessing")

        # assign to self variables
        grasp_client.update_object_metadata(object_metadata)

        # # while loop to generate heaps_per_item of heaps for each item.
        j = 0
        while j < heaps_per_item:
            # Create dirs to save rgb/depth
            grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

            # Spawn objects
            target_poses = [0.5, 0.0, 0.2, 0, 0, np.random.uniform(0, 2 * np.pi)]
            # Delete old target object, Spawn a new target object in Gazebo and moveit in a random valid pose and delete the old object
            pose_arr = grasp_client.spawn_object(pose_type="init", pose_arr=target_poses)

            grasp_client.set_path_and_save_visual_data(grasp_phase="single")
            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.segment_object_client(down_sample_pcd=True)


            obstacle_objects = find_random_obstacles(object_metadata)
            distribute_obstacle_objects_randomly(target_poses, obstacle_objects, min_center_to_center_distance=0.1)
            try:
                grasp_client.spawn_obstacle_objects(obstacle_objects)
            except Exception as e:
                print(e)
                grasp_client.remove_obstacle_objects(obstacle_objects)
                continue

            grasp_client.set_path_and_save_visual_data(grasp_phase="pre")

            # clean up obstacles
            grasp_client.remove_obstacle_objects(obstacle_objects)

            j+=1
            # grasp_client.clean_moveit_scene_client()

        #     # First take a shot of the scene and store RGB, depth and point cloud to disk
        #     # Then segment the object point cloud from the rest of the scene
        #     grasp_client.segment_object_client(down_sample_pcd=True)

        #     grasp_client.spawn_obstacle_objects(obstacle_objects)

        #     # Check validity of the heap, if there is collision already


        #     # Skip items which are too much occluded


        #     # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
        #     # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
        #     grasp_client.get_valid_preshape_for_all_points(obstacle_objects)

        #     grasp_client.set_path_and_save_visual_data(grasp_phase="pre")

        #     # Save all grasp data including post grasp images
        #     grasp_client.save_visual_data_and_record_grasp(obstacle_objects)

        #     # Finally write the time to file it took to test all poses
        #     grasp_client.remove_obstacle_objects_from_moveit_scene()

        #     # Clean up the environment by removeing all obstacle objects in gazebo and moveit
        #     grasp_client.remove_obstacle_objects(obstacle_objects)


        # # Remove target object from moveit scene. (maybe not necessry). The target object in gazebo will be removed in spawn_object function.
        # grasp_client.remove_target_object_from_moveit_scene(object_metadata["name"])
