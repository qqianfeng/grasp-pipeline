#!/usr/bin/env python
from pickletools import anyobject

import rospy
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient, GraspClientCollData
import os
import time
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils import utils
import numpy as np
import h5py
import shutil
import open3d as o3d
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

if __name__ == '__main__':
    ####################  Init #################
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Create grasp client and metadata handler
    grasp_client = GraspClientCollData(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)

    # clean up
    grasp_client.clean_moveit_scene_client()
    grasp_client.delete_collision_h5_file()
    ############################################

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):
        if i < 10:
            continue
        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object(case="postprocessing")

        # assign to self variables
        grasp_client.update_object_metadata(object_metadata)

        # create object in collision ids h5 file
        grasp_client.create_object_group_in_h5_file(object_metadata)

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
            grasp_client.segment_object_client(down_sample_pcd=True,need_to_transfer_pcd_to_world_frame=True)

            # Move saved point cloud to rgb depth folder
            segmented_obj_pcd_path = "/home/yb/object.pcd"
            scene_pcd_path = "/home/yb/scene.pcd"
            single_pcd = o3d.io.read_point_cloud(segmented_obj_pcd_path)

            shutil.move(segmented_obj_pcd_path, os.path.dirname(grasp_client.color_img_save_path))
            shutil.move(scene_pcd_path, os.path.dirname(grasp_client.color_img_save_path))

            # Spawn obstacles
            obstacle_objects = grasp_client.find_random_obstacles(all_grasp_objects, object_metadata)
            grasp_client.distribute_obstacle_objects_randomly(target_poses, obstacle_objects, min_center_to_center_distance=0.1)
            try:
                grasp_client.spawn_obstacle_objects(obstacle_objects)
            except Exception as e:
                print(e)
                grasp_client.remove_obstacle_objects(obstacle_objects)
                continue

            grasp_client.set_path_and_save_visual_data(grasp_phase="pre")
            grasp_client.segment_object_client(down_sample_pcd=True,need_to_transfer_pcd_to_world_frame=True)

            multi_pcd = o3d.io.read_point_cloud(segmented_obj_pcd_path)

            # Move saved point cloud to rgb depth folder
            shutil.move(segmented_obj_pcd_path, os.path.dirname(grasp_client.color_img_save_path))
            shutil.move(scene_pcd_path, os.path.dirname(grasp_client.color_img_save_path))

            target_pcd, obstacle_pcd, obj_occluded = grasp_client.find_overlapped_pcd(single_pcd,multi_pcd)
            # filter the case when target too much occluded
            if obj_occluded:
                print('target obj is too much occluded')
                continue

            # save segmented pcd to file where rgb/depth is saved
            grasp_client.save_segmented_pcds(target_pcd, obstacle_pcd)

            # get grasps for target obj
            grasps_world = grasp_client.find_grasp_dist(object_metadata)

            # filter grasps
            prune_idxs, no_ik_idxs, collision_idxs = grasp_client.filter_palm_goal_poses_client(palm_poses=grasps_world)

            # save index to h5
            obj_full = object_metadata['name_rec_path']
            num_str = str(j).zfill(3)
            obj_full_scene_name = obj_full + '_scene' + num_str
            grasp_client.save_filtered_collision_label_per_grasp(obj_full,obj_full_scene_name,collision_idxs)

            # clean up obstacles and scene
            grasp_client.remove_obstacle_objects(obstacle_objects)
            grasp_client.clean_moveit_scene_client()

            j+=1

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
