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

poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
         [0.5, 0.0, 0.2, 0, 0, -1.571]]

all_grasp_objects = []


def get_obstacle_objects(metadata_handler, grasp_object, obstacle_data, target_pose):

    objects = []
    obstacle_group = obstacle_data[grasp_object['name']]
    for obj_idx in obstacle_group.keys():
        name = obstacle_group[obj_idx]['name'][()]
        obj = metadata_handler.get_object_metadata(False, name)
        obstacle_pose = obstacle_group[obj_idx]['obstacle_pose_in_target_frame'][()]

        ##############
        # Calculations
        target_T_obstacle_pose = utils.hom_matrix_from_pos_quat_list(obstacle_pose)
        target_pose_quat = utils.get_rot_quat_list_from_array(target_pose)
        world_T_target_pose = utils.hom_matrix_from_pos_quat_list(target_pose_quat)
        # world_Obstacle_pose = world_T_target x target_T_obstacle_pose
        obstacle_pose = np.matmul(world_T_target_pose, target_T_obstacle_pose)
        obstacle_pose_stamped = utils.pose_stamped_from_hom_matrix(obstacle_pose, 'world')
        obstacle_pose_stamped.pose.position.z = obj["spawn_height_z"]
        ###############

        obj["mesh_frame_pose"] = obstacle_pose_stamped
        objects.append(obj)
    return objects


if __name__ == '__main__':
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Create grasp client and metadata handler
    grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
    obstacle_data = h5py.File("/home/yb/Documents/obstacle_data.h5", 'r')

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):
        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object(case="generation")
        grasp_client.update_object_metadata(object_metadata)
        for pose_idx, pose in enumerate(poses):
            object_cycle_start = time.time()
            start = object_cycle_start

            obstacle_objects = get_obstacle_objects(metadata_handler, grasp_client.object_metadata,
                                                    obstacle_data, pose)

            # Create dirs
            grasp_client.create_dirs_new_grasp_trial_multi_obj(
                pose_idx, is_new_pose_or_object=True)  # TODO: here True is not always true?

            # Reset panda and hithand
            grasp_client.reset_hithand_and_panda()
            grasp_client.remove_obstacle_objects(obstacle_objects)
            grasp_client.clean_moveit_scene_client()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            grasp_client.spawn_object(pose_type="init", pose_arr=pose)

            grasp_client.set_path_and_save_visual_data(grasp_phase="single")
            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.segment_object_client(down_sample_pcd=True)

            rospy.logdebug("grasp_client.spawn_obstacle_objects(obstacle_objects)")
            grasp_client.spawn_obstacle_objects(obstacle_objects)

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_client.get_valid_preshape_for_all_points(obstacle_objects)

            grasp_client.set_path_and_save_visual_data(grasp_phase="pre")

            j = 0
            rospy.logdebug("start while loop to execute all sampled grasp poses")
            while grasp_client.grasps_available:

                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_client.create_dirs_new_grasp_trial_multi_obj(
                        pose_idx, is_new_pose_or_object=False)

                    # Reset panda and hithand
                    grasp_client.reset_hithand_and_panda()

                    # Spawn object in same position
                    grasp_client.spawn_object(pose_type="same")
                    rospy.logdebug("grasp_client.spawn_object(pose_type='same')")

                    grasp_client.reset_obstacle_objects(obstacle_objects)
                    rospy.logdebug("grasp_client.reset_obstacle_objects(obstacle_objects)")

                # Grasp and lift object
                execution_success = grasp_client.grasp_and_lift_object(obstacle_objects)

                # Save all grasp data including post grasp images
                grasp_client.save_visual_data_and_record_grasp(obstacle_objects)

                # measure time
                print("One cycle took: " + str(time.time() - start))
                if execution_success:
                    j += 1

                # for temp. test
                break
            # Finally write the time to file it took to test all poses
            grasp_client.log_object_cycle_time(time.time() - object_cycle_start)

            grasp_client.remove_obstacle_objects_from_moveit_scene()

        grasp_client.remove_target_object_from_moveit_scene(object_metadata["name"])
        # TODO remove target object from gazebo scene???
