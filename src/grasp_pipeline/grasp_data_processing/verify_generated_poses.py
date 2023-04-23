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

poses = [[0.5, 0.0, 0.2, 0, 0, 0]]

all_grasp_objects = []


def get_obstacle_objects(metadata_handler, grasp_object, obstacle_data, target_pose):

    objects = []
    obstacle_group = obstacle_data[grasp_object['name']]
    for obj_idx in obstacle_group.keys():
        name = obstacle_group[obj_idx]['name'][()]
        obj = metadata_handler.get_object_metadata(False, name)
        obstacle_pose = obstacle_group[obj_idx]['obstacle_pose_in_target_frame'][()]

        ##############
        # Update target pose according to dataset
        target_pose[3] = grasp_object["spawn_angle_roll"]  # 0
        target_pose[2] = grasp_object["spawn_height_z"]  # 0.05

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
    obstacle_data = h5py.File("/home/vm/Documents/obstacle_data.h5", 'r')

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
            if pose_idx != 0:
                grasp_client.remove_obstacle_objects(obstacle_objects)

            # grasp_client.clean_moveit_scene_client()

            # Delede old target object, Spawn a new target object in Gazebo and moveit in a random valid pose and delete the old object
            grasp_client.spawn_object(pose_type="init", pose_arr=pose)

            grasp_client.spawn_obstacle_objects(obstacle_objects)

            j = 0
            rospy.logdebug("start while loop to execute all sampled grasp poses")
            while grasp_client.grasps_available:

                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_client.create_dirs_new_grasp_trial_multi_obj(pose_idx,
                                                                       is_new_pose_or_object=False)

                    # Reset panda and hithand
                    grasp_client.reset_hithand_and_panda()

                    # Spawn object in same position
                    grasp_client.spawn_object(pose_type="same")
                    rospy.logdebug("grasp_client.spawn_object(pose_type='same')")

                    grasp_client.reset_obstacle_objects(obstacle_objects)
                    rospy.logdebug("grasp_client.reset_obstacle_objects(obstacle_objects)")

                # for temp. test
                break

            grasp_client.remove_obstacle_objects_from_moveit_scene()

        # End of this target object grasping experiment

        # Clean up the environment by removeing all obstacle objects in gazebo and moveit
        grasp_client.remove_obstacle_objects(obstacle_objects)

        # Remove target object from moveit scene. (maybe not necessry). The target object in gazebo will be removed in spawn_object function.
        grasp_client.remove_target_object_from_moveit_scene(object_metadata["name"])
