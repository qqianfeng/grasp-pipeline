"""Test the collision detection module by manually define the hand pose"""
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

pose = [0.5, 0.0, 0.2, 0, 0, 0]

all_grasp_objects = []


def get_obstacle_objects(metadata_handler, grasp_client, obstacle_data, target_pose):

    grasp_object = grasp_client.object_metadata
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
    grasp_client = GraspClient(is_rec_sess=False)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)

    obstacle_data = h5py.File(rospy.get_param('obstacle_data_path'), 'r')

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):
        # Specify the object to be grasped, its pose, dataset, type, name etc.
        if i == 0:
            continue
        object_metadata = metadata_handler.choose_next_grasp_object(case="generation")
        grasp_client.update_object_metadata(object_metadata)

        obstacle_objects = get_obstacle_objects(metadata_handler, grasp_client, obstacle_data,
                                                pose)

        # # Create dirs
        # grasp_client.create_dirs_new_grasp_trial_multi_obj(
        #     0, is_new_pose_or_object=True)  # TODO: here True is not always true?

        # Reset panda and hithand
        grasp_client.reset_hithand_and_panda()

        # grasp_client.clean_moveit_scene_client()

        # Delede old target object, Spawn a new target object in Gazebo and moveit in a random valid pose and delete the old object
        pose_arr = grasp_client.spawn_object(pose_type="init", pose_arr=pose)

        grasp_client.set_path_and_save_visual_data(grasp_phase="single")

        # First take a shot of the scene and store RGB, depth and point cloud to disk
        # Then segment the object point cloud from the rest of the scene
        grasp_client.segment_object_client(down_sample_pcd=True)

        rospy.logdebug("grasp_client.spawn_obstacle_objects(obstacle_objects)")
        grasp_client.spawn_obstacle_objects(obstacle_objects)


        # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
        # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
        # grasp_client.get_valid_preshape_for_all_points(obstacle_objects)

        quat = [0,0,0,1] #-> not good, acc 0.02~0.03
        # quat = [0.081108347351,0.727595910875,-0.0502049014956,0.679341662445] # vertical pose
        # quat = [-0.081108347351,-0.727595910875,0.0502049014956,0.679341662445] # vertical pose
        # quat = [0.532688740778,0.298678626628,0.457068261427,0.646623837977] # horizontal, -> good. acc 0.01
        # grasp_client.set_path_and_save_visual_data(grasp_phase="pre")
        while True:
            z = raw_input()
            transl = [0.4,-0.2,float(z)]
            grasp_pose = utils.get_pose_stamped_from_trans_and_quat(transl,quat)
            collision = grasp_client.filter_palm_goal_poses_client([grasp_pose])
            print(collision)
            grasp_pose_arr = utils.get_pose_array_from_stamped(grasp_pose)
            grasp_client.spawn_hand(grasp_pose_arr)
            raw_input('hand ok?')
            grasp_client.delete_hand()
