""" The idea of this is to create a dataset for training the grasp VAE. Different objects are being spawned sequentially
in random poses. They are being segmented and the object-centric frame in camera coords is extracted and saved. Also the mesh frame of the object is being stored.
This script only verifies the grasp_data_all.h5 file
"""

import os
import copy
import rospy
import numpy as np
import h5py
from time import time
import pickle

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandler
import grasp_pipeline.utils.utils as utils
from grasp_pipeline.utils.object_names_in_datasets import *
from grasp_pipeline.utils.check_gazebo_collision import get_contact, check_collision

def mkdir(base_folder, folder_name=None):
    path = os.path.join(base_folder, folder_name) if folder_name is not None else base_folder
    if not os.path.exists(path):
        os.mkdir(path)


def get_all_objects(gazebo_objects_path):
    all_grasp_objects = []
    metadata_handler = MetadataHandler(gazebo_objects_path)
    num_total = metadata_handler.get_total_num_objects()
    for _ in range(num_total):
        all_grasp_objects.append(metadata_handler.choose_next_grasp_object(case=''))
    return all_grasp_objects


def find_objects(all_objects, name1, name2, name3):
    objects = []
    print('obstacle object to find:')
    print(name1, name2, name3)
    for obj in all_objects:
        if obj['name'] == name1:
            print('found object:', name1)
            objects.append(obj)
        elif obj['name'] == name2:
            print('found object:', name2)
            objects.append(obj)
        elif obj['name'] == name3:
            print('found object:', name3)
            objects.append(obj)
    return objects


def assign_obstacle_objects_pose(obstacle_object, pose_stamped):
    obstacle_object['mesh_frame_pose'] = pose_stamped


def test_grasp_pose_transform(dset_obj_name, grasp_client):
    # Create data and metadata handler
    file_path = os.path.join("/home/yb/multi_grasp_data", "grasp_data_all.h5")
    data_handler = GraspDataHandler(file_path=file_path, sess_name='recording_session_0001')

    # Get a single successful grasp example
    grasp_data = data_handler.get_single_successful_grasp(dset_obj_name, random=True)

    # TODO: Change preshape palm world pose to preshape palm mesh frame
    # Take the 6D palm position and the mesh frame during data generation and transform to pose stamped
    palm_pose = grasp_data["true_preshape_palm_world_pose"]  # w.r.t the mesh frame NOT world
    object_mesh_frame_data_gen = grasp_data["object_world_sim_pose"]
    palm_pose_mf_dg_stamped = utils.get_pose_stamped_from_rot_quat_list(
        palm_pose, frame_id="object_mesh_frame_data_gen")
    obj_mf_dg_stamped = utils.get_pose_stamped_from_rot_quat_list(object_mesh_frame_data_gen)

    # Publish the object mesh frame as it was during data generation TODO: Change object_world_sim_pose to object_mesh_frame_world
    grasp_client.update_object_mesh_frame_data_gen_client(obj_mf_dg_stamped)

    # Update the transformed pose and check validity in ros
    grasp_client.update_grasp_palm_pose_client(palm_pose_mf_dg_stamped)

    # Transform first the chosen pose from mesh_frame during data gen to current mesh frame by just changing the frame_id
    assert palm_pose_mf_dg_stamped.header.frame_id == "object_mesh_frame_data_gen"
    palm_pose_curr_mf = copy.deepcopy(palm_pose_mf_dg_stamped)
    palm_pose_curr_mf.header.frame_id = "object_mesh_frame"

    # Transform the valid grasp pose from current mesh frame to world
    palm_pose_world = grasp_client.transform_pose(
        palm_pose_curr_mf, from_frame='object_mesh_frame', to_frame='world')

    # Update the transformed pose and check validity in ros
    grasp_client.update_grasp_palm_pose_client(palm_pose_world)

    # Lookup transform between current mesh frame and object_centroid_vae
    transform_cent_mf = grasp_client.tf_buffer.lookup_transform(
        "object_centroid_vae", "object_mesh_frame", rospy.Time(0), timeout=rospy.Duration(10))
    t = transform_cent_mf.transform.translation
    # Get the hom matrix from transform
    cent_T_mf = utils.hom_matrix_from_ros_transform(transform_cent_mf)

    # Get hom matrix from rot_quat list
    grasp_mf = utils.hom_matrix_from_pose_stamped(palm_pose_mf_dg_stamped)
    print "Transform from meshframe to centroid: \n", cent_T_mf

    # Get the transformed grasp
    grasp_cent = np.matmul(cent_T_mf, grasp_mf)
    print "Grasp mesh Frame\n", grasp_mf
    print "Grasp centroid\n", grasp_cent

    # Transform the grasp to a stamped pose
    grasp_cent_stamped = utils.pose_stamped_from_hom_matrix(
        grasp_cent, frame_id='object_centroid_vae')

    # Update palm pose verify that it is correct
    grasp_client.update_grasp_palm_pose_client(grasp_cent_stamped)

    transformed_pose = grasp_client.transform_pose(palm_pose_mf_dg_stamped, 'object_mesh_frame',
                                                   'object_centroid_vae')
    tp_hom = utils.hom_matrix_from_pose_stamped(transformed_pose)
    print("tp_hom \n", tp_hom)

    while (True):
        rospy.sleep(5)


def create_object_group_in_h5_file(full_obj_name, full_save_path):
    with h5py.File(full_save_path, 'a') as hdf:
        if full_obj_name not in hdf.keys():
            hdf.create_group(full_obj_name)
        else:
            print("Name %s already existed in file." % full_obj_name)


def save_mesh_frame_centroid_tf(obj_full, full_save_path, obj_full_pcd, tf_list):
    with h5py.File(full_save_path, 'r+') as hdf:
        try:
            hdf[obj_full].create_dataset(obj_full_pcd + '_mesh_to_centroid', data=tf_list)
        except RuntimeError:
            hdf[obj_full][obj_full_pcd + '_mesh_to_centroid'][...] = tf_list


def save_mesh_frame_world_tf(obj_full, full_save_path, obj_full_pcd, tf_list):
    with h5py.File(full_save_path, 'r+') as hdf:
        try:
            hdf[obj_full].create_dataset(obj_full_pcd + '_mesh_to_world', data=tf_list)
        except RuntimeError:
            hdf[obj_full][obj_full_pcd + '_mesh_to_world'][...] = tf_list


if __name__ == '__main__':
    # Some "hyperparameters"
    # For debug purpose, create a file like 'new_data' and replace the name.
    # n_pcds_per_obj = 50

    # This h5 should be the file that is merged
    # input_grasp_data_file = '/home/vm/Documents/grasp_data_all.h5'
    input_grasp_data_file = '/data/hdd1/qf/hithand_data/collision_only_data_new/grasp_data_all.h5'
    gazebo_objects_path = '/home/yb/Projects/gazebo-objects/objects_gazebo/'
    # gazebo_objects_path = '/home/vm/gazebo-objects/objects_gazebo/'

    # Get all available objects and choose one
    with h5py.File(input_grasp_data_file, 'r') as hdf:
        objects = hdf.keys()

    all_objects = get_all_objects(gazebo_objects_path)
    # Make the base directory
    # dest_folder = os.path.join('/home', os.getlogin(), 'new_data_full/')
    # pcds_folder = os.path.join(dest_folder, 'point_clouds')
    # pcd_tfs_path = os.path.join(dest_folder, 'pcd_transforms.h5')
    # mkdir(pcds_folder)
    data_recording_path = os.path.join('/tmp/')
    # Instantiate grasp client
    grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path)

    with h5py.File(input_grasp_data_file, 'r') as hdf:
        # Iterate through all objects
        for obj_full_name in hdf.keys():
            if obj_full_name.split('_')[0] == 'bigbird':
                continue
            if obj_full_name != 'kit_BakingVanilla':
                continue
            obj_data = hdf[obj_full_name]  # -> collision, negative, positive

            # test_data = obj_data['collision']
            test_data = obj_data['non_collision_not_executed']
            # test_data = obj_data['negative']

            collision_grasp = test_data[test_data.keys()[0]]
            object_mesh_frame_world = collision_grasp['object_mesh_frame_world'][()]
            object_mesh_frame_world_mat = utils.hom_matrix_from_pos_quat_list(
                    object_mesh_frame_world)
            # Get obstacle names and poses
            obstacle1_name = collision_grasp['obstacle1_name'][()]
            obstacle2_name = collision_grasp['obstacle2_name'][()]
            obstacle3_name = collision_grasp['obstacle3_name'][()]

            # [7,] pose, trans+quat
            obstacle1_mesh_frame_world = collision_grasp['obstacle1_mesh_frame_world'][()]
            obstacle2_mesh_frame_world = collision_grasp['obstacle2_mesh_frame_world'][()]
            obstacle3_mesh_frame_world = collision_grasp['obstacle3_mesh_frame_world'][()]

            obstacle_objects = find_objects(all_objects, obstacle1_name, obstacle2_name,
                                            obstacle3_name)

            dset = obj_full_name.split('_')[0]
            obj_name = obj_full_name[len(dset) + 1:]
            # Get metadata for new object and set in grasp_client
            object_metadata = metadata_handler.get_object_metadata(dset, obj_name)
            grasp_client.update_object_metadata(object_metadata)

            # grasp_client.create_dirs_new_grasp_trial(
            #     is_new_pose_or_object=True)  # TODO: here True is not always true?

            grasp_client.remove_obstacle_objects(obstacle_objects, moveit=False)

            # Spawn object in random position and orientation. NOTE: currently this will only spawn the objects upright with random z orientation
            array_object_mesh_frame_world = utils.get_array_from_rot_quat_list(
                object_mesh_frame_world)
            grasp_client.spawn_object(
                pose_type='replicate', pose_arr=array_object_mesh_frame_world)

            # grasp_client.set_path_and_save_visual_data(grasp_phase="single")

            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.segment_object_client(down_sample_pcd=True)

            ###############################################
            # Segment object and save visual data
            # grasp_client.set_path_and_save_visual_data(
            #     grasp_phase='single', object_pcd_record_path=single_pcd_save_path)
            # grasp_client.segment_object_client(
            #     down_sample_pcd=False, need_to_transfer_pcd_to_world_frame=True)
            # print('time to segment single object,', time() - time2)

            obstacle1_pose = utils.hom_matrix_from_pos_quat_list(obstacle1_mesh_frame_world)
            obstacle2_pose = utils.hom_matrix_from_pos_quat_list(obstacle2_mesh_frame_world)
            obstacle3_pose = utils.hom_matrix_from_pos_quat_list(obstacle3_mesh_frame_world)

            obstacle1_pose_stamped = utils.pose_stamped_from_hom_matrix(
                obstacle1_pose, frame_id='mesh_frame_pose')
            obstacle2_pose_stamped = utils.pose_stamped_from_hom_matrix(
                obstacle2_pose, frame_id='mesh_frame_pose')
            obstacle3_pose_stamped = utils.pose_stamped_from_hom_matrix(
                obstacle3_pose, frame_id='mesh_frame_pose')
            for obj in obstacle_objects:
                if obj['name'] == obstacle1_name:
                    assign_obstacle_objects_pose(obj, obstacle1_pose_stamped)
                elif obj['name'] == obstacle2_name:
                    assign_obstacle_objects_pose(obj, obstacle2_pose_stamped)
                elif obj['name'] == obstacle3_name:
                    assign_obstacle_objects_pose(obj, obstacle3_pose_stamped)
                else:
                    raise ValueError("obj name not found", obj['name'])

            grasp_client.spawn_obstacle_objects(obstacle_objects, moveit=True)

            # visualize the hand in the scene
            target_obj_pose = grasp_client.get_grasp_object_pose_client()
            obstacle_obj_poses = grasp_client.get_obstacle_objects_poses(obstacle_objects)

            collision_count = 0
            for grasp_id in test_data.keys():

                # TODO: check if objects are moved if so respawn

                print("verify grasp of", grasp_id)
                collision_grasp = test_data[grasp_id]

                # get grasp palm pose in worl frame
                palm_mesh_frame = collision_grasp['desired_preshape_palm_mesh_frame'][()]

                palm_mesh_frame_mat = utils.hom_matrix_from_pos_quat_list(palm_mesh_frame)
                palm_world_frame_mat = np.matmul(object_mesh_frame_world_mat, palm_mesh_frame_mat)
                palm_world_frame_stamp = utils.pose_stamped_from_hom_matrix(
                    palm_world_frame_mat, 'world')

                palm_world_arr = utils.get_pose_array_from_stamped(palm_world_frame_stamp)
                grasp_client.spawn_hand(palm_world_arr)
                rospy.sleep(0.5)
                home_folder = os.path.expanduser('~')
                coll_flag_path = os.path.join(home_folder,'collision_flag.pickle')
                with open(coll_flag_path, 'rb') as file:
                    b = pickle.load(file)
                    print(b)
                if b == True:
                    collision_count += 1
                    print(collision_count)
                # get_contact()
                # result = check_collision()
                # print(result)

                grasp_client.delete_hand()

            print('found collisions of : ', collision_count)
            print(1.0*collision_count/len(test_data.keys()))
