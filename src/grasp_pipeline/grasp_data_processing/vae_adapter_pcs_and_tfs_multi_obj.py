""" The idea of this is to create a dataset for training the grasp VAE. Different objects are being spawned sequentially
in random poses. They are being segmented and the object-centric frame in camera coords is extracted and saved. Also the mesh frame of the object is being stored.
"""
import os
import copy
import rospy
import numpy as np
import h5py

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandler
import grasp_pipeline.utils.utils as utils
from grasp_pipeline.utils.object_names_in_datasets import *


def mkdir(base_folder, folder_name=None):
    path = os.path.join(base_folder, folder_name) if folder_name is not None else base_folder
    if not os.path.exists(path):
        os.mkdir(path)


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
    palm_pose_world = grasp_client.transform_pose(palm_pose_curr_mf,
                                                  from_frame='object_mesh_frame',
                                                  to_frame='world')

    # Update the transformed pose and check validity in ros
    grasp_client.update_grasp_palm_pose_client(palm_pose_world)

    # Lookup transform between current mesh frame and object_centroid_vae
    transform_cent_mf = grasp_client.tf_buffer.lookup_transform("object_centroid_vae",
                                                                "object_mesh_frame",
                                                                rospy.Time(0),
                                                                timeout=rospy.Duration(10))
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
    grasp_cent_stamped = utils.pose_stamped_from_hom_matrix(grasp_cent,
                                                            frame_id='object_centroid_vae')

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
        hdf[obj_full].create_dataset(obj_full_pcd + '_mesh_to_centroid', data=tf_list)


if __name__ == '__main__':
    # Some "hyperparameters"
    n_pcds_per_obj = 50
    input_grasp_data_file = '/home/yb/multi_grasp_data/grasp_data_all.h5'

    # Get all available objects and choose one
    with h5py.File(input_grasp_data_file, 'r') as hdf:
        objects = hdf.keys()

    # Make the base directory
    dest_folder = '/home/yb/multi_grasp_data/'
    pcds_folder = os.path.join(dest_folder, 'point_clouds')
    pcd_tfs_path = os.path.join(dest_folder, 'pcd_transforms.h5')
    mkdir(pcds_folder)

    # Instantiate grasp client
    grasp_client = GraspClient(is_rec_sess=False)
    metadata_handler = MetadataHandler()

    # Iterate over all objects
    for obj_full in objects:
        obj = '_'.join(obj_full.split('_')[1:])
        dset = obj_full.split('_')[0]
        # Create directory for new object
        object_folder = os.path.join(pcds_folder, obj_full)
        mkdir(object_folder)

        #Create group for new object
        create_object_group_in_h5_file(obj_full, pcd_tfs_path)

        # Get metadata for new object and set in grasp_client
        object_metadata = metadata_handler.get_object_metadata(dset, obj)
        grasp_client.update_object_metadata(object_metadata)

        for i in xrange(n_pcds_per_obj):
            # Setup the save path for next pcd
            num_str = str(i).zfill(3)
            obj_full_pcd = obj_full + '_pcd' + num_str
            pcd_save_path = os.path.join(object_folder, obj_full_pcd + '.pcd')

            # Spawn object in random position and orientation. NOTE: currently this will only spawn the objects upright with random z orientation
            grasp_client.spawn_object(pose_type='random')

            ######## TODO: get transformation between new object pose and old object pose #######

            ###############################################################################

            grasp_client.save_visual_data_multi_obj(grasp_phase="single",object_pcd_record_path=pcd_save_path)
            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene

            rospy.logdebug("grasp_client.spawn_obstacle_objects(obstacle_objects)")
            ######### TODO create obstacle objects #########
            # with correct name and pose
            
            ################################################
            grasp_client.spawn_obstacle_objects(obstacle_objects)

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_client.segment_object_client(down_sample_pcd=True)
            grasp_client.save_visual_data_multi_obj(grasp_phase="pre",object_pcd_record_path=pcd_save_path)


            ## TASK find the transformations to apply to the ground truth grasp to transfer it to object_frame and verify that the transformation is correct in RVIZ
            #test_grasp_pose_transform(dset_obj_name=obj_full, grasp_client=grasp_client)

            # Lookup transform between current mesh frame and object_centroid_vae
            transform_cent_mf = grasp_client.tf_buffer.lookup_transform("object_centroid_vae",
                                                                        "object_mesh_frame",
                                                                        rospy.Time(0),
                                                                        timeout=rospy.Duration(10))

            # Bring the transform to list format
            trans_cent_mf_list = utils.trans_rot_list_from_ros_transform(transform_cent_mf)

            # Save the transform to file
            save_mesh_frame_centroid_tf(obj_full, pcd_tfs_path, obj_full_pcd, trans_cent_mf_list)
