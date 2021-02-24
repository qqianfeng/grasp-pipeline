""" The script takes as input the generated grasp data and generates same amount of Utah format training data as in 
https://arxiv.org/pdf/2001.09242.pdf

Side:     +1559 -2616
Overhead: +749  -4064
"""
import os
import copy

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandler
from grasp_pipeline.utils.metadata_handler import MetadataHandler
import grasp_pipeline.utils.utils as utils

if __name__ == '__main__':

    # grasp data file path
    file_path = os.path.join('/home/vm', 'grasp_data.h5')

    grasp_client = GraspClient(is_rec_sess=False)
    data_handler = GraspDataHandler(file_path=file_path, sess_name='recording_session_0001')
    metadata_handler = MetadataHandler()

    # Get all available objects and choose one
    objects = data_handler.get_objects_list()
    full_object_name = objects[-1]
    object_name = '_'.join(objects[-1].split('_')[1:])
    dataset_name = objects[-1].split('_')[0]

    # Get metadata for this object
    object_metadata = metadata_handler.get_object_metadata(dataset_name=dataset_name,
                                                           object_name=object_name)

    # Set metadata in Grasp Client
    grasp_client.update_object_metadata(object_metadata)

    # Spawn random object in random pose:
    grasp_client.spawn_object(pose_type='random')

    # Segment object and publish object-centric frame (also gets the dim_w_h_d)
    grasp_client.save_visual_data_and_segment_object()

    # TODO: Change .*_world_pose to *_mesh_frame
    # Grasp_data.keys() = [u'is_top_grasp', u'lifted_joint_state', u'desired_preshape_joint_state', u'desired_preshape_palm_world_pose', 'object_name', u'true_preshape_joint_state', u'closed_joint_state', u'object_world_sim_pose', u'time_stamp', u'true_preshape_palm_world_pose', u'grasp_success_label']

    is_valid_pose = False
    while not is_valid_pose:
        # get all the data on the grasp
        grasp_data = data_handler.get_single_successful_grasp(full_object_name, random=True)

        # Take the 6D palm position and the mesh frame during data generation
        palm_pose = grasp_data["true_preshape_palm_world_pose"]  # w.r.t to mesh frame NOT world
        object_mesh_frame_data_gen = grasp_data["object_world_sim_pose"]

        # Transform the palm and object mesh frame pose from a list to a pose stamped
        palm_pose_mf_dg_stamped = utils.get_pose_stamped_from_rot_quat_list(
            palm_pose, frame_id="object_mesh_frame_data_gen")
        obj_mf_dg_stamped = utils.get_pose_stamped_from_rot_quat_list(object_mesh_frame_data_gen)

        # Publish the object mesh frame as it was during data generation TODO: Change object_world_sim_pose to object_mesh_frame_world
        grasp_client.update_object_mesh_frame_data_gen_client(obj_mf_dg_stamped)

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

        # Check if the pose is valid in UTAH world (from front, side or top of current object)
        if grasp_data["is_top_grasp"]:
            is_valid_pose = False
        else:
            is_valid_pose = grasp_client.check_pose_validity_utah(palm_pose_world)

    # Transform the valid grasp pose from current mesh frame to object-centric frame which is the format in which UTAH stores it
    palm_pose_obj_aligned = grasp_client.transform_pose(palm_pose_curr_mf,
                                                        from_frame='object_mesh_frame',
                                                        to_frame='object_pose_aligned')

    # Generate voxel grid from pcd
    grasp_client.generate_voxel_from_pcd_client()

    # Combine the true hand preshape 6D pose and true joint state into a 16-dim array
    grasp_config_obj = utils.get_utah_grasp_config_from_pose_and_joints(
        palm_pose_obj_aligned, grasp_data["true_preshape_joint_state"])

    # Lastly use the utah server to store info
    grasp_id = 1
    grasp_client.record_sim_grasp_data_utah_client(grasp_id, grasp_data["object_name"],
                                                   grasp_config_obj, grasp_data["is_top_grasp"],
                                                   grasp_data["grasp_success_label"])
