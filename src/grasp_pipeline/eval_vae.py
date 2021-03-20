""" This file is used to evaluate the model and sample grasps. 
"""
import torch

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler

objs_list = ['kit_InstantSauce']

grasp_client = GraspClient(is_rec_sess=False)
metadata_handler = MetadataHandler()

for obj_full in objs_list:
    split = obj_full.split('_')[0]
    dset = split[0]
    obj_name = '_'.join(split[1:])

    # get metadata on object
    metadata = metadata_handler.get_object_metadata(dset, obj_name)
    grasp_client.update_object_metadata(metadata)

    # Spawn model
    grasp_client.spawn_object(pose_type='random')

    # Get point cloud (mean-free, orientation of camera frame)
    grasp_client.save_visual_data_and_segment_object(down_sample_pcd=False)

    # Compute BPS of point cloud, stores encoding to disk
    grasp_client.encode_pcd_with_bps()

    # Build the model

    # Sample N latent variables and get the poses

    # Execute the grasps and record results