""" This file is used to evaluate the model and sample grasps. 
"""
import torch

from grasp_pipeline.grasp_client.grasp_real_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler

n_poses = 300

grasp_client = GraspClient(is_rec_sess=False)
metadata_handler = MetadataHandler()

while True:
    # Reset
    #grasp_client.reset_hithand_and_panda()

    # Get point cloud (mean-free, orientation of camera frame)
    grasp_client.save_visual_data_and_segment_object(down_sample_pcd=False)

    # Compute BPS of point cloud, stores encoding to disk
    grasp_client.encode_pcd_with_bps()

    # Sample N latent variables and get the poses
    palm_poses_obj_frame, joint_confs = grasp_client.infer_grasp_poses(n_poses=n_poses,
                                                                       visualize_poses=True)

    # Show object and poses in o3d viewer
    grasp_client.show_grasps_o3d_viewer(palm_poses_obj_frame)

    # Execute the grasps and record results
    for i in range(n_poses):
        if i != 0:
            grasp_client.reset_hithand_and_panda()

        grasp_arm_plan = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[i],
                                                               joint_confs[i])
