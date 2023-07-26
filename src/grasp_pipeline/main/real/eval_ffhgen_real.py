""" This file is used to evaluate the model and sample grasps. 
"""
import torch

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler

n_poses = 10

grasp_client = GraspClient(is_rec_sess=False)
metadata_handler = MetadataHandler()

i = 1
while i <= 1:
    # Reset
    # grasp_client.reset_hithand_and_panda()

    # Get point cloud (mean-free, orientation of camera frame)
    grasp_client.save_visual_data_and_segment_object(down_sample_pcd=False)

    # Compute BPS of point cloud, stores encoding to disk
    grasp_client.encode_pcd_with_bps()

    # Sample N latent variables and get the poses
    palm_poses_obj_frame, joint_confs = grasp_client.infer_grasp_poses(n_poses=n_poses,
                                                                       visualize_poses=True)
    print("palm_poses_obj_frame", palm_poses_obj_frame)
    print("joint_confs", joint_confs)

    palm_poses, joint_confs = grasp_client.evaluate_and_filter_grasp_poses_client(
        palm_poses_obj_frame, joint_confs, thresh=0.5)

    # TODO: Why we can execute grasps for real world experiemnts with unknown objects?
    # Execute the grasps and record results
    # for i in range(n_poses):
    #     if i != 0:
    #         grasp_client.reset_hithand_and_panda()

    #         grasp_client.spawn_object(pose_type='same')

    #     grasp_arm_plan = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[i],
    #    joint_confs[i])
    i += 1
