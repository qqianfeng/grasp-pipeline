""" This file is used to evaluate the model and sample grasps. 
"""
import numpy as np
import shutil
import torch
import rospy

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_FOR_EVAL as obj_list
# from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_DATA_GEN_PAPER_VIDEO as obj_list

N_POSES = 400
FILTER_THRESH = -1  # set to -1 if no filtering desired, default 0.9
FILTER_NUM_GRASPS = 20
NUM_TRIALS_PER_OBJ = 10

shutil.rmtree('/home/vm/grasp_data', ignore_errors=True)
grasp_client = GraspClient(grasp_data_recording_path='/tmp/', is_rec_sess=True, is_eval_sess=True)
metadata_handler = MetadataHandler()

for obj_full in obj_list:
    # Skip object
    # txt = "Skip object: {}? ".format(obj_full)
    # l = raw_input(txt)
    # if (l == "y") or (l == "Y"):
    #     print("Skipped")
    #     continue

    dset, obj_name = metadata_handler.split_full_name(obj_full)

    # get metadata on object
    metadata = metadata_handler.get_object_metadata(dset, obj_name)
    grasp_client.update_object_metadata(metadata)

    # create new folder
    grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

    rospy.loginfo("Now start experiement of object: %s" % obj_name)

    for trial in range(NUM_TRIALS_PER_OBJ - 1):
        # Reset
        grasp_client.reset_hithand_and_panda()

        # Spawn model
        # grasp_client.spawn_object(pose_type='init', pose_arr=[0.75, 0, 0, 0, 0, -2.57])
        grasp_client.spawn_object(pose_type="random")

        # Get point cloud (mean-free, orientation of camera frame)
        grasp_client.save_visual_data_and_segment_object(down_sample_pcd=False)

        # Compute BPS of point cloud, stores encoding to disk
        grasp_client.encode_pcd_with_bps()

        # Sample N latent variables and get the poses
        palm_poses_obj_frame, joint_confs = grasp_client.infer_grasp_poses(n_poses=N_POSES,
                                                                           visualize_poses=True)

        # Evaluate the generated poses according to the FFHEvaluator
        palm_poses_obj_frame, joint_confs = grasp_client.evaluate_and_remove_grasps(
            palm_poses_obj_frame, joint_confs, thresh=FILTER_THRESH, visualize_poses=True)
        palm_poses_obj_frame = palm_poses_obj_frame[:FILTER_NUM_GRASPS]
        joint_confs = joint_confs[:FILTER_NUM_GRASPS]

        # Execute the grasps and record results
        rospy.loginfo("The {} time of trial out of {}".format(trial, NUM_TRIALS_PER_OBJ))
        is_skipped = False
        for i in range(FILTER_NUM_GRASPS):
            if i != 0 and not is_skipped:
                grasp_client.reset_hithand_and_panda()

                grasp_client.spawn_object(pose_type='same')
            # idx = np.random.randint(0, len(joint_confs))
            idx = i
            # if palm_poses_obj_frame[idx].pose.position.y < -0.03:
            #     is_skipped = True
            #     continue
            # else:
            #     is_skipped = False
            # palm_poses_obj_frame[idx].pose.position.z += 0.015
            # palm_poses_obj_frame[idx].pose.position.y += 0.025
            # palm_poses_obj_frame[idx].pose.position.z -= 0.015
            # palm_poses_obj_frame[idx].pose.position.y -= 0.025

            grasp_executed = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[idx],
                                                                   joint_confs[idx])
            is_skipped = not grasp_executed
            if grasp_executed:
                grasp_client.record_grasp_trial_data_client()
                break
