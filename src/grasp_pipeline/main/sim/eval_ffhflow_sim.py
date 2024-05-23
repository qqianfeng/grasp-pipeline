""" This file is used to evaluate the model and sample grasps.
"""
import numpy as np
import shutil
import torch
import rospy
import os
import csv

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_FOR_EVAL as obj_list
# from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_DATA_GEN_PAPER_VIDEO as obj_list

# Define parameters:
FILTER_THRESH = -1  # set to -1 if no filtering desired, default 0.9
FILTER_NUM_GRASPS = 20
NUM_TRIALS_PER_OBJ = 20
path2grasp_data = os.path.join(os.path.expanduser("~"), 'grasp_data')
object_datasets_folder = rospy.get_param('object_datasets_folder')
gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

shutil.rmtree(path2grasp_data, ignore_errors=True)
data_recording_path = rospy.get_param('data_recording_path')
grasp_client = GraspClient(grasp_data_recording_path=data_recording_path, is_rec_sess=True, is_eval_sess=True)
metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
with open('grasp_result.csv', 'wb') as file:
    writer = csv.writer(file)
    for obj_full in obj_list:

        dset, obj_name = metadata_handler.split_full_name(obj_full)

        # get metadata on object
        metadata = metadata_handler.get_object_metadata(dset, obj_name)
        grasp_client.update_object_metadata(metadata)

        # create new folder
        grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

        rospy.loginfo("Now start experiement of object: %s" % obj_name)
        total_trials = 0
        success_trials = 0
        for trial in range(NUM_TRIALS_PER_OBJ):
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
            palm_poses_obj_frame, joint_confs = grasp_client.infer_flow_grasp_poses(visualize_poses=True)

            # Evaluate the generated poses according to the FFHEvaluator
            palm_poses_obj_frame, joint_confs = grasp_client.evaluate_and_remove_grasps(
                palm_poses_obj_frame, joint_confs, thresh=FILTER_THRESH, visualize_poses=True)

            palm_poses_obj_frame = palm_poses_obj_frame[:FILTER_NUM_GRASPS]
            joint_confs = joint_confs[:FILTER_NUM_GRASPS]

            # Execute the grasps and record results
            rospy.loginfo("The {} time of trial out of {}".format(trial, NUM_TRIALS_PER_OBJ))
            is_skipped = False
            for i in range(FILTER_NUM_GRASPS):
                rospy.loginfo('start a new grasp')
                if i != 0:
                    grasp_client.reset_hithand_and_panda()

                    grasp_client.spawn_object(pose_type='same')
                # idx = np.random.randint(0, len(joint_confs))
                idx = i
                try:
                    grasp_executed, grasp_label = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[idx],
                                                                    joint_confs[idx])
                except IndexError:
                    print('palm_poses_obj_frame:', len(palm_poses_obj_frame))
                    print('joint_confs:', len(joint_confs))
                    print(idx)
                is_skipped = not grasp_executed
                if grasp_executed:
                    total_trials += 1
                    if grasp_label == 1:
                        success_trials += 1
                    print('current status of %s success grasps out of %s trials', success_trials, total_trials)
                    grasp_client.record_grasp_trial_data_client()
                    break

        writer.writerow([obj_name,success_trials,total_trials])