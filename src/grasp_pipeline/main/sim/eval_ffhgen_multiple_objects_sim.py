""" This file is used to evaluate the model and sample grasps. 
"""
import numpy as np
import shutil
import torch
import rospy
import os
import random

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_FOR_EVAL as obj_list
# from grasp_pipeline.utils.object_names_in_datasets import OBJECTS_DATA_GEN_PAPER_VIDEO as obj_list



# Define parameters:
N_POSES = 400
FILTER_THRESH = -1  # set to -1 if no filtering desired, default 0.9
FILTER_NUM_GRASPS = 20
NUM_TRIALS_PER_OBJ = 20
NUM_OBSTACLE_OBJECTS = 3
path2grasp_data = os.path.join(os.path.expanduser("~"), 'grasp_data')
object_datasets_folder = rospy.get_param('object_datasets_folder')
gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

shutil.rmtree(path2grasp_data, ignore_errors=True)
data_recording_path = rospy.get_param('data_recording_path')
grasp_client = GraspClient(grasp_data_recording_path=data_recording_path, is_rec_sess=True, is_eval_sess=True)
metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
object_metadata_buffer = None
all_grasp_objects = []

def get_obstacle_objects(grasp_object, amount=3):
    global all_grasp_objects
    if len(all_grasp_objects) == 0:
        # initilize once
        metadata_handler = MetadataHandler(gazebo_objects_path)
        
        for obj_full in obj_list:
            dset, obj_name = metadata_handler.split_full_name(obj_full)

            # get metadata on object
            metadata = metadata_handler.get_object_metadata(dset, obj_name)
            all_grasp_objects.append(metadata)
    
    objects = []
    object_names = set()
    num_total = len(all_grasp_objects)
    if num_total < 10:
        raise ValueError('There should be more than 20 objects in the dataset, however only '
                         + str(num_total) + ' is found.')
    amount = min(amount, num_total)
    for _ in range(amount):
        obj = random.choice(all_grasp_objects)
        while obj['name'] == grasp_object['name'] or obj['name'] in object_names:
            obj = random.choice(all_grasp_objects)
        object_names.add(obj['name'])
        rospy.loginfo("obstacle object: %s"% obj['name'])
        objects.append(obj)
    return objects


def distribute_obstacle_objects_randomly(grasp_object_pose, obstacle_objects, min_center_to_center_distance=0.1):
    existing_object_positions = [np.array(grasp_object_pose)[:3]]
    for idx, obj in enumerate(obstacle_objects):
        obstacle_objects[idx] = grasp_client.set_to_random_pose(obj)
        position = np.array([obstacle_objects[idx]['mesh_frame_pose'].pose.position.x, obstacle_objects[idx]['mesh_frame_pose'].pose.position.y, obstacle_objects[idx]['mesh_frame_pose'].pose.position.z])
        while not all([np.linalg.norm(position[:2] - existing_position[:2]) > min_center_to_center_distance for existing_position in existing_object_positions]):
            obstacle_objects[idx] = grasp_client.set_to_random_pose(obj)
            position = np.array([obstacle_objects[idx]['mesh_frame_pose'].pose.position.x, 
                                 obstacle_objects[idx]['mesh_frame_pose'].pose.position.y, 
                                 obstacle_objects[idx]['mesh_frame_pose'].pose.position.z])
        existing_object_positions.append(position)
    return obstacle_objects

def replace_object_metadata(grasp_client, object_metadata):
    global object_metadata_buffer
    if object_metadata_buffer in not None:
        raise RuntimeError('Replace_object_metadata called twice before recover_object_metadata.')
    object_metadata_buffer = grasp_client.object_metadata
    grasp_client.object_metadata = object_metadata

def recover_object_metadata(grasp_client):
    global object_metadata_buffer
    grasp_client.object_metadata = object_metadata_buffer
    object_metadata_buffer = None 

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
    grasp_objecet_pose = [0.45, 0, 0, 0, 0, -2.57]
    rospy.loginfo("Now start experiement of object: %s" % obj_name)

    for trial in range(NUM_TRIALS_PER_OBJ):
        # Reset
        grasp_client.reset_hithand_and_panda()

        # Spawn model
        # grasp_client.spawn_object(pose_type='init', pose_arr=grasp_objecet_pose)
        grasp_client.spawn_object(pose_type="random")
        
        obstacle_objects = get_obstacle_objects(grasp_client.object_metadata, NUM_OBSTACLE_OBJECTS)
        grasp_object_pose = grasp_client.object_metadata['mesh_frame_pose']
        grasp_object_pose = np.array([grasp_object_pose.pose.position.x, 
                                      grasp_object_pose.pose.position.y, 
                                      grasp_object_pose.pose.position.z])
        obstacle_objects = distribute_obstacle_objects_randomly(grasp_object_pose, obstacle_objects)
        grasp_client.spawn_obstacle_objects(obstacle_objects)

        # Get point cloud (mean-free, orientation of camera frame)
        grasp_client.save_visual_data(down_sample_pcd=False)
        ROI = grasp_client.segment_object_as_point_cloud() # outputs segmented object to self.object_pcd_save_path
        name_of_object_in_ROI = grasp_client._get_name_of_objcet_in_ROI(ROI, obstacle_objects)
        
        grasp_client.post_process_object_point_cloud() # goes through the origional segmentation process to get object frame published

        # Compute BPS of point cloud, stores encoding to disk
        grasp_client.encode_pcd_with_bps()

        # Sample N latent variables and get the poses
        palm_poses_obj_frame, joint_confs = grasp_client.infer_grasp_poses(n_poses=N_POSES, visualize_poses=True)

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

                grasp_client.reset_obstacle_objects(obstacle_objects)

            # idx = np.random.randint(0, len(joint_confs))
            idx = i
            # if palm_poses_obj_frame[idx].pose.position.y < -0.03:
            #     is_skipped = True
            #     continue
            # else:
            #     is_skipped = False

            grasp_executed = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[idx], joint_confs[idx])
            is_skipped = not grasp_executed
            if grasp_executed:
                grasp_client.record_grasp_trial_data_client()
                break

        grasp_client.remove_obstacle_objects(obstacle_objects)
