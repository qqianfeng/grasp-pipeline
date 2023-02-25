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
FILTER_NUM_GRASPS = 5
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

def prepare_object_metadata(name, grasp_client, obstacle_objects):
    global object_metadata_buffer
    if object_metadata_buffer is not None:
        raise RuntimeError('please call recover_object_metadata first') 
    if name == grasp_client.object_metadata['name']:
        return
    for obj in obstacle_objects:
        if obj['name'] == name:
            _replace_object_metadata(grasp_client, obj)

def recover_object_metadata(grasp_client):
    global object_metadata_buffer
    if object_metadata_buffer is None:
        return
    grasp_client.object_metadata = object_metadata_buffer
    object_metadata_buffer = None 

def _replace_object_metadata(grasp_client, object_metadata):
    global object_metadata_buffer
    object_metadata_buffer = grasp_client.object_metadata
    grasp_client.object_metadata = object_metadata

for obj_full in obj_list:
    dset, obj_name = metadata_handler.split_full_name(obj_full)
    metadata = metadata_handler.get_object_metadata(dset, obj_name)
    grasp_client.update_object_metadata(metadata)
    obj_name = grasp_client.object_metadata['name']
    grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=True)
    
    grasp_client.reset_hithand_and_panda()
    grasp_client.spawn_object(pose_type="random")
    grasp_object_pose = grasp_client.object_metadata['mesh_frame_pose']
    grasp_object_pose = np.array([grasp_object_pose.pose.position.x, 
                                    grasp_object_pose.pose.position.y, 
                                    grasp_object_pose.pose.position.z])
    obstacle_objects = get_obstacle_objects(grasp_client.object_metadata, NUM_OBSTACLE_OBJECTS)
    obstacle_objects = distribute_obstacle_objects_randomly(grasp_object_pose, obstacle_objects)
    grasp_client.remove_obstacle_objects(obstacle_objects)
    grasp_client.spawn_obstacle_objects(obstacle_objects, moveit=False)
    ROIs, names = grasp_client.select_ROIs(obstacle_objects)

    for ROI, name in zip(ROIs, names):
        print('grasping', name)
        # Reset
        is_grasp_object_visiable = True
        prepare_object_metadata(name, grasp_client, obstacle_objects)

        # Get point cloud (mean-free, orientation of camera frame)
        grasp_client.save_visual_data(down_sample_pcd=False)
        grasp_client.remove_ground_plane()
        grasp_client.segment_object_as_point_cloud(ROI) # outputs segmented object to self.object_pcd_save_path
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
        for i in range(FILTER_NUM_GRASPS):
            grasp_executed = grasp_client.grasp_from_inferred_pose(palm_poses_obj_frame[i], joint_confs[i])
            if grasp_executed:
                grasp_client.record_grasp_trial_data_client()
                grasp_client.reset_hithand_and_panda()
                if is_grasp_object_visiable:
                    recover_object_metadata(grasp_client)
                    grasp_client.spawn_object(pose_type='same')
                    prepare_object_metadata(name, grasp_client, obstacle_objects)

                grasp_client.reset_obstacle_objects(obstacle_objects)

            if grasp_client.grasp_label == 1:
                grasp_client.change_model_visibility(name, False)
                if name == obj_name:
                    is_grasp_object_visiable = False
                break

        grasp_client.reset_hithand_and_panda()
        if is_grasp_object_visiable:
            recover_object_metadata(grasp_client)
            grasp_client.spawn_object(pose_type='same')
            prepare_object_metadata(name, grasp_client, obstacle_objects)
        
        recover_object_metadata(grasp_client)
        grasp_client.reset_obstacle_objects(obstacle_objects)
        
        if grasp_client.grasp_label != 1:
            # skip the rest if one of the objects could not be picked up after 
            # several trials
            break

    grasp_client.remove_obstacle_objects(obstacle_objects)
