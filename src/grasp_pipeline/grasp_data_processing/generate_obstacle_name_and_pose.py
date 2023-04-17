"""
Before running the multi obj data generation, we first randomly choose obstacle objects and their random poses and store them in file.
The reason is that if we generate random poses during data generation, the generation script can crash but we need fixed poses."""

import rospy
import os
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
import time
import numpy as np
import h5py
from grasp_pipeline.utils import utils
import random

target_obj_pose = [0.5, 0.0, 0.2, 0, 0, 0]

all_grasp_objects = []

def get_obstacle_objects(gazebo_objects_path, grasp_object, amount=3):
    """Randomly choose a number of obstacble objects

    Args:
        gazebo_objects_path (str):
        grasp_object (dict):
        amount (int, optional): number of obstacle objects. Defaults to 3.

    Raises:
        ValueError:

    Returns:
        objects (list): a list of chosen obstacle objects
    """
    global all_grasp_objects
    if len(all_grasp_objects) == 0:
        # initilize once
        metadata_handler = MetadataHandler(gazebo_objects_path)
        num_total = metadata_handler.get_total_num_objects()
        for _ in range(num_total):
            all_grasp_objects.append(metadata_handler.choose_next_grasp_object(case="generation"))

    objects = []
    num_total = len(all_grasp_objects)
    if num_total < 20:
        raise ValueError('There should be more than 20 objects in the dataset, however only '
                         + str(num_total) + ' is found.')
    amount = min(amount, num_total)
    for _ in range(amount):
        obj = random.choice(all_grasp_objects)
        while obj['name'] == grasp_object['name']:
            obj = random.choice(all_grasp_objects)
        rospy.loginfo("obstacle object: %s"% obj['name'])
        objects.append(obj)
    return objects


def distribute_obstacle_objects_randomly(grasp_object_pose, obstacle_objects, min_center_to_center_distance=0.1):
    """Assign random location to each obstacle objects. Location is defined within a certain space.

    Args:
        grasp_object_pose (_type_):
        obstacle_objects (list): a list of obstacle objects
        min_center_to_center_distance (float, optional): distance between object center. Describe the clutterness of the scene. Defaults to 0.1.

    Returns:
        objects (list): a list of chosen obstacle objects
    """
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


if __name__ == '__main__':
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Create grasp client and metadata handler
    grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)

    path_to_store = "/home/yb/Documents/obstacle_data.h5"
    
    test_list = []
    with h5py.File(path_to_store, 'a') as file:
        for i in range(metadata_handler.get_total_num_objects()):
            # Specify the object to be grasped, its pose, dataset, type, name etc.
            object_metadata = metadata_handler.choose_next_grasp_object(case="generation")
            # target_obj_group = file.create_group(object_metadata["name"])

            # grasp_client.update_object_metadata(object_metadata)

            # object_cycle_start = time.time()
            # start = object_cycle_start

            # obstacle_objects = get_obstacle_objects(gazebo_objects_path, grasp_client.object_metadata)
            # obstacle_objects = distribute_obstacle_objects_randomly(target_obj_pose, obstacle_objects)

            test_list.append(object_metadata["name"])
            # Data to store
            # for idx in range(len(obstacle_objects)):
            #     obstacle_obj_group = target_obj_group.create_group("obstacle_"+str(idx))
            #     obstacle_obj_group.create_dataset('name',
            #                         data=obstacle_objects[idx]["name"])
            #     pose_list = [obstacle_objects[idx]['mesh_frame_pose'].pose.position.x,
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.position.y,
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.position.z,
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.orientation.x, 
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.orientation.y, 
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.orientation.z,
            #                 obstacle_objects[idx]['mesh_frame_pose'].pose.orientation.w]
            #     # Convert xya+quat pose into tranf matrix
            #     obs_obj_pose_mat = utils.hom_matrix_from_pos_quat_list(pose_list)
            #     tar_obj_pose_mat = utils.hom_matrix_from_pos_quat_list(target_obj_pose)
            #     # world_T_obs_obj_pose = world_T_tar_obj_pose x target_obj_T_obs_obj
            #     # inv(world_T_tar_obj_pose) x world_T_obs_obj_pose = target_obj_T_obs_obj
            #     target_obj_T_obs_obj = np.matmul(np.linalg.inv(tar_obj_pose_mat),obs_obj_pose_mat)
            #     # calculate the relation pose of obstacle obj cooresponding to target object
            #     obstacle_obj_group.create_dataset('obstacle_pose_in_target_frame',
            #                         data=target_obj_T_obs_obj)

    print(test_list)
