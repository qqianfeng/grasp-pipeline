#!/usr/bin/env python
from pickletools import anyobject

import rospy
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
import os
import time
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler
import random
import numpy as np

poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
         [0.5, 0.0, 0.2, 0, 0, -1.571]]

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

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):
        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object(case="generation")
        grasp_client.update_object_metadata(object_metadata)
        for pose_idx, pose in enumerate(poses):
            object_cycle_start = time.time()
            start = object_cycle_start

            obstacle_objects = get_obstacle_objects(gazebo_objects_path, grasp_client.object_metadata)
            obstacle_objects = distribute_obstacle_objects_randomly(pose, obstacle_objects)

            # Create dirs
            grasp_client.create_dirs_new_grasp_trial_multi_obj(pose_idx, is_new_pose_or_object=True) # TODO: here True is not always true?

            # Reset panda and hithand
            grasp_client.reset_hithand_and_panda()
            grasp_client.remove_obstacle_objects(obstacle_objects)
            grasp_client.clean_moveit_scene_client()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            grasp_client.spawn_object(pose_type="init", pose_arr=pose)

            grasp_client.set_path_and_save_visual_data(grasp_phase="single")
            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.segment_object_client(down_sample_pcd=True)

            rospy.logdebug("grasp_client.spawn_obstacle_objects(obstacle_objects)")
            grasp_client.spawn_obstacle_objects(obstacle_objects)

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_client.get_valid_preshape_for_all_points()

            grasp_client.set_path_and_save_visual_data(grasp_phase="pre")

            j = 0
            rospy.logdebug("start while loop to execute all sampled grasp poses")
            while grasp_client.grasps_available:

                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_client.create_dirs_new_grasp_trial_multi_obj(pose_idx, is_new_pose_or_object=False)

                    # Reset panda and hithand
                    grasp_client.reset_hithand_and_panda()

                    # Spawn object in same position
                    grasp_client.spawn_object(pose_type="same")
                    rospy.logdebug("grasp_client.spawn_object(pose_type='same')")

                    grasp_client.reset_obstacle_objects(obstacle_objects)
                    rospy.logdebug("grasp_client.reset_obstacle_objects(obstacle_objects)")

                # Grasp and lift object
                execution_success = grasp_client.grasp_and_lift_object(obstacle_objects)

                # Save all grasp data including post grasp images
                grasp_client.save_visual_data_and_record_grasp(obstacle_objects)

                # measure time
                print("One cycle took: " + str(time.time() - start))
                if execution_success:
                    j += 1

            # Finally write the time to file it took to test all poses
            grasp_client.log_object_cycle_time(time.time() - object_cycle_start)

            grasp_client.remove_obstacle_objects_from_moveit_scene()
