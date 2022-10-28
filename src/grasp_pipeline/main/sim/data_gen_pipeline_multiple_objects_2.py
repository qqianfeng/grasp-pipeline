#!/usr/bin/env python
from pickletools import anyobject

import rospy
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
import os
import time
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler

poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
         [0.5, 0.0, 0.2, 0, 0, -1.571]]


def get_objects(gazebo_objects_path, amount=4):
    # TODO add to choose random object
    objects = []
    metadata_handler = MetadataHandler(gazebo_objects_path)
    num_total = metadata_handler.get_total_num_objects()
    amount = min(amount, num_total)

    for _ in range(amount):
        objects.append(metadata_handler.choose_next_grasp_object())
    return objects


if __name__ == '__main__':
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Remove these while testing
    #shutil.rmtree('/home/ffh/grasp_data', ignore_errors=True)

    # Create grasp client and metadata handler
    grasp_client = GraspClient(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)

    grasp_objects = get_objects(gazebo_objects_path)
    for idx, obj in enumerate(grasp_objects):
        grasp_objects[idx] = grasp_client.set_to_random_pose(obj)

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object()
        grasp_client.update_object_metadata(object_metadata)

        for pose in poses:
            object_cycle_start = time.time()
            start = object_cycle_start

            # Create dirs
            grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

            # Reset panda and hithand
            grasp_client.reset_hithand_and_panda()
            # grasp_client.reset_scene()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            grasp_client.spawn_object(pose_type="init", pose_arr=pose)
            
            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_client.segment_object_client(down_sample_pcd=True)

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_client.get_valid_preshape_for_all_points()

            # grasp_client.update_gazebo_object_client(grasp_objects)

            # TODO: grasp_objects has no attribute of "mesh_frame_pose"
            grasp_client.update_multiple_gazebo_objects_client(grasp_objects)

            grasp_client.save_visual_data()

            j = 0
            while grasp_client.grasps_available:
                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_client.create_dirs_new_grasp_trial(is_new_pose_or_object=False)

                    # Reset panda and hithand
                    grasp_client.reset_hithand_and_panda()

                    # TODO reset the scene
                    # grasp_client.reset_scene()
                    
                    # Spawn object in same position
                    grasp_client.spawn_object(pose_type="same")

                    grasp_client.update_multiple_gazebo_objects_client(grasp_objects)

                # Grasp and lift object
                grasp_arm_plan = grasp_client.grasp_and_lift_object()

                # Save all grasp data including post grasp images
                grasp_client.save_visual_data_and_record_grasp()

                # measure time
                print("One cycle took: " + str(time.time() - start))
                j += 1

            # Finally write the time to file it took to test all poses
            grasp_client.log_object_cycle_time(time.time() - object_cycle_start)
