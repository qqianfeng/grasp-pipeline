#!/usr/bin/env python
import rospy
from grasp_pipeline.grasp_client.grasp_sim_client_robotiq import *
import os
import time
import shutil
from grasp_pipeline.utils.metadata_handler import MetadataHandler

poses = [[0.5, 0.0, 0.2, 0, 0, 0], [0.5, 0.0, 0.2, 0, 0, 1.571], [0.5, 0.0, 0.2, 0, 0, 3.14],
         [0.5, 0.0, 0.2, 0, 0, -1.571]]

if __name__ == '__main__':
    # Some relevant variables
    data_recording_path = rospy.get_param('data_recording_path')
    object_datasets_folder = rospy.get_param('object_datasets_folder')
    gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

    # Intput a grasp mode for the robotiq
    robotiq_grasp_mode = raw_input('Choose a grasp mode for the robotiq: ')

    # Create grasp client and metadata handler
    grasp_data_rec = VisualGraspDataHandler(is_rec_sess=True, grasp_data_recording_path=data_recording_path)
    metadata_handler = MetadataHandler(gazebo_objects_path=gazebo_objects_path)
    grasp_sampler = GraspSampler()
    grasp_controller = GraspController(grasp_mode = robotiq_grasp_mode)
    object_spawner = ObjectSpawner(is_rec_sess=True)
    grasp_sampler.get_transform_pose_func(func=grasp_data_rec.transform_pose )
    grasp_controller.define_tf_buffer(grasp_data_rec.tf_buffer)
    rospy.sleep(1)

    # This loop runs for all objects, 4 poses, and evaluates N grasps per pose
    for i in range(metadata_handler.get_total_num_objects()):

        # Specify the object to be grasped, its pose, dataset, type, name etc.
        object_metadata = metadata_handler.choose_next_grasp_object()

        # update object metadata for all classes that require it
        for class_inst in [grasp_data_rec, grasp_sampler, object_spawner, grasp_controller]:
            class_inst.update_object_metadata(object_metadata)
        
        # Loop over all 4 poses:
        for k, pose in enumerate(poses):
            print('trying pose '+str(k), pose )
            # start timer
            object_cycle_start = time.time()
            start = object_cycle_start

            # Create dirs
            grasp_data_rec.create_dirs_new_grasp_trial(is_new_pose_or_object=True)

            # Reset panda and hithand
            grasp_controller.reset_robotiq_and_panda()

            # Spawn a new object in Gazebo and moveit in a random valid pose and delete the old object
            object_spawner.spawn_object(pose_type="init", pose_arr=pose)

            # First take a shot of the scene and store RGB, depth and point cloud to disk
            # Then segment the object point cloud from the rest of the scene
            grasp_data_rec.save_visual_data_and_segment_object()

            # Generate hithand preshape, this is crucial. Samples multiple heuristics-based hithand preshapes, stores it in an instance variable
            # Also one specific desired grasp preshape should be chosen. This preshape (characterized by the palm position, hithand joint states, and the is_top boolean gets stored in other instance variables)
            grasp_sampler.get_valid_preshape_for_all_points(object_segment_response = grasp_data_rec.object_segment_response)
            j = 0
            while grasp_sampler.grasps_available:
                print('Grasp # '+str(j))
                # Save pre grasp visual data
                if j != 0:
                    # Measure time
                    start = time.time()

                    # Create dirs
                    grasp_data_rec.create_dirs_new_grasp_trial(is_new_pose_or_object=False)
                    grasp_sampler.get_object_segment_response(grasp_data_rec.object_segment_response)

                    # Reset panda and hithand
                    grasp_controller.reset_robotiq_and_panda()

                    # Spawn object in same position
                    object_spawner.spawn_object(pose_type="same")

                grasp_controller.get_is_grasps_available(grasp_sampler.grasps_available)
                grasp_controller.get_heuristic_preshapes(grasp_sampler.heuristic_preshapes)
                grasp_controller.get_num_preshapes(grasp_sampler.num_preshapes)
                grasp_controller.get_save_only_depth_and_color_func(grasp_data_rec.save_only_depth_and_color)
                grasp_controller.get_pos_idxs(
                    grasp_sampler.top_idxs,
                    grasp_sampler.side1_idxs,
                    grasp_sampler.side2_idxs)

                # Grasp and lift object
                grasp_arm_plan = grasp_controller.grasp_and_lift_object()
                # Update grasps available if all grasp idxs used 
                grasp_sampler.update_grasps_available(grasp_controller.grasps_available)

                # Save all grasp data including post grasp images
                grasp_data_rec.get_palm_poses(grasp_controller.palm_poses)
                grasp_data_rec.get_post_grasp_data(
                    grasp_controller.chosen_is_top_grasp,
                    grasp_controller.grasp_label,
                    grasp_controller.object_metadata,
                    grasp_controller.hand_joint_states
                )
                grasp_data_rec.save_visual_data_and_record_grasp()

                # measure time
                print("One cycle took: " + str(time.time() - start))
                j += 1
                # only for shorter test:
                # if j >= 2 :
                #     grasp_sampler.grasps_available = False
                #     # break
            # Finally write the time to file it took to test all poses
            grasp_data_rec.log_object_cycle_time(time.time() - object_cycle_start)
