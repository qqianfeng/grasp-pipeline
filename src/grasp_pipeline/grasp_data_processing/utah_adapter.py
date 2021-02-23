""" The script takes as input the generated grasp data and generates same amount of Utah format training data as in 
https://arxiv.org/pdf/2001.09242.pdf

Side:     +1559 -2616
Overhead: +749  -4064
"""
import os
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandler
from grasp_pipeline.utils.metadata_handler import MetadataHandler

if __name__ == '__main__':

    # grasp data file path
    file_path = os.path.join('/home/vm', 'grasp_data.h5')

    grasp_client = GraspClient(is_rec_sess=False)
    data_handler = GraspDataHandler(file_path=file_path)
    metadata_handler = MetadataHandler()

    # Get all available objects and choose one
    objects = data_handler.get_objects_list()
    full_object_name = objects[-1]
    object_name = '_'.join(objects[-1].split('_')[1:])
    dataset_name = objects[-1].split('_')[0]

    # Get metadata for this object
    object_metadata = metadata_handler.get_object_metadata(dataset_name=dataset_name,
                                                           object_name=object_name)

    # Set metadata in Grasp Client
    grasp_client.update_object_metadata(object_metadata)

    # Spawn random object in random pose:
    grasp_client.spawn_object(pose_type='random')

    # Segment object and publish object-centric frame (also gets the dim_w_h_d)
    grasp_client.save_visual_data_and_segment_object()

    # Until valid pose:
    #       Choose a random valid pose from grasp_data.h5 file
    #       Reject sampled pose if pose is behind object or to the right side of object
    is_valid_pose = False
    while not is_valid_pose:
        grasp_pose = data_handler.get_single_successful_grasp(full_object_name, random=True)
        is_valid_pose = grasp_client.check_pose_validity_utah(grasp_pose)

    # Publish the object mesh frame pose
    grasp_client.update_object_mesh_frame_pose_client()

    # Tranform the valid grasp pose from object mesh frame to object-centric frame/pose
    grasp_pose_utah = grasp_client.transform_pose(grasp_pose,
                                                  from_frame='object_mesh_frame',
                                                  to_frame='object_pose_aligned')

    # Update the transformed pose and check validity in ros
    grasp_client.update_grasp_palm_pose_client(grasp_pose_utah)

    # Generate voxel grid from pcd
    grasp_client.generate_voxel_from_pcd_client()

    # Lastly use the utah server to store info
    grasp_id = 1
    grasp_client.recrod_sim_grasp_data_utah_client(grasp_id)