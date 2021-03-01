""" The idea of this is to create a dataset for training the grasp VAE. Different objects are being spawned sequentially
in random poses. They are being segmented and the object-centric frame in camera coords is extracted and saved. Also the mesh frame of the object is being stored.
"""
import os
import copy

from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
from grasp_pipeline.utils.metadata_handler import MetadataHandler
import grasp_pipeline.utils.utils as utils
from grasp_pipeline.utils.object_names_in_datasets import *


def mkdir(base_folder, folder_name=None):
    path = os.path.join(base_folder, folder_name) if folder_name is not None else base_folder
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    # Some "hyperparameters"
    n_pcds_per_obj = 5
    objects = KIT_OBJECTS + BIGBIRD_OBJECTS + YCB_OBJECTS
    datasets = ['kit'] * len(KIT_OBJECTS) + \
        ['bigbird'] * len(BIGBIRD_OBJECTS) + ['ycb'] * len(YCB_OBJECTS)

    # Make the base directory
    dest_folder = '/home/vm/data/vae-grasp'
    pcds_folder = os.path.join(dest_folder, 'point_clouds')
    mkdir(pcds_folder)

    # Instantiate grasp client
    grasp_client = GraspClient(is_rec_sess=False)

    # Iterate over all objects
    for obj, dset in zip(objects, datasets):
        # Create directory for new object
        obj_full = dset + '_' + obj
        object_folder = os.path.join(pcds_folder, obj_full)
        mkdir(object_folder)

        for i in xrange(n_pcds_per_obj):
            # Setup the save path for next pcd
            num_str = str(i).zfill(3)
            obj_full_pcd = obj_full + '_pcd' + num_str
            pcd_save_path = os.path.join(object_folder, obj_full_pcd)

            # Spawn object in random position and orientation. NOTE: currently this will only spawn the objects upright with random z orientation
            grasp_client.spawn_object(pose_type='random')

            # Segment object and save visual data
            grasp_client.save_visual_data_and_segment_object(keep_object_in_camera_frame=True,
                                                             down_sample_pcd=False)
