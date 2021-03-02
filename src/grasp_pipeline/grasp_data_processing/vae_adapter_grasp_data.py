""" Small script to read in all the grasp-data-files in a directory and merges them into one file.
This file consists of: 
    - level 1:  a.) object names

    - level 2:  a.) metadata
                b.) positive
                c.) negative
                d.) collision
Storing all grasps object- and outcome-wise.
"""
import h5py
import os

RC = 'recording_sessions'
RC1 = 'recording_session_0001'
GT = 'grasp_trials'
G = 'grasps'
C = 'collision'
NC = 'no_collision'
GPS = 'grasp_success_label'

base_path = '/home/vm/data/exp_data'
dst_path = os.path.join(base_path, 'grasp_data_vae.h5')


def log_grasp(src_grasp_gp, dest_grasp_gp):
    true_joint_conf = src_grasp_gp["true_preshape_joint_state"][()]
    des_joint_conf = src_grasp_gp["desired_preshape_joint_state"][()]
    true_palm_mesh_frame = src_grasp_gp["true_preshape_palm_mesh_frame"][()]
    des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_mesh_frame"][()]

    dest_grasp_gp.create_dataset("true_preshape_joint_state", data=true_joint_conf)
    dest_grasp_gp.create_dataset("desired_preshape_joint_state", data=des_joint_conf)
    dest_grasp_gp.create_dataset("true_preshape_palm_mesh_frame", data=true_palm_mesh_frame)
    dest_grasp_gp.create_dataset("desired_preshape_palm_mesh_frame", data=des_palm_mesh_frame)


def create_grasp_group(group, idx):
    return group.create_group('grasp_' + str(idx).zfill(4))


# go through all the dirs, each dir contains one grasp_data.h5
for dir in os.listdir(base_path):
    src_path = os.path.join(base_path, dir, 'grasp_data.h5')

    hdf_src = h5py.File(src_path, 'r')
    hdf_dst = h5py.File(dst_path, 'a')

    objs_gp = hdf_src[RC][RC1][GT]
    for obj in objs_gp.keys():
        # Grasp idxs
        pos_idx = 0
        neg_idx = 0
        coll_idx = 0

        # Get the object_group in dest file
        if obj not in hdf_dst.keys():
            dst_obj_gp = hdf_dst.create_group(obj)
            dst_obj_gp.create_group('positive')
            dst_obj_gp.create_group('negative')
            dst_obj_gp.create_group('collision')
        else:
            dst_obj_gp = hdf_dst[obj]

        # Get the grasps from no collision gp from src_file
        no_coll_gp = objs_gp[obj][G][NC]
        for grasp in no_coll_gp.keys():
            src_grasp_gp = no_coll_gp[grasp]
            label = src_grasp_gp["grasp_success_label"][()]
            if label:
                dst_grasp_gp = create_grasp_group(dst_obj_gp['positive'], pos_idx)
                pos_idx += 1
            else:
                dst_grasp_gp = create_grasp_group(dst_obj_gp['negative'], neg_idx)
                neg_idx += 1
            log_grasp(src_grasp_gp, dst_grasp_gp)

        # Get the grasps from collision gp from src file
        coll_gp = objs_gp[obj][G][C]
        for grasp in coll_gp.keys():
            src_grasp_gp = coll_gp[grasp]
            dst_grasp_gp = create_grasp_group(dst_obj_gp['collision'], coll_idx)
            coll_idx += 1
