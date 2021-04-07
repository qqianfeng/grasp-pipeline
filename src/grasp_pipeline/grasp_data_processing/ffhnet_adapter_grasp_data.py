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
import pandas as pd

P = 'positive'
N = 'negative'
RC = 'recording_sessions'
RC1 = 'recording_session_0001'
GT = 'grasp_trials'
G = 'grasps'
C = 'collision'
NC = 'no_collision'
GPS = 'grasp_success_label'


def log_to_csv(src_file, save_path):
    data = {'names': [], 'positive': [], 'negative': [], 'collision': []}
    # Add three rows positive negative collision
    for obj in src_file.keys():
        # Init counter
        pos_cnt = 0
        neg_cnt = 0
        coll_cnt = 0

        obj_gp = src_file[obj]
        use_ix = False
        if obj in data['names']:
            ix = data['names'].index(obj)
            use_ix = True
        else:
            data['names'].append(obj)
        # Add new object as columns wit
        for grasp_type in obj_gp.keys():
            grasp_type_gp = obj_gp[grasp_type]
            for grasp in grasp_type_gp.keys():
                if grasp_type == 'positive':
                    pos_cnt += 1
                if grasp_type == 'negative':
                    neg_cnt += 1
                if grasp_type == 'collision':
                    coll_cnt += 1

            if use_ix:
                if grasp_type == 'positive':
                    data['positive'][ix] += pos_cnt
                if grasp_type == 'negative':
                    data['negative'][ix] += neg_cnt
                if grasp_type == 'collision':
                    data['collision'][ix] += coll_cnt
            else:
                if grasp_type == 'positive':
                    data['positive'].append(pos_cnt)
                if grasp_type == 'negative':
                    data['negative'].append(neg_cnt)
                if grasp_type == 'collision':
                    data['collision'].append(coll_cnt)

    # Save to disk
    obj_names = data['names']
    data.pop('names')
    df = pd.DataFrame(data)
    df.index = obj_names
    df.to_csv(save_path)


def log_grasp(src_grasp_gp, dest_grasp_gp, is_coll=False):
    """Logs a grasp to the new h5 file. NOTE: Now takes the joint state closed as true joint state, as desired joint
    state is always zero

    Args:
        src_grasp_gp (hdf group): Grasp group of a source file.
        dest_grasp_gp (hdf group): Grasp group of the destiation file
        is_coll (bool, optional): Indicated whether the grasp would. Defaults to False.

    Raises:
        Exception: If the grasp pose is not in the mesh frame.
    """
    if not is_coll:
        true_joint_conf = src_grasp_gp["closed_joint_state"][()]
        des_joint_conf = src_grasp_gp["desired_preshape_joint_state"][()]
        if "true_preshape_palm_mesh_frame" in src_grasp_gp.keys():
            true_palm_mesh_frame = src_grasp_gp["true_preshape_palm_mesh_frame"][()]
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_mesh_frame"][()]
        elif "true_preshape_palm_world_pose" in src_grasp_gp.keys():
            # Here I just had a naming issue. The preshape palm is not in world but in mesh frame
            true_palm_mesh_frame = src_grasp_gp["true_preshape_palm_world_pose"][()]
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_world_pose"][()]
        else:
            raise Exception("Something is wrong, not world_frame no mesh_frame")

        dest_grasp_gp.create_dataset("true_preshape_joint_state", data=true_joint_conf)
        dest_grasp_gp.create_dataset("desired_preshape_joint_state", data=des_joint_conf)
        dest_grasp_gp.create_dataset("true_preshape_palm_mesh_frame", data=true_palm_mesh_frame)
        dest_grasp_gp.create_dataset("desired_preshape_palm_mesh_frame", data=des_palm_mesh_frame)
    else:
        des_joint_conf = src_grasp_gp["desired_joint_state"][()]
        if "desired_palm_pose_mesh_frame" in src_grasp_gp.keys():
            des_palm_mesh_frame = src_grasp_gp["desired_palm_pose_mesh_frame"][()]
        elif "desired_preshape_palm_world_pose" in src_grasp_gp.keys():
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_world_pose"][()]
        else:
            raise Exception("Something is wrong, not world_frame no mesh_frame.")

        dest_grasp_gp.create_dataset("desired_preshape_joint_state", data=des_joint_conf)
        dest_grasp_gp.create_dataset("desired_preshape_palm_mesh_frame", data=des_palm_mesh_frame)


def create_grasp_group(group, idx):
    return group.create_group('grasp_' + str(idx).zfill(4))


def log_idxs(path, obj, pos, neg, coll):
    l = [obj, pos, neg, coll]
    if os.path.exists(path):
        string = ''
    else:
        string = 'obj_name \t \t \t pos \t neg \t coll \n'
    with open(path, 'w') as f:
        for i, txt in enumerate(l):
            string += str(txt) + '\t'
            if i == 0:
                string += '\t \t'

        string += '\n'
        f.write(string)


if __name__ == "__main__":

    base_path = '/home/vm/data/exp_data'
    dst_path = '/home/vm/data/ffhnet-data/ffhnet-grasp.h5'
    hdf_dst = h5py.File(dst_path, 'a')

    # go through all the dirs, each dir contains one grasp_data.h5
    for dir in sorted(os.listdir(base_path)):
        src_path = os.path.join(base_path, dir, 'grasp_data.h5')
        hdf_src = h5py.File(src_path, 'r')

        # Loop over all recording sessions, there might be multiple ones in one file
        for rc_n in hdf_src[RC].keys():
            src_objs_gp = hdf_src[RC][rc_n][GT]
            print 'All objects: ', src_objs_gp.keys()
            for obj in src_objs_gp.keys():
                # do not process ycb objects
                if obj.split('_')[0] == 'ycb':
                    print('Skipping object:', obj)
                    continue
                print('Processing object:', obj)
                # Grasp idxs
                pos_idx = 0
                neg_idx = 0
                coll_idx = 0

                # Get the object_group in dest file, create if does not exist
                if obj not in hdf_dst.keys():
                    dst_obj_gp = hdf_dst.create_group(obj)
                    dst_obj_gp.create_group('positive')
                    dst_obj_gp.create_group('negative')
                    dst_obj_gp.create_group('collision')
                else:
                    dst_obj_gp = hdf_dst[obj]
                    if dst_obj_gp[P].keys():
                        pos_idx = int(sorted(dst_obj_gp[P].keys())[-1].split('_')[-1]) + 1
                    if dst_obj_gp[N].keys():
                        neg_idx = int(sorted(dst_obj_gp[N].keys())[-1].split('_')[-1]) + 1
                    if dst_obj_gp[C].keys():
                        coll_idx = int(sorted(dst_obj_gp[C].keys())[-1].split('_')[-1]) + 1

                # Get the grasps from no collision gp from src_file
                no_coll_gp = src_objs_gp[obj][G][NC]
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
                src_coll_gp = src_objs_gp[obj][G][C]
                for grasp in src_coll_gp.keys():
                    src_grasp_gp = src_coll_gp[grasp]
                    dst_grasp_gp = create_grasp_group(dst_obj_gp['collision'], coll_idx)
                    coll_idx += 1
                    log_grasp(src_grasp_gp, dst_grasp_gp, is_coll=True)

                # Finally log the pos, neg coll idx to a txt file
                path = os.path.join(os.path.split(base_path)[0], 'ffhnet-data', 'obj_metadata.txt')
                log_idxs(path, obj, pos_idx, neg_idx, coll_idx)

    # Create pandas dataframe and log
    save_path = os.path.join(os.path.split(dst_path)[0], 'metadata.csv')
    log_to_csv(hdf_dst, save_path)

    # close files
    hdf_src.close()
    hdf_dst.close()
