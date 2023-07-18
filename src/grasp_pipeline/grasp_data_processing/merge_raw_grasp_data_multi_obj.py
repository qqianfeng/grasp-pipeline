""" Small script to read in all the grasp-data-files in a directory and merges them into one file.
This file consists of:
    - level 1:  a.) object names

    - level 2:  a.) metadata
                b.) positive
                c.) negative
                d.) collision
Storing all grasps object- and outcome-wise.


|-- grasp_data.h5
    |-- metadata # use g5['key'][()] to get value
        |-- 'datetime_recording_start', 'total_num_close_finger_collide_obstacle_objects', 'total_num_collision_to_approach_pose',
        |-- 'total_num_collision_to_grasp_pose', 'total_num_collisions', 'total_num_failures', 'total_num_grasp_pose_collide_obstacle_objects',
        |-- 'total_num_grasp_pose_collide_target_object', 'total_num_grasps', 'total_num_lift_motion_moved_obstacle_objects',
        |-- 'total_num_recordings', 'total_num_successes', 'total_num_tops'

    |-- recording_sessions
        |-- 'recording_session_0001'
            |-- 'metadata'
                |-- 'sess_num_close_finger_collide_obstacle_objects', 'sess_num_collision_to_approach_pose', 'sess_num_collision_to_grasp_pose', 'sess_num_collisions', 'sess_num_failures', 'sess_num_grasp_pose_collide_obstacle_objects', 'sess_num_grasp_pose_collide_target_object', 'sess_num_grasps', 'sess_num_lift_motion_moved_obstacle_objects', 'sess_num_successes', 'sess_num_tops', 'sess_start'
            |-- 'grasp_trials' # object names
                |-- 'bigbird_red_bull'
                    |-- 'metadata'
                        |-- 'object_num_collisions', 'object_num_failures', 'object_num_grasps', 'object_num_successes', 'object_num_tops'
                    |-- 'grasps'
                        |-- 'collision'
                            |-- 'collision_0001'
                                |-- 'desired_joint_state', 'desired_palm_pose_mesh_frame', 'object_mesh_frame_world'
                            |-- 'collision_0002'...
                        |-- 'no_ik'
                            |-- 'no_ik_0001'
                                |-- 'desired_joint_state', 'desired_palm_pose_mesh_frame', 'object_mesh_frame_world'

                        |-- 'no_collision'
                            |-- 'grasp_0001'
                                |-- 'close_finger_collide_obstacle_objects', 'closed_joint_state', 'collision_to_approach_pose',
                                |-- 'collision_to_grasp_pose', 'desired_preshape_joint_state', 'desired_preshape_palm_mesh_frame',
                                |-- 'grasp_pose_collide_obstacle_objects', 'grasp_pose_collide_target_object', 'grasp_success_label', 'is_top_grasp',
                                |-- 'lift_motion_moved_obstacle_objects', 'lifted_joint_state', 'object_mesh_frame_world', 'object_name',
                                |-- 'time_stamp', 'true_preshape_joint_state', 'true_preshape_palm_mesh_frame'
                            |-- 'grasp_0002', 'grasp_0003', 'grasp_0004'

                |-- 'bigbird_softsoap_gold'

        |-- 'recording_session_0002'
        |-- 'recording_session_0003'
        |-- ...

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
NIK = 'no_ik'
GPS = 'grasp_success_label'

# The label for 'non_collision_not_executed' is for the collision-only dataset where grasps are not executed so only labeled as either
# collision or non-collision.
def log_to_csv(src_file, save_path):
    data = {'names': [], 'positive': [], 'negative': [], 'collision': [], 'non_collision_not_executed': []}
    # Add three rows positive negative collision
    for obj in src_file.keys():
        # Init counter
        pos_cnt = 0
        neg_cnt = 0
        coll_cnt = 0
        non_coll_cnt = 0

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
                if grasp_type == 'non_collision_not_executed':
                    non_coll_cnt += 1

            if use_ix:
                if grasp_type == 'positive':
                    data['positive'][ix] += pos_cnt
                if grasp_type == 'negative':
                    data['negative'][ix] += neg_cnt
                if grasp_type == 'collision':
                    data['collision'][ix] += coll_cnt
                if grasp_type == 'non_collision_not_executed':
                    data['non_collision_not_executed'][ix] += non_coll_cnt
            else:
                if grasp_type == 'positive':
                    data['positive'].append(pos_cnt)
                if grasp_type == 'negative':
                    data['negative'].append(neg_cnt)
                if grasp_type == 'collision':
                    data['collision'].append(coll_cnt)
                if grasp_type == 'non_collision_not_executed':
                    data['non_collision_not_executed'].append(non_coll_cnt)
    # Save to disk
    obj_names = data['names']
    data.pop('names')
    df = pd.DataFrame(data)
    df.index = obj_names
    df.to_csv(save_path)


def log_grasp(src_grasp_gp, dest_grasp_gp, is_coll=False):
    if not is_coll:
        true_joint_conf = src_grasp_gp["true_preshape_joint_state"][()]
        des_joint_conf = src_grasp_gp["desired_preshape_joint_state"][()]
        if "true_preshape_palm_mesh_frame" in src_grasp_gp.keys():
            true_palm_mesh_frame = src_grasp_gp["true_preshape_palm_mesh_frame"][()]
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_mesh_frame"][()]
        elif "true_preshape_palm_world_pose" in src_grasp_gp.keys():
            true_palm_mesh_frame = src_grasp_gp["true_preshape_palm_world_pose"][()]
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_world_pose"][()]
        else:
            raise Exception("Something is wrong, not world_frame no mesh_frame")

        dest_grasp_gp.create_dataset("true_preshape_joint_state", data=true_joint_conf)
        dest_grasp_gp.create_dataset("desired_preshape_joint_state", data=des_joint_conf)
        dest_grasp_gp.create_dataset("true_preshape_palm_mesh_frame", data=true_palm_mesh_frame)
        dest_grasp_gp.create_dataset("desired_preshape_palm_mesh_frame", data=des_palm_mesh_frame)

        object_name = src_grasp_gp["object_name"][()]
        obstacle1_name = src_grasp_gp["obstacle1_name"][()]
        obstacle2_name = src_grasp_gp["obstacle2_name"][()]
        obstacle3_name = src_grasp_gp["obstacle3_name"][()]

        object_mesh_frame_world = src_grasp_gp["object_mesh_frame_world"][()]
        obstacle1_mesh_frame_world = src_grasp_gp["obstacle1_mesh_frame_world"][()]
        obstacle2_mesh_frame_world = src_grasp_gp["obstacle2_mesh_frame_world"][()]
        obstacle3_mesh_frame_world = src_grasp_gp["obstacle3_mesh_frame_world"][()]

        dest_grasp_gp.create_dataset("object_name", data=object_name)
        dest_grasp_gp.create_dataset("obstacle1_name", data=obstacle1_name)
        dest_grasp_gp.create_dataset("obstacle2_name", data=obstacle2_name)
        dest_grasp_gp.create_dataset("obstacle3_name", data=obstacle3_name)
        dest_grasp_gp.create_dataset("object_mesh_frame_world", data=object_mesh_frame_world)
        dest_grasp_gp.create_dataset("obstacle1_mesh_frame_world", data=obstacle1_mesh_frame_world)
        dest_grasp_gp.create_dataset("obstacle2_mesh_frame_world", data=obstacle2_mesh_frame_world)
        dest_grasp_gp.create_dataset("obstacle3_mesh_frame_world", data=obstacle3_mesh_frame_world)

    else:
        des_joint_conf = src_grasp_gp["desired_joint_state"][()]
        object_mesh_frame_world = src_grasp_gp["object_mesh_frame_world"][()]
        if "desired_palm_pose_mesh_frame" in src_grasp_gp.keys():
            des_palm_mesh_frame = src_grasp_gp["desired_palm_pose_mesh_frame"][()]
        elif "desired_preshape_palm_world_pose" in src_grasp_gp.keys():
            des_palm_mesh_frame = src_grasp_gp["desired_preshape_palm_world_pose"][()]
        else:
            raise Exception("Something is wrong, not world_frame no mesh_frame.")

        dest_grasp_gp.create_dataset("desired_preshape_joint_state", data=des_joint_conf)
        dest_grasp_gp.create_dataset("desired_preshape_palm_mesh_frame", data=des_palm_mesh_frame)
        dest_grasp_gp.create_dataset("object_mesh_frame_world", data=object_mesh_frame_world)

        # New data to record with collision
        object_name = src_grasp_gp["object_name"][()]
        obstacle1_name = src_grasp_gp["obstacle1_name"][()]
        obstacle2_name = src_grasp_gp["obstacle2_name"][()]
        obstacle3_name = src_grasp_gp["obstacle3_name"][()]

        obstacle1_mesh_frame_world = src_grasp_gp["obstacle1_mesh_frame_world"][()]
        obstacle2_mesh_frame_world = src_grasp_gp["obstacle2_mesh_frame_world"][()]
        obstacle3_mesh_frame_world = src_grasp_gp["obstacle3_mesh_frame_world"][()]

        dest_grasp_gp.create_dataset("object_name", data=object_name)
        dest_grasp_gp.create_dataset("obstacle1_name", data=obstacle1_name)
        dest_grasp_gp.create_dataset("obstacle2_name", data=obstacle2_name)
        dest_grasp_gp.create_dataset("obstacle3_name", data=obstacle3_name)
        dest_grasp_gp.create_dataset("obstacle1_mesh_frame_world", data=obstacle1_mesh_frame_world)
        dest_grasp_gp.create_dataset("obstacle2_mesh_frame_world", data=obstacle2_mesh_frame_world)
        dest_grasp_gp.create_dataset("obstacle3_mesh_frame_world", data=obstacle3_mesh_frame_world)


def create_grasp_group(group, idx):
    return group.create_group('grasp_' + str(idx).zfill(5))


def log_idxs(path, obj, pos, neg, coll, non_coll):
    l = [obj, pos, neg, coll, non_coll]
    if os.path.exists(path):
        string = ''
    else:
        string = 'obj_name \t \t \t pos \t neg \t coll \t non_coll \n'
    with open(path, 'a') as f:
        for i, txt in enumerate(l):
            string += str(txt) + '\t'
            if i == 0:
                string += '\t \t'

        string += '\n'
        f.write(string)


if __name__ == "__main__":
    # base_path contains all folders each contains a complete recording data.
    base_path = '/data/hdd1/qf/hithand_data/collision_only_data'
    dst_path = '/data/hdd1/qf/hithand_data/collision_only_data/grasp_data_all.h5'
    hdf_dst = h5py.File(dst_path, 'a')

    # go through all the dirs, each dir contains one grasp_data.h5
    for dir in sorted(os.listdir(base_path)):
        if dir[-2:] == 'h5' or dir[-3:] == 'txt':
            continue
        print('start foler: ', dir)
        src_path = os.path.join(base_path, dir, 'grasp_data.h5')
        hdf_src = h5py.File(src_path, 'r')

        for key in hdf_src[RC].keys():
            src_objs_gp = hdf_src[RC][key][GT]
            print 'All objects: ', src_objs_gp.keys()
            for obj in src_objs_gp.keys():
                print('Processing object:', obj)
                # if obj == 'bigbird_3m_high_tack_spray_adhesive':
                #     print('here')
                # Grasp idxs
                pos_idx = 0
                neg_idx = 0
                coll_idx = 0
                # For quick collision dataset
                non_coll_idx = 0
                # Get the object_group in dest file
                if obj not in hdf_dst.keys():
                    dst_obj_gp = hdf_dst.create_group(obj)
                    dst_obj_gp.create_group('positive')
                    dst_obj_gp.create_group('negative')
                    dst_obj_gp.create_group('collision')
                    dst_obj_gp.create_group('non_collision_not_executed')
                else:
                    dst_obj_gp = hdf_dst[obj]
                    if dst_obj_gp[P].keys():
                        pos_idx = int(dst_obj_gp[P].keys()[-1].split('_')[-1]) + 1
                    if dst_obj_gp[N].keys():
                        neg_idx = int(dst_obj_gp[N].keys()[-1].split('_')[-1]) + 1
                    if dst_obj_gp[C].keys():
                        coll_idx = int(dst_obj_gp[C].keys()[-1].split('_')[-1]) + 1
                    # Comment this if not for collision only
                    # if dst_obj_gp['non_collision_not_executed'].keys():
                    #     non_coll_idx = int(
                    #         dst_obj_gp['non_collision_not_executed'].keys()[-1].split('_')[-1]) + 1
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

                # Get the grasps that are non_collision and not executed
                # comment here if not for collision only
                # non_coll_gp = src_objs_gp[obj][G]['non_collision_not_executed']
                # for grasp in non_coll_gp.keys():
                #     src_grasp_gp = non_coll_gp[grasp]
                #     dst_grasp_gp = create_grasp_group(dst_obj_gp['non_collision_not_executed'],
                #                                       non_coll_idx)
                #     non_coll_idx += 1
                #     log_grasp(src_grasp_gp, dst_grasp_gp, is_coll=True)

                # Get the grasps from collision gp from src file
                src_coll_gp = src_objs_gp[obj][G][C]
                for grasp in src_coll_gp.keys():
                    src_grasp_gp = src_coll_gp[grasp]
                    dst_grasp_gp = create_grasp_group(dst_obj_gp['collision'], coll_idx)
                    coll_idx += 1
                    log_grasp(src_grasp_gp, dst_grasp_gp, is_coll=True)

                # Finally log the pos, neg coll idx to a txt file
                path = os.path.join(base_path, 'obj_metadata.txt')
                log_idxs(path, obj, pos_idx, neg_idx, coll_idx, non_coll_idx)

    # Create pandas dataframe and log
    save_path = os.path.join(os.path.split(dst_path)[0], 'metadata.csv')
    log_to_csv(hdf_dst, save_path)

    # close files
    hdf_src.close()
    hdf_dst.close()
