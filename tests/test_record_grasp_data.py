import numpy as np
import h5py
from datetime import datetime
from geometry_msgs.msg import PoseStamped
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('grasp_pipeline')
sys.path.append(pkg_path + '/src')
import numpy as np
from utils import plot_voxel, convert_to_full_voxel_grid

PRINT_TREE = True

utah_grasp_data = "/home/vm/Downloads/merged_grasp_data_10_sets.h5"
#vincent_grasp_data = "/home/vm/grasp_data.h5"
vincent_grasp_data = "/home/vm/Documents/2021-02-27/grasp_data.h5"

joints = [
    'index0', 'index1', 'index2', 'index3', 'little0', 'little1', 'little2', 'little3', 'middle0',
    'middle1', 'middle2', 'middle3', 'ring0', 'ring1', 'ring2', 'ring3', 'thumb0', 'thumb1',
    'thumb2', 'thumb3'
]

# with h5py.File(vincent_grasp_data, "r") as hdf:
#     valid_grasps = hdf['recording_sessions']['recording_session_0001']['grasp_trials'][
#         'ycb_035_power_drill']['grasps']['no_collision']
#     last_grasp = valid_grasps.keys()[-2]
#     grasp = valid_grasps[last_grasp]
#     desired_pre = grasp['desired_preshape_joint_state'][()]
#     true_pre = grasp['true_preshape_joint_state'][()]
#     # print("Desired: ")
#     # print(np.round(desired_pre, 4))
#     # print("True: ")
#     # print(np.round(true_pre, 4))
#     for joint, number1, number2 in zip(joints, desired_pre, true_pre):
#         print '{}: \t \t {:9.4f} | {:9.4f}'.format(joint, number1, number2)

# with h5py.File(vincent_grasp_data, "r") as hdf:
#     last_sess_name = hdf['recording_sessions'].keys()[-1]
#     grasp_trials = hdf['recording_sessions'][last_sess_name]['grasp_trials']
#     for key in grasp_trials.keys():
#         print("The grasp trial is: " + key)
#         grasp = grasp_trials[key]
#         for grasp_key in grasp.keys():
#             print("The grasp key is: " + grasp_key)
#             print(grasp[grasp_key][()])

# Print the whole file tree
if PRINT_TREE:
    with h5py.File(vincent_grasp_data, "r") as hdf:
        for l1 in hdf.keys():
            print("Level 1: " + l1)
            try:
                for l2 in hdf[l1].keys():
                    print("Level 2: " + l2)
                    try:
                        for l3 in hdf[l1][l2].keys():
                            print("Level 3: " + l3)
                            try:
                                for l4 in hdf[l1][l2][l3].keys():
                                    print("Level 4: " + l4)
                                    try:
                                        for l5 in hdf[l1][l2][l3][l4].keys():
                                            print("Level 5: " + l5)
                                            try:
                                                for l6 in hdf[l1][l2][l3][l4][l5].keys():
                                                    print("Level 6: " + l6)
                                                    try:
                                                        for l7 in hdf[l1][l2][l3][l4][l5][l6].keys(
                                                        ):
                                                            print("Level 7: " + l7)
                                                            try:
                                                                for l8 in hdf[l1][l2][l3][l4][l5][
                                                                        l6][l7].keys():
                                                                    print("Level 8: " + l8)
                                                                    print(hdf[l1][l2][l3][l4][l5]
                                                                          [l6][l7][l8][()])
                                                            except:
                                                                continue
                                                    except:
                                                        continue
                                            except:
                                                continue
                                    except:
                                        continue
                            except:
                                continue
                    except:
                        continue
            except:
                continue

# with h5py.File(vincent_grasp_data, "r") as hdf:
#     keys = hdf.keys()
#     whd = [x for x in keys if "dim_w_h_d" in x]
#     source = [x for x in keys if "source" in x]

#     for i, name in enumerate(whd):
#         print(hdf[source[i]][()])
#         print(hdf[whd[i]][()])
#         if i == 50:
#             break
#     # print(hdf["grasp_0_source_info"][()])
#     # print(hdf["grasp_0_config_obj"][()])
#     # print(hdf["grasp_0_dim_w_h_d"][()])
#     # print(hdf["grasp_0_top_grasp"][()])
#     # print(hdf["grasp_0_label"][()])

# def print_attrs(name, obj):
#     print name
#     for key, val in obj.attrs.iteritems():
#         print "    %s: %s" % (key, val)

# def foo(name, obj):
#     print(name, obj)
#     return None

# with h5py.File("/home/vm/Downloads/merged_grasp_data_10_sets.h5", "r") as hdf:
#     #base_items = list(hdf.items())
#     keys = list(hdf.keys())
#     #hdf.visititems(foo)
#     a = list(hdf)
#     b = list(hdf.attrs)
#     data_set = hdf["grasp_3351_top_grasp"]
#     print(data_set.value)
#     print(list(data_set.attrs))
#     pass
#     "datetime_recordings"

# def convert_pose_to_list(pose):
#     return [
#         pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x,
#         pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
#     ]

# p = PoseStamped()

# print(convert_pose_to_list(p))

# Add grasp 0001
# with h5py.File("test.h5", "r+") as file:
#     grasps_group = file["grasps"]
#     # grasp_1 = grasps_group.create_group("grasp_000001")
#     # grasp_2 = grasps_group.create_group("grasp_000002")
#     print(list(grasps_group.keys()))
#     last_entry = grasps_group.keys()[-2]
#     number = int(last_entry.split('_')[-1])
#     number = number + 100
#     str_number = str(number).zfill(6)
#     # use .zfill(6) to convert back
#     print(number)
#     print(str_number)

# with h5py.File("test.h5", "r+") as file:
#     print(file.keys())
#     if "dim_w_h_d" in file.keys():
#         print("in file")
#     grasps = file["grasps"]["grasp_1"]["dim_w_h_d"]
#     grasps[()] = np.array([1, 2, 3])
#     dataset = file["metadata"]["dates_of_recording_sessions"]
#     dates = dataset[1]
#     print(dates)

# with h5py.File("test.h5", "a") as file:
#     grasps_group = file.create_group("grasps")
#     metadata_group = file.create_group("metadata")

#     metadata_group.create_dataset("total_grasps_num", data=0)
#     grasp1 = grasps_group.create_group("grasp_1")
#     grasp1.create_dataset("dim_w_h_d", shape=(3, ))
#     dim_w_h_d = grasp1["dim_w_h_d"]
#     grasp1["dim_w_h_d"] = np.array([0.1, 0.1, 0.1])
#     pass
#     metadata_group.create_dataset("dates_of_recording_sessions",
#                                    data=[datetime.now().isoformat()],
#                                    maxshape=(None, ))  # print(grasps.dtype)

# print(dataset.shape)
# dataset.resize((dataset.shape[0] + 1, ))
# print(dataset.shape)
# dataset[-1] = datetime.now().isoformat()
