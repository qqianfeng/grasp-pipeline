import numpy as np
import h5py
from datetime import datetime
from geometry_msgs.msg import PoseStamped

with h5py.File("/home/vm/Downloads/merged_grasp_data_10_sets.h5", "r") as hdf:
    keys = hdf.keys()
    whd = [x for x in keys if "dim_w_h_d" in x]
    source = [x for x in keys if "source" in x]

    for i, name in enumerate(whd):
        print(hdf[source[i]][()])
        print(hdf[whd[i]][()])
        if i == 50:
            break
    # print(hdf["grasp_0_source_info"][()])
    # print(hdf["grasp_0_config_obj"][()])
    # print(hdf["grasp_0_dim_w_h_d"][()])
    # print(hdf["grasp_0_top_grasp"][()])
    # print(hdf["grasp_0_label"][()])


def print_attrs(name, obj):
    print name
    for key, val in obj.attrs.iteritems():
        print "    %s: %s" % (key, val)


def foo(name, obj):
    print(name, obj)
    return None


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


def convert_pose_to_list(pose):
    return [
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x,
        pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
    ]


p = PoseStamped()

print(convert_pose_to_list(p))

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
