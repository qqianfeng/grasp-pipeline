from sensor_msgs.msg import Image, PointCloud2, PointField
import open3d as o3d
from ctypes import *
import sensor_msgs.point_cloud2 as pc2
import numpy as np

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


def convert_rgbUint32_to_tuple(rgb_uint32):
    return ((rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8,
            (rgb_uint32 & 0x000000ff))


def convert_rgbFloat_to_tuple(rgb_float):
    return convert_rgbUint32_to_tuple(
        int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value))


def cloud_from_ros_to_o3d(ros_cloud):
    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))
    # Check empty
    o3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data) == 0:
        print("Converting an empty cloud")
        return None
    # Set o3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD = 3  # x, y, z, rgb
        # Get xyz
        # (why cannot put this line below rgb?)
        xyz = [(x, y, z) for x, y, z, rgb in cloud_data]
        # Get rgb
        # Check whether int or float
        # if float (from pcl::toROSMsg)
        if type(cloud_data[0][IDX_RGB_IN_FIELD]) == float:
            rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data]
        # combine
        o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        o3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb) / 255.0)
    else:
        xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
        o3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    return o3d_cloud


# Convert the datatype of point cloud from o3d to ROS PointCloud2 (XYZRGB only)


def cloud_from_o3d_to_ros(o3d_cloud, frame_id="camera_depth_optical_frame"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # Set "fields" and "cloud_data"
    points = np.asarray(o3d_cloud.points)
    if not o3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(o3d_cloud.colors) * 255)  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + \
            colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        cloud_data = np.c_[points, colors]
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)
