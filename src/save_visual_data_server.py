#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from grasp_pipeline.srv import *
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import open3d
from ctypes import *
from open3d_ros_helper import open3d_ros_helper as orh

# This module should save, restore and display depth images as well as point clouds

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
        int(
            cast(pointer(c_float(rgb_float)),
                 POINTER(c_uint32)).contents.value))


def cloud_from_ros_to_open3d(ros_cloud):
    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(
        pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))
    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data) == 0:
        print("Converting an empty cloud")
        return None
    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD = 3  # x, y, z, rgb
        # Get xyz
        # (why cannot put this line below rgb?)
        xyz = [(x, y, z) for x, y, z, rgb in cloud_data]
        # Get rgb
        # Check whether int or float
        # if float (from pcl::toROSMsg)
        if type(cloud_data[0][IDX_RGB_IN_FIELD]) == float:
            rgb = [
                convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]
        else:
            rgb = [
                convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]
        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(
            np.array(rgb) / 255.0)
    else:
        xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
    return open3d_cloud


# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)


def cloud_from_open3d_to_ros(open3d_cloud,
                             frame_id="camera_depth_optical_frame"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors) * 255)  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + \
            colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        cloud_data = np.c_[points, colors]
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)


class VisualDataSaver():
    def __init__(self):
        rospy.init_node("save_visual_data_server")
        rospy.Service('save_visual_data', SaveVisualData,
                      self.handle_visual_data_saver)
        rospy.loginfo('Service save_visual_data:')
        rospy.loginfo('Ready to save your awesome visual data.')
        self.bridge = CvBridge()

    def save_depth_img(self, depth_img, depth_img_save_path):
        depth_img_u16 = cv2.normalize(depth_img,
                                      depth_img,
                                      0,
                                      65535,
                                      cv2.NORM_MINMAX,
                                      dtype=cv2.CV_16UC1)
        cv2.imwrite(depth_img_save_path, depth_img_u16)

    def save_color_img(self, color_img, color_img_save_path):
        cv2.imwrite(color_img_save_path, color_img)

    def save_point_cloud(self, point_cloud, point_cloud_save_path):
        open3d.io.write_point_cloud(point_cloud_save_path, point_cloud)

    def load_depth_img(self, depth_img_save_path):
        depth_img_u16 = cv2.imread(depth_img_save_path, -1)
        depth_img_f32 = cv2.normalize(depth_img_u16,
                                      depth_img_u16,
                                      dtype=cv2.CV_32FC1)
        return depth_img_f32

    def load_color_img(self, color_img_save_path):
        color_img = cv2.imread(color_img_save_path)
        return color_img

    def load_point_cloud(self, point_cloud_save_path):
        point_cloud = open3d.io.read_point_cloud(point_cloud_save_path)
        return point_cloud

    ################ Handle below ########################

    def handle_visual_data_saver(self, req):
        depth_img = self.bridge.imgmsg_to_cv2(req.depth_img, "32FC1")
        color_img = self.bridge.imgmsg_to_cv2(req.color_img, "bgr8")
        point_cloud = cloud_from_ros_to_open3d(req.point_cloud)

        #open3d.visualization.draw_geometries([point_cloud])

        self.save_depth_img(depth_img, req.depth_img_save_path)
        self.save_color_img(color_img, req.color_img_save_path)
        self.save_point_cloud(point_cloud, req.point_cloud_save_path)

        response = SaveVisualDataResponse()
        response.save_visual_data_success = True
        return response


if __name__ == "__main__":
    Saver = VisualDataSaver()
    rospy.spin()
