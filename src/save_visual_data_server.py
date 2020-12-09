#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from grasp_pipeline.srv import *
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os

import open3d as o3d
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


def cloud_from_ros_to_o3d(ros_cloud):
    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(
        pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))
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
            rgb = [
                convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]
        else:
            rgb = [
                convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]
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


class VisualDataSaver():
    def __init__(self):
        rospy.init_node("save_visual_data_server")
        rospy.Service('save_visual_data', SaveVisualData,
                      self.handle_visual_data_saver)
        rospy.loginfo('Service save_visual_data:')
        rospy.loginfo('Ready to save your awesome visual data.')
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # essential, otherwise next line crashes
        self.transform_camera_world = self.tf_buffer.lookup_transform(
            'world', 'camera_color_optical_frame', rospy.Time())
        rospy.loginfo(self.transform_camera_world)

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
        # Transform the point cloud into world frame
        o3d.io.write_point_cloud(point_cloud_save_path, point_cloud)

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
        point_cloud = o3d.io.read_point_cloud(point_cloud_save_path)
        return point_cloud

    ################ Handle below ########################


# NOTE: use do_transform_cloud, receive the transform and convert it to TransformStamped message

    def handle_visual_data_saver(self, req):
        # First of all, delete any old pcd files with the same name, because they are not being replaced when saving a second time
        if os.path.exists(req.point_cloud_save_path):
            os.remove(req.point_cloud_save_path)

        # Transform the pointcloud message into world frame
        rospy.loginfo(req.point_cloud.header)
        point_cloud_world = do_transform_cloud(req.point_cloud,
                                               self.transform_camera_world)
        rospy.loginfo(point_cloud_world.header)

        # Transform format in order to save data to disk
        depth_img = self.bridge.imgmsg_to_cv2(req.depth_img, "32FC1")
        color_img = self.bridge.imgmsg_to_cv2(req.color_img, "bgr8")
        point_cloud = cloud_from_ros_to_o3d(point_cloud_world)

        # Actually save the stuff
        self.save_depth_img(depth_img, req.depth_img_save_path)
        self.save_color_img(color_img, req.color_img_save_path)
        self.save_point_cloud(point_cloud, req.point_cloud_save_path)

        response = SaveVisualDataResponse()
        response.save_visual_data_success = True
        return response

if __name__ == "__main__":
    Saver = VisualDataSaver()
    rospy.spin()
