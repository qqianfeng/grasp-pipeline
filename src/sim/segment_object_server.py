#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *
import open3d as o3d
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
import copy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf.transformations as tft
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('grasp_pipeline')
import sys
sys.path.append(pkg_path + '/src')
from utils import pcd_from_ros_to_o3d
import tf2_ros
import tf

DEBUG = False


class ObjectSegmenter():
    def __init__(self):
        rospy.init_node("object_segmentation_node")
        self.align_bounding_box = rospy.get_param('align_bounding_box', 'true')
        self.scene_pcd_topic = rospy.get_param('scene_pcd_topic')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # essential, otherwise next line crashes

        pcd_topic = rospy.get_param('scene_pcd_topic')
        if pcd_topic == '/camera/depth/points':
            pcd_frame = 'camera_depth_optical_frame'
        elif pcd_topic == '/depth_registered_points':
            pcd_frame = 'camera_color_optical_frame'
        else:
            rospy.logerr(
                'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

        self.transform_camera_world = self.tf_buffer.lookup_transform(
            'world', pcd_frame, rospy.Time())
        if not DEBUG:
            self.bounding_box_corner_pub = rospy.Publisher(
                '/segmented_object_bounding_box_corner_points',
                Float64MultiArray,
                latch=True,
                queue_size=1)
            # for the camera position in 3D space w.r.t world
            self.camera_tf_listener = rospy.Subscriber('/' + pcd_frame + '_in_world'
                                                       PoseStamped,
                                                       self.callback_camera_tf,
                                                       queue_size=5)
            self.tf_broadcaster_object_pose = tf.TransformBroadcaster()

        self.bounding_box_corner_points = None
        self.world_R_cam = None
        self.world_t_cam = None
        self.colors = np.array(
            [
                [0, 0, 0],  #black,       left/front/up
                [1, 0, 0],  #red          right/front/up
                [0, 1, 0],  #green        left/front/down
                [0, 0, 1],  #blue         left/back/up
                [0.5, 0.5, 0.5],  #grey   right/back/down
                [0, 1, 1],  # light blue   left/back/down
                [1, 1, 0],  # yellow      right/back/up
                [1, 0, 1],
            ]
        )  # this was just to understand the corner numbering logic, point 0 and point 4 in the list are cross diagonal, points 1,2,3 are attached to 0 in right handed sense, same for 5,6,7
        self.service_is_called = False
        self.object_pose = None
        self.object_size = None
        self.object_center = None

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def custom_draw_scene(self, pcd):
        o3d.visualization.draw_geometries([pcd])

    def construct_corner_box_objects(self):
        boxes = []
        for i in range(8):
            box = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
            box.translate(self.bounding_box_corner_points[i, :])
            box.paint_uniform_color(self.colors[i, :])
            boxes.append(box)
        return boxes

    def custom_draw_object(self, pcd, bounding_box=None, show_normal=False, draw_box=False):
        if bounding_box == None:
            o3d.visualization.draw_geometries([pcd])
        else:
            boxes = self.construct_corner_box_objects()
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            obj_origin = copy.deepcopy(origin)
            pcd_origin = copy.deepcopy(origin)

            obj_origin.translate(self.object_center)
            obj_origin.rotate(self.object_R)

            pcd_origin.translate(pcd.get_center())

            o3d.visualization.draw_geometries([
                pcd, bounding_box, boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5],
                boxes[6], boxes[7], origin, obj_origin, pcd_origin
            ])

    def pick_points(self, pcd, bounding_box):
        print("")
        print("1) Please pick at least 3 corresp. using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(bounding_box)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    # +++++++++++++++++ Part II: Main business logic ++++++++++++++++++++++++
    def broadcast_object_pose(self):
        if self.service_is_called:
            # Publish the palm goal tf
            self.tf_broadcaster_object_pose.sendTransform(
                (self.object_pose.position.x, self.object_pose.position.y,
                 self.object_pose.position.z),
                (self.object_pose.orientation.x, self.object_pose.orientation.y,
                 self.object_pose.orientation.z, self.object_pose.orientation.w), rospy.Time.now(),
                "object_pose", "world")

    def callback_camera_tf(self, msg):
        """ Get the camera transform from ROS and extract the camera 3D position for visualization purposes.
        """
        print("Entered callback camera tf")
        self.camera_tf_listener.unregister()
        position = msg.pose.position
        self.world_t_cam = np.array([position.x, position.y, position.z])
        print("Unsubscribed camera_tf_listener")

    def handle_segment_object(self, req):
        print("handle_segment_object received the service call")

        self.scene_pcd_path = req.scene_pcd_path
        self.object_pcd_path = req.object_pcd_path

        pcd = o3d.io.read_point_cloud(self.scene_pcd_path)

        if self.world_t_cam is None:
            self.world_t_cam = [0.8275, -0.996, 0.36]
        if DEBUG:
            self.custom_draw_scene(pcd)
        #start = time.time()

        # downsample point cloud
        down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # downsample
        if DEBUG:
            self.custom_draw_scene(down_pcd)

        # segment plane
        _, inliers = down_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
        object_pcd = down_pcd.select_down_sample(inliers, invert=True)
        #print(time.time() - start)
        if DEBUG:
            self.custom_draw_object(object_pcd)

        # compute bounding box and object_pose
        object_bounding_box = object_pcd.get_oriented_bounding_box()
        object_bounding_box_aligned = object_pcd.get_axis_aligned_bounding_box()
        self.object_size = object_bounding_box.extent
        self.object_center = object_pcd.get_center()
        self.object_R = copy.deepcopy(object_bounding_box.R)
        # Attention, self.object_R can be an improper rotation meaning det(R)=-1, therefore check which eigenvalue is negative and turn the corresponding column of rotation matrix around
        eigs = np.linalg.eigvals(self.object_R)
        for i in range(3):
            if eigs[i] < 0:
                self.object_R[:, i] = (-1) * self.object_R[:, i]
        object_T_world = np.eye(4)
        object_T_world[:3, :3] = self.object_R
        object_quat = tft.quaternion_from_matrix(object_T_world)
        object_pose = Pose()
        object_pose.position.x = self.object_center[0]
        object_pose.position.y = self.object_center[1]
        object_pose.position.z = self.object_center[2]
        object_pose.orientation.x = object_quat[0]
        object_pose.orientation.y = object_quat[1]
        object_pose.orientation.z = object_quat[2]
        object_pose.orientation.w = object_quat[3]
        self.object_pose = object_pose

        # get the 8 corner points, from these you can compute the bounding box face center points from which you can get the nearest neighbour from the point cloud
        self.bounding_box_corner_points = np.asarray(object_bounding_box.get_box_points())
        print(self.bounding_box_corner_points)

        self.custom_draw_object(object_pcd, object_bounding_box)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # orient normals towards camera
        rospy.loginfo('Orienting normals towards this location:')
        rospy.loginfo(self.world_t_cam)
        object_pcd.orient_normals_towards_camera_location(self.world_t_cam)
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # Draw object, bounding box and colored corners
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, False, True)

        # Store segmented object to disk
        if os.path.exists(self.object_pcd_path):
            os.remove(self.object_pcd_path)
        o3d.io.write_pcd(self.object_pcd_path, object_pcd)
        print("Object.pcd saved successfully with normals oriented towards camera")

        # Publish and latch newly computed dimensions and bounding box points
        if not DEBUG:
            print("I will publish the corner points now:")
            print(self.bounding_box_corner_points)
            corner_msg = Float64MultiArray()
            corner_msg.data = np.ndarray.tolist(np.ndarray.flatten(
                self.bounding_box_corner_points))
            self.bounding_box_corner_pub.publish(corner_msg)
            print("I published them")

        res = SegmentGraspObjectResponse()
        res.object.header.frame_id = 'world'
        res.object.header.stamp = rospy.Time.now()
        res.object.pose = self.object_pose
        res.object.object_pcd_path = self.object_pcd_path
        res.object.width = self.object_size[0].tolist()
        res.object.height = self.object_size[2].tolist()
        res.object.depth = self.object_size[1].tolist()
        res.success = True

        self.service_is_called = True
        return res

    def create_segment_object_server(self):
        rospy.Service('segment_object', SegmentGraspObject, self.handle_segment_object)
        rospy.loginfo('Service segment_object:')
        rospy.loginfo('Ready to segment the table from the object point cloud.')


if __name__ == "__main__":
    oseg = ObjectSegmenter()
    oseg.create_segment_object_server()
    if DEBUG:
        oseg = ObjectSegmenter()
        req = SegmentGraspObjectRequest()
        req.object_pcd_path = "/home/vm/object.pcd"
        req.scene_pcd_path = "/home/vm/scene.pcd"
        oseg.handle_segment_object(req)

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        oseg.broadcast_object_pose()
        rate.sleep()