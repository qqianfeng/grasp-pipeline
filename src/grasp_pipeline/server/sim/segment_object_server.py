#!/usr/bin/env python
import rospy
import open3d as o3d
import os
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation

import tf.transformations as tft
import tf2_ros
import tf

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker

from grasp_pipeline.srv import *


class ObjectSegmenter():
    def __init__(self):
        rospy.init_node("object_segmentation_node")
        self.align_bounding_box = rospy.get_param('align_bounding_box', 'true')
        self.scene_pcd_topic = rospy.get_param('scene_pcd_topic')
        self.VISUALIZE = rospy.get_param('visualize', True)
        self.is_simulation = rospy.get_param('simulation', True)
        self.init_pcd_frame(self.scene_pcd_topic)

        self.x_threshold = 0.1  # Remove all points from pointcloud with x < 0.1, because they belong to panda base

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)  # essential, otherwise next line crashes
        self.transform_camera_world = self.tf_buffer.lookup_transform(
            'world', self.pcd_frame, rospy.Time())
        self.transform_world_camera = self.tf_buffer.lookup_transform(
            self.pcd_frame, 'world', rospy.Time())
        q = self.transform_world_camera.transform.rotation
        r = self.transform_world_camera.transform.translation
        self.camera_T_world = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        self.camera_T_world[:, 3] = [r.x, r.y, r.z, 1]

        self.bounding_box_corner_pub = rospy.Publisher(
            '/segmented_object_bounding_box_corner_points',
            Float64MultiArray,
            latch=True,
            queue_size=1)
        self.bounding_box_corner_vis_pub = rospy.Publisher('/box_corner_points',
                                                           MarkerArray,
                                                           queue_size=1,
                                                           latch=True)
        self.tf_broadcaster_object_pose = tf.TransformBroadcaster()

        self.bounding_box_corner_points = None
        self.world_t_cam = np.array([
            self.transform_camera_world.transform.translation.x,
            self.transform_camera_world.transform.translation.y,
            self.transform_camera_world.transform.translation.z
        ])
        self.cam_q = self.transform_camera_world.transform.rotation
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

        self.object_centroid = None

        self.setup_workspace_boundaries()

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def setup_workspace_boundaries(self):
        self.x_min = 0.1
        self.x_max = 1
        self.y_min = 0.1
        self.y_max = 0.4
        self.z_min = -0.02
        self.z_max = 0.3

    def init_pcd_frame(self, pcd_topic):
        if pcd_topic in ['/camera/depth/points', '/camera/depth/color/points']:
            self.pcd_frame = 'camera_depth_optical_frame'
        elif pcd_topic == '/depth_registered_points':
            self.pcd_frame = 'camera_color_optical_frame'
        else:
            rospy.logerr(
                'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

    def draw_pcd_and_origin(self, pcd):
        if not self.VISUALIZE:
            return
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    def construct_corner_box_objects(self):
        if not self.VISUALIZE:
            return
        boxes = []
        for i in range(8):
            box = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
            box.translate(self.bb_corner_points[i, :])
            box.paint_uniform_color(self.colors[i, :])
            boxes.append(box)
        return boxes

    def custom_draw_object(self, pcd, bounding_box=None, show_normal=False, draw_box=False):
        if not self.VISUALIZE:
            return
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

        if self.object_centroid is not None:
            self.tf_broadcaster_object_pose.sendTransform(
                (self.object_centroid[0], self.object_centroid[1], self.object_centroid[2]),
                (self.cam_q.x, self.cam_q.y, self.cam_q.z, self.cam_q.w), rospy.Time.now(),
                "object_centroid_vae", "world")

    def publish_box_corner_points(self, object_bb, color=(1., 0., 0.)):
        #visualization of corners in rviz
        markerArray = MarkerArray()
        bb_corner_points = np.asarray(object_bb.get_box_points())
        for i, corner in enumerate(bb_corner_points):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03
            marker.pose.orientation.w = 1.0
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            marker.pose.position.x = corner[0]
            marker.pose.position.y = corner[1]
            marker.pose.position.z = corner[2]
            marker.id = i
            markerArray.markers.append(marker)
        self.bounding_box_corner_vis_pub.publish(markerArray)

        # dont remember why necessary
        corner_msg = Float64MultiArray()
        corner_msg.data = np.ndarray.tolist(np.ndarray.flatten(bb_corner_points))
        self.bounding_box_corner_pub.publish(corner_msg)

        self.bb_corner_points = bb_corner_points

    def filter_pcd_workspace_boundaries(self, pcd):
        points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        if self.is_simulation == False:
            # filter x direction
            mask = np.logical_and(points[:, 0] > self.x_min, points[:, 0] < self.x_max)
            points, colors = points[mask], colors[mask]

            # filter y direction
            mask = np.logical_and(points[:, 1] > self.y_min, points[:, 1] < self.y_max)
            points, colors = points[mask], colors[mask]

            # filter z direction
            mask = np.logical_and(points[:, 2] > self.z_min, points[:, 2] < self.z_max)
            points, colors = points[mask], colors[mask]
        elif self.is_simulation == True:
            mask = points[:, 0] > self.x_threshold
            points, colors = points[mask], colors[mask]
        else:
            raise Exception("is_simulation must be bool.")

        # new pcd
        del pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def segment_object_from_scene(self, scene_pcd):
        # Remove table plane
        distance_threshold = 0.01
        _, inliers = scene_pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=30)
        object_pcd = scene_pcd.select_down_sample(inliers, invert=True)
        self.display_inlier_outlier(object_pcd, scene_pcd.select_down_sample(inliers))

        # For real data add some radius outlier removal
        if self.is_simulation == False:
            _, inliers = object_pcd.remove_radius_outlier(nb_points=80, radius=0.03)
            object_pcd_outliers = object_pcd.select_down_sample(inliers, invert=True)
            object_pcd = object_pcd.select_down_sample(inliers)
            self.display_inlier_outlier(object_pcd_outliers, object_pcd)

        return object_pcd

    def display_inlier_outlier(self, outlier_cloud, inlier_cloud):
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def get_object_bb_pose_and_size(self, object_pcd):
        object_bounding_box = object_pcd.get_oriented_bounding_box()
        object_size = object_bounding_box.extent
        object_center = object_pcd.get_center()
        object_R = copy.deepcopy(object_bounding_box.R)
        object_R[:, 2] = np.cross(object_R[:, 0], object_R[:, 1])

        object_T_world = np.eye(4)
        object_T_world[:3, :3] = object_R
        object_quat = tft.quaternion_from_matrix(object_T_world)

        object_pose = Pose()
        object_pose.position.x = object_center[0]
        object_pose.position.y = object_center[1]
        object_pose.position.z = object_center[2]
        object_pose.orientation.x = object_quat[0]
        object_pose.orientation.y = object_quat[1]
        object_pose.orientation.z = object_quat[2]
        object_pose.orientation.w = object_quat[3]

        # Set these as attributes, they are needed only for optional visualization
        self.object_R = object_R
        self.object_center = object_center

        return object_bounding_box, object_pose, object_size


## ========================================================================================
# ========================= Part III: Main Server code ====================================

    def handle_segment_object(self, req):
        # read pcd
        self.scene_pcd_path = req.scene_pcd_path
        self.object_pcd_path = req.object_pcd_path
        pcd = o3d.io.read_point_cloud(self.scene_pcd_path)

        # Transform cloud, momentary solution NEEDS TO BE CHANGED
        if self.is_simulation == False:
            # Transform cloud only during debugging
            #cam_T_base = tft.euler_matrix(2.0, 0.1, 0.1)
            cam_T_base = tft.euler_matrix(
                -2.0, -0.1, -0.1)  # ATTENTION: camera coordinate system not same as in simulation
            cam_T_base[:3, 3] = [0.4, -0.35, 0.4]
            #pcd.transform(np.linalg.inv(cam_T_base))
            pcd.transform(cam_T_base)
            # TODO: Replace with actual transform
            print("Replace this with actual transform!!!!")

        # Filter regions which are not in workspace
        self.draw_pcd_and_origin(pcd)
        pcd = self.filter_pcd_workspace_boundaries(pcd)
        self.draw_pcd_and_origin(pcd)

        # segment plane
        object_pcd = self.segment_object_from_scene(pcd)
        self.custom_draw_object(object_pcd)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))
        self.draw_pcd_and_origin(object_pcd)

        # downsample point cloud
        if req.down_sample_pcd:
            object_pcd = object_pcd.voxel_down_sample(voxel_size=0.003)
            self.draw_pcd_and_origin(object_pcd)

        # Get pose and size of object from oriented bounding box. Stores more object information in attributes.
        object_bb, object_pose, object_size = self.get_object_bb_pose_and_size(object_pcd)

        # Make object pose an attribute for broadcasting it
        self.object_pose = object_pose

        # Publish the bounding box corner points for visualization in Rviz
        self.publish_box_corner_points(object_bb)

        # orient normals towards camera
        object_pcd.orient_normals_towards_camera_location(self.world_t_cam)
        print("Original scene point cloud reference frame assumed as: " + str(self.pcd_frame))

        # Draw object, bounding box and colored corners
        self.custom_draw_object(object_pcd, object_bb, False, True)

        # TODO: This is dirty, make more explicit
        # Transform back the cloud if downsample is false, which assumes VAE format
        if not req.down_sample_pcd:  # if req.down_sample is false, we assume this should be stored in VAE format, therefore transform the cloud back to camera frame
            self.object_centroid = object_pcd.get_center()
            if self.is_simulation:
                object_pcd.transform(self.camera_T_world)
                object_pcd.translate((-1) * object_pcd.get_center())
            else:
                object_pcd.translate(((-1) * object_pcd.get_center()))
                cam_T_base[:3, 3] = 0
                object_pcd.transform(np.linalg.inv(cam_T_base))
            self.draw_pcd_and_origin(object_pcd)

        # Store segmented object to disk
        if os.path.exists(self.object_pcd_path):
            os.remove(self.object_pcd_path)
        o3d.io.write_point_cloud(self.object_pcd_path, object_pcd)
        if req.object_pcd_record_path != '':
            o3d.io.write_point_cloud(req.object_pcd_record_path, object_pcd)
        print("Object.pcd saved successfully with normals oriented towards camera")

        # Prepare response
        res = SegmentGraspObjectResponse()
        res.object.header.frame_id = 'world'
        res.object.header.stamp = rospy.Time.now()
        res.object.pose = object_pose
        res.object.object_pcd_path = self.object_pcd_path
        res.object.width = object_size[0]  # corresponds to x in oriented bb frame
        res.object.height = object_size[1]  # corresponds to y in oriented bb frame
        res.object.depth = object_size[2]  # corresponds to z in oriented bb frame
        res.success = True

        self.service_is_called = True
        return res

    def create_segment_object_server(self):
        rospy.Service('segment_object', SegmentGraspObject, self.handle_segment_object)
        rospy.loginfo('Service segment_object:')
        rospy.loginfo('Ready to segment the table from the object point cloud.')

DEBUG = False

if __name__ == "__main__":
    oseg = ObjectSegmenter()

    if DEBUG:
        req = SegmentGraspObjectRequest()
        req.down_sample_pcd = False
        req.scene_pcd_path = '/home/vm/scene.pcd'
        req.object_pcd_path = '/home/vm/object.pcd'
        oseg.handle_segment_object(req)

    oseg.create_segment_object_server()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        oseg.broadcast_object_pose()
        rate.sleep()