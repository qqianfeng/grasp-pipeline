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
        self.VISUALIZE = rospy.get_param('visualize', False)
        pcd_topic = rospy.get_param('scene_pcd_topic')
        self.init_pcd_frame(pcd_topic)

        ########## Tune this parameter if camera is mounted differently ##########
        # Remove all points from pointcloud with x < 0.1, because they belong to panda base
        # for simulation
        self.x_min = 0.1 # for simulation
        self.z_max = 10
        # remove points that is too far away from the camera

        # for real world
        # self.x_min = 0
        # self.z_max = 0.8

        ###################################

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
        print('self.camera_T_world',self.camera_T_world)
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
                [0, 0, 0],  # black,       left/front/up
                [1, 0, 0],  # red          right/front/up
                [0, 1, 0],  # green        left/front/down
                [0, 0, 1],  # blue         left/back/up
                [0.5, 0.5, 0.5],  # grey   right/back/down
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

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def init_pcd_frame(self, pcd_topic):
        # as stated in grasp-pipeline/launch/grasp_pipeline_servers_real.launch, the pcd_topic for
        # realsense is either /camera/depth/points from simulation or the other one in real world
        if pcd_topic == '/camera/depth/points' or pcd_topic == '/camera/depth/color/points':
            self.pcd_frame = 'camera_depth_optical_frame'
        elif pcd_topic == '/depth_registered_points':
            self.pcd_frame = 'camera_color_optical_frame'
        else:
            rospy.logerr(
                'Wrong parameter set for scene_pcd_topic in grasp_pipeline_servers.launch')

    def visualize_normals(self, pcd):
        if not self.VISUALIZE:
            return
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(origin)
        vis.add_geometry(pcd)
        vis.get_render_option().load_from_json(os.path.join(
            ros.get_param('hithand_ws_path'), "src/grasp-pipeline/save.json"))
        vis.run()

    def custom_draw_scene(self, pcd):
        if not self.VISUALIZE:
            return
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    def construct_corner_box_objects(self):
        if not self.VISUALIZE:
            return
        boxes = []
        for i in range(8):
            box = o3d.geometry.TriangleMesh.create_box(
                width=0.01, height=0.01, depth=0.01)
            box.translate(self.bounding_box_corner_points[i, :])
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
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1)
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
                (self.object_centroid[0], self.object_centroid[1],
                 self.object_centroid[2]),
                (self.cam_q.x, self.cam_q.y, self.cam_q.z,
                 self.cam_q.w), rospy.Time.now(),
                "object_centroid_vae", "world")

    def publish_box_corner_points(self, points_stamped, color=(1., 0., 0.)):
        markerArray = MarkerArray()
        for i, pnt in enumerate(points_stamped):
            marker = Marker()
            marker.header.frame_id = pnt.header.frame_id
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

            marker.pose.position.x = pnt.point.x
            marker.pose.position.y = pnt.point.y
            marker.pose.position.z = pnt.point.z
            marker.id = i
            markerArray.markers.append(marker)
        self.bounding_box_corner_vis_pub.publish(markerArray)

    def _add_ground_to_segmentation(self, scene_pcd, object_pcd):
        """ Add ground point cloud with a ground margin exceeding the object bbox
        """
        scene_points = np.asarray(scene_pcd.points)
        object_points = np.asarray(object_pcd.points)
        ground_margin = 0.01 # m
        min_x = np.min(object_points[:,0]) -ground_margin
        max_x = np.max(object_points[:,0]) + ground_margin
        min_y = np.min(object_points[:,1]) - ground_margin
        max_y = np.max(object_points[:,1]) + ground_margin
        points = scene_points[scene_points[:,0]>min_x]
        points = points[points[:,0]<max_x]
        points = points[points[:,1]>min_y]
        points = points[points[:,1]<max_y]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def handle_segment_object(self, req):
        print("handle_segment_object received the service call")

        self.scene_pcd_path = req.scene_pcd_path
        self.object_pcd_path = req.object_pcd_path

        # if they are equal, segmentation is done by grabcut before hand so plane segmentation should be skipped
        if self.scene_pcd_path == self.object_pcd_path:
            object_pcd = o3d.io.read_point_cloud(self.scene_pcd_path)
        # otherwise run plane segmentation
        else:
            pcd = o3d.io.read_point_cloud(self.scene_pcd_path)
            # segment the panda base from point cloud
            points = np.asarray(pcd.points)  # shape [x,3]
            colors = np.asarray(pcd.colors)

            # currently the mask cropping is removed
            mask1 = points[:, 0] > self.x_min
            mask2 = points[:, 2] < self.z_max
            mask = np.logical_and(mask1, mask2)
            del pcd
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            print("draw the scene")
            self.custom_draw_scene(pcd)

            # segment plane
            _, inliers = pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=30)
            object_pcd = pcd.select_down_sample(inliers, invert=True)

            print("draw the object pcd")
            self.custom_draw_object(object_pcd)

            # add plane point cloud
            object_pcd = self._add_ground_to_segmentation(pcd,object_pcd)
            print("draw the object pcd with ground")
            self.custom_draw_object(object_pcd)
            del pcd, points, colors

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))

        print("draw object pcd with normal estimations")
        print("center:",object_pcd.get_center())
        self.custom_draw_scene(object_pcd)

        # downsample point cloud or make mean free if downsampling is not requested
        if req.down_sample_pcd:
            object_pcd = object_pcd.voxel_down_sample(voxel_size=0.003)

        self.custom_draw_scene(object_pcd)

        # compute bounding box and object_pose
        object_bounding_box = object_pcd.get_oriented_bounding_box()
        #object_bounding_box_aligned = object_pcd.get_axis_aligned_bounding_box()
        self.object_size = object_bounding_box.extent
        self.object_center = object_pcd.get_center()
        self.object_R = copy.deepcopy(object_bounding_box.R)
        self.object_R[:, 2] = np.cross(
            self.object_R[:, 0], self.object_R[:, 1])
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
        self.bounding_box_corner_points = np.asarray(
            object_bounding_box.get_box_points())

        # Publish the bounding box corner points for visualization in Rviz
        box_corners_world_frame = []
        corner_stamped_world = PointStamped()
        corner_stamped_world.header.frame_id = 'world'
        for corner in self.bounding_box_corner_points:
            corner_stamped_world.point.x = corner[0]
            corner_stamped_world.point.y = corner[1]
            corner_stamped_world.point.z = corner[2]
            box_corners_world_frame.append(copy.deepcopy(corner_stamped_world))
        self.publish_box_corner_points(box_corners_world_frame)

        # orient normals towards camera
        rospy.loginfo('Orienting normals towards this location:')
        rospy.loginfo(self.world_t_cam)
        object_pcd.orient_normals_towards_camera_location(self.world_t_cam)
        print("Original scene point cloud reference frame assumed as: " +
              str(self.pcd_frame))

        # Draw object, bounding box and colored corners
        self.custom_draw_object(object_pcd, object_bounding_box, False, True)

        # Store segmented object to disk
        if os.path.exists(self.object_pcd_path):
            os.remove(self.object_pcd_path)

        # only in the data_gen_pipeline.py, data_gen_pipeline_multi_obj and verify_collision_label.py,
        # down_sample_pcd is set to true.
        # TODO: fix this super wired flag
        if not req.down_sample_pcd:  # if req.down_sample is false, we assume this should be stored in VAE format, therefore transform the cloud back to camera frame
            self.object_centroid = object_pcd.get_center()
            # if no need to transfer to world frame, pcd stay in self center frame.
            if not req.need_to_transfer_pcd_to_world_frame:
                object_pcd.transform(self.camera_T_world)
                object_pcd.translate((-1) * object_pcd.get_center())
            # else pcd stays in world frame
            print("pcd in world frame")
            self.custom_draw_scene(object_pcd)

        o3d.io.write_point_cloud(self.object_pcd_path, object_pcd)
        if req.object_pcd_record_path != '':
            o3d.io.write_point_cloud(req.object_pcd_record_path, object_pcd)
        print("Object.pcd saved successfully with normals oriented towards camera")

        # Publish and latch newly computed dimensions and bounding box points
        print("I will publish the corner points now:")
        corner_msg = Float64MultiArray()
        corner_msg.data = np.ndarray.tolist(
            np.ndarray.flatten(self.bounding_box_corner_points))
        self.bounding_box_corner_pub.publish(corner_msg)

        res = SegmentGraspObjectResponse()
        res.object.header.frame_id = 'world'
        res.object.header.stamp = rospy.Time.now()
        res.object.pose = self.object_pose
        res.object.object_pcd_path = self.object_pcd_path
        # corresponds to x in oriented bb frame
        res.object.width = self.object_size[0]
        # corresponds to y in oriented bb frame
        res.object.height = self.object_size[1]
        # corresponds to z in oriented bb frame
        res.object.depth = self.object_size[2]
        res.success = True

        self.service_is_called = True
        return res

    def create_segment_object_server(self):
        rospy.Service('segment_object', SegmentGraspObject,
                      self.handle_segment_object)
        rospy.loginfo('Service segment_object:')
        rospy.loginfo(
            'Ready to segment the table from the object point cloud.')


if __name__ == "__main__":
    oseg = ObjectSegmenter()
    oseg.create_segment_object_server()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        oseg.broadcast_object_pose()
        rate.sleep()
