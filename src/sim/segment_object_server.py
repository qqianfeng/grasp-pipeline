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
from geometry_msgs.msg import PoseStamped

DEBUG = False


class ObjectSegmenter():
    def __init__(self,
                 scene_point_cloud_path=None,
                 object_point_cloud_path=None):
        if not DEBUG:
            rospy.init_node("object_segmentation_node")
            self.object_size_pub = rospy.Publisher('/segmented_object_size',
                                                   Float64MultiArray,
                                                   latch=True,
                                                   queue_size=1)
            self.bounding_box_corner_pub = rospy.Publisher(
                '/segmented_object_bounding_box_corner_points',
                Float64MultiArray,
                latch=True,
                queue_size=1)
            # for the camera position in 3D space w.r.t world
            self.camera_tf_listener = rospy.Subscriber(
                '/camera_color_optical_frame_in_world',
                PoseStamped,
                self.callback_camera_tf,
                queue_size=5)
            # Where to save the object point cloud and where to load it from
            self.object_point_cloud_path = rospy.get_param(
                'scene_point_cloud_path')
            self.scene_point_cloud_path = rospy.get_param(
                'object_point_cloud_path')

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
                [0.6, 0.2, 0.4]  #purple/ red-ish
            ]
        )  # this was just to understand the corner numbering logic, point 0 and point 4 in the list are cross diagonal, points 1,2,3 are attached to 0 in right handed sense, same for 5,6,7

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def custom_draw_scene(self, pcd):
        o3d.visualization.draw_geometries([pcd],
                                          front=[0.236, -0.97, 0.0435],
                                          lookat=[-0.244, 3.89, 0.0539],
                                          up=[0., 0.045, 0.998],
                                          zoom=0.2399)

    def construct_nine_box_objects(self):
        boxes = []
        for i in range(9):
            box = o3d.geometry.TriangleMesh.create_box(width=0.01,
                                                       height=0.01,
                                                       depth=0.01)
            box.translate(self.bounding_box_corner_points[
                i, :]) if i != 8 else box.translate([0, 0, 0])
            box.paint_uniform_color(self.colors[i, :])
            boxes.append(box)
        return boxes

    def custom_draw_object(self,
                           pcd,
                           bounding_box=None,
                           show_normal=False,
                           draw_box=False):
        if bounding_box == None:
            o3d.visualization.draw_geometries([pcd])
        else:
            boxes = self.construct_nine_box_objects()
            o3d.visualization.draw_geometries([
                pcd, bounding_box, boxes[0], boxes[1], boxes[2], boxes[3],
                boxes[4], boxes[5], boxes[6], boxes[7], boxes[8]
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
    def callback_camera_tf(self, msg):
        """ Get the camera transform from ROS and extract the camera 3D position for visualization purposes.
        """
        print("Entered callback camera tf")
        self.camera_tf_listener.unregister()
        position = msg.pose.position
        self.world_t_cam = np.array([position.x, position.y, position.z])
        print("Unsubscribed camera_tf_listener")

    def handle_segment_object(self, req):
        self.scene_point_cloud_path = req.scene_point_cloud_path
        self.object_point_cloud_path = req.object_point_cloud_path
        while (self.scene_point_cloud_path is
               None) or (self.object_point_cloud_path is None):
            print(
                "I haven't received the pointcloud paths from the corrsponding topic. Stuck in a while loop until they are received."
            )
            time.sleep(2)

        print("handle_segment_object received the service call")
        pcd = o3d.io.read_point_cloud(self.object_point_cloud_path)

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
        _, inliers = down_pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=30)
        object_pcd = down_pcd.select_down_sample(inliers, invert=True)
        #print(time.time() - start)
        if DEBUG:
            self.custom_draw_object(object_pcd)

        # compute bounding box and size
        object_bounding_box = object_pcd.get_axis_aligned_bounding_box()
        object_size = object_bounding_box.get_extent()
        # get the 8 corner points, from these you can compute the bounding box face center points from which you can get the nearest neighbour from the point cloud
        self.bounding_box_corner_points = np.asarray(
            object_bounding_box.get_box_points())
        print(self.bounding_box_corner_points)
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,
                                                              max_nn=50))
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # orient normals towards camera
        object_pcd.orient_normals_towards_camera_location(self.world_t_cam)
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # Draw object, bounding box and colored corners
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, False,
                                    True)

        # In the end the object pcd, bounding box corner points, bounding box size information need to be stored to disk
        # Could also be sent over a topic
        if os.path.exists(self.scene_point_cloud_path):
            os.remove(self.scene_point_cloud_path)
        o3d.io.write_point_cloud(self.scene_point_cloud_path, object_pcd)
        print(
            "Object.pcd saved successfully with normals oriented towards camera"
        )

        # Publish and latch newly computed dimensions and bounding box points
        if not DEBUG:
            print("I will publish the size now:")
            size_msg = Float64MultiArray()
            size_msg.data = np.ndarray.tolist(object_size)
            self.object_size_pub.publish(size_msg)
            print("I will publish the corner points now:")
            print(self.bounding_box_corner_points)
            corner_msg = Float64MultiArray()
            corner_msg.data = np.ndarray.tolist(
                np.ndarray.flatten(self.bounding_box_corner_points))
            self.bounding_box_corner_pub.publish(corner_msg)
            print("I published them")

        res = SegmentGraspObjectResponse()
        res.success = True

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
    if DEBUG:
        oseg = ObjectSegmenter(
            scene_point_cloud_path='/home/vm/test_cloud.pcd',
            object_point_cloud_path='/home/vm/object.pcd')
        oseg.handle_segment_object(None)

    rospy.spin()