#!/usr/bin/env python
import rospy
from grasp_pipeline.srv import *

import tf
import tf.transformations as tft

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState

import numpy as np
import copy
import open3d as o3d

from eigengrasps_hithand import *


class BoundingBoxFace():
    """Simple class to store properties of a bounding box face.
    """
    def __init__(self, color, center, orient_a, orient_b, size_a, size_b, is_top=False):
        self.center = np.array(center)
        self.orient_a = orient_a
        self.orient_b = orient_b
        self.size_a = size_a
        self.size_b = size_b
        self.color = color
        self.is_top = is_top

    def get_longer_side_orient(self):
        if self.size_a > self.size_b:
            orient = self.orient_a
        else:
            orient = self.orient_b
        # Check that orient does not point down "too much"
        if orient[2] < -0.2:
            orient = (-1) * orient
        return orient


class GenerateHithandPreshape():
    def __init__(self):
        rospy.init_node("generate_hithand_preshape_node")
        # Publish information about bounding box
        self.bounding_box_center_pub = rospy.Publisher(
            '/box_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points
        self.bounding_box_face_centers_pub = rospy.Publisher(
            '/box_face_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points

        # Ros params / hyperparameters
        self.VISUALIZE = rospy.get_param('visualize', False)
        self.num_samples_from_bb_face_center = 3
        self.num_samples_around_bb_face_center = 1
        self.num_samples_from_random_object_points = 3
        self.palm_obj_min = 0.04
        self.palm_obj_max = 0.12
        self.approach_dist = 0.1

        # Publish goal poses over TF
        self.tf_broadcaster_palm_poses = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        # Segmented object vars
        self.object_pcd_path = rospy.get_param('object_pcd_path', '/home/vm/object.pcd')
        self.segmented_object_pcd = None
        self.segmented_object_points = None
        self.segmented_object_normals = None
        self.object_size = None
        self.bounding_box_center = None
        self.object_bounding_box_corner_points = None
        self.bbp1, self.bbp2, self.bbp3, self.bbp4 = 4 * [None]
        self.bbp5, self.bbp6, self.bbp7, self.bbp8 = 4 * [None]
        self.colors = np.array([
            [0, 0, 0],  #black,       left/front/up
            [1, 0, 0],  #red          right/front/up
            [0, 1, 0],  #green        left/front/down
            [0, 0, 1],  #blue         left/back/up
            [0.5, 0.5, 0.5],  #grey   right/back/down
            [0, 1, 1],  # light blue   left/back/down
            [1, 1, 0],  # yellow      right/back/up
            [1, 0, 1],
        ])

        # Palm poses
        self.palm_goal_poses_world = []
        self.palm_approach_poses_world = []

        self.service_is_called = False

    def nearest_neighbor(self, object_points, point):
        """ Find closest point to "point" in object_points.
            Return closest point and it's normal.
        """
        m = self.segmented_object_points.shape[0]
        center_aug = np.tile(point, (m, 1))
        squared_dist = np.sum(np.square(self.segmented_object_points - center_aug), axis=1)
        min_idx = np.argmin(squared_dist)
        closest_point = self.segmented_object_points[min_idx, :]
        normal = self.segmented_object_normals[min_idx, :]
        center_to_point = point - self.bounding_box_center
        if np.dot(normal, center_to_point) < 0.:
            normal = (-1.) * normal
        normal /= np.linalg.norm(normal)
        return closest_point, normal

    def full_quat_from_normal_tangent_projection(self, normal, orient):
        """ Project the orient vector into tangent space of normal. Normal serves as palm_link x, projected orient as y
        """
        # Get x as negative normal
        x_axis = (-1) * normal

        # Project orientation onto x
        orient_on_x = orient.dot(x_axis) * x_axis

        # Find y as distance between orientation and projection
        y_axis = orient - orient_on_x
        y_axis /= np.linalg.norm(y_axis)

        # Find z as cross product between y and x
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        # Get full rotation matrix
        R = np.matrix([x_axis, y_axis, z_axis]).T

        # Compute quaternion from matrix and return
        T = np.matrix(np.zeros((4, 4)))
        T[:3, :3] = R
        T[3, 3] = 1
        q = tft.quaternion_from_matrix(T)

        return q

    # PART I: Visualize and broadcast information on bounding box and palm poses
    def broadcast_palm_poses(self):
        if self.service_is_called:
            # Publish the palm goal tf
            for i, palm_pose_world in enumerate(self.palm_goal_poses_world):
                self.tf_broadcaster_palm_poses.sendTransform(
                    (palm_pose_world.pose.position.x, palm_pose_world.pose.position.y,
                     palm_pose_world.pose.position.z),
                    (palm_pose_world.pose.orientation.x, palm_pose_world.pose.orientation.y,
                     palm_pose_world.pose.orientation.z, palm_pose_world.pose.orientation.w),
                    rospy.Time.now(), 'heu_' + str(i), palm_pose_world.header.frame_id)

    def publish_points(self, faces_world, color=(1., 0., 0.)):
        rospy.loginfo('Publishing the box center points now!')
        face_centers_world = []
        center_stamped_world = PointStamped()
        center_stamped_world.header.frame_id = 'world'
        for i, face in enumerate(faces_world):
            center_stamped_world.point.x = face.center[0]
            center_stamped_world.point.y = face.center[1]
            center_stamped_world.point.z = face.center[2]
            face_centers_world.append(copy.deepcopy(center_stamped_world))
        markerArray = MarkerArray()

        for i, pnt in enumerate(face_centers_world):
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
        self.bounding_box_center_pub.publish(markerArray)

    def publish_face_centers(self, faces_world):
        center_stamped_world = PointStamped()
        center_stamped_world.header.frame_id = 'world'
        face_centers_world = []
        for i, face in enumerate(faces_world):
            center_stamped_world.point.x = face.center[0]
            center_stamped_world.point.y = face.center[1]
            center_stamped_world.point.z = face.center[2]
            face_centers_world.append(copy.deepcopy(center_stamped_world))

        markerArray = MarkerArray()
        for i, pnt in enumerate(face_centers_world):
            marker = Marker()
            color = self.colors[i, :]
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
        self.bounding_box_face_centers_pub.publish(markerArray)

    def visualize(self, points):
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points)
        pcd_vis.paint_uniform_color([1, 0, 0])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        bb = self.segmented_object_pcd.get_oriented_bounding_box()
        bb.color = (0, 1, 0)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(origin)
        vis.add_geometry(bb)
        vis.add_geometry(self.segmented_object_pcd)
        vis.add_geometry(pcd_vis)
        vis.get_render_option().load_from_json("/home/vm/hand_ws/src/grasp-pipeline/save.json")
        vis.run()
        #vis.get_render_option().save_to_json("save.json")
        print('Done')

    def update_object_information(self):
        """ Update instance variables related to the object of interest

        This is intended to 1.) receive a single message from the segmented_object topics and store them in instance attributes and 
        2.) read the segmented object point cloud from disk
        """
        # Bounding box corner points and center
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        obbcp = np.array(
            rospy.wait_for_message('/segmented_object_bounding_box_corner_points',
                                   Float64MultiArray,
                                   timeout=5).data)
        self.object_bounding_box_corner_points = np.reshape(obbcp, (8, 3))

        self.bbp1 = self.object_bounding_box_corner_points[0, :]
        self.bbp2 = self.object_bounding_box_corner_points[1, :]
        self.bbp3 = self.object_bounding_box_corner_points[2, :]
        self.bbp4 = self.object_bounding_box_corner_points[3, :]
        self.bbp5 = self.object_bounding_box_corner_points[4, :]
        self.bbp6 = self.object_bounding_box_corner_points[5, :]
        self.bbp7 = self.object_bounding_box_corner_points[6, :]
        self.bbp8 = self.object_bounding_box_corner_points[7, :]
        self.bounding_box_center = np.array(0.5 * (self.bbp1 + self.bbp5))

        # Object pcd, store points and normals
        self.segmented_object_pcd = o3d.io.read_point_cloud(self.object_pcd_path)
        self.segmented_object_pcd.normalize_normals()  # normalize the normals
        self.segmented_object_points = np.asarray(self.segmented_object_pcd.points)  # Nx3 shape
        self.segmented_object_normals = np.asarray(self.segmented_object_pcd.normals)

    ##################################################
    ############ Part II Sampling grasps #############
    ##################################################
    def get_oriented_bounding_box_faces(self, grasp_object):
        """ Get the center points of 3 oriented bounding box faces, 1 top and 2 closest to camera.
        """
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        # Also the bounding box axis are aligned to the world frame
        object_T_world = self.listener.fromTranslationRotation(
            (.0, .0, .0), (grasp_object.pose.orientation.x, grasp_object.pose.orientation.y,
                           grasp_object.pose.orientation.z, grasp_object.pose.orientation.w))
        x_axis_world = object_T_world[:3, 0]
        y_axis_world = object_T_world[:3, 1]
        z_axis_world = object_T_world[:3, 2]

        # Get the center from the oriented bounding box

        bb_center_world = self.bounding_box_center

        half_width = 0.5 * grasp_object.width
        half_height = 0.5 * grasp_object.height
        half_depth = 0.5 * grasp_object.depth

        faces_world = [
            BoundingBoxFace(color="black",
                            center=bb_center_world + half_width * x_axis_world,
                            orient_a=y_axis_world,
                            orient_b=z_axis_world,
                            size_a=grasp_object.height,
                            size_b=grasp_object.depth),
            BoundingBoxFace(color="red",
                            center=bb_center_world - half_width * x_axis_world,
                            orient_a=y_axis_world,
                            orient_b=z_axis_world,
                            size_a=grasp_object.height,
                            size_b=grasp_object.depth),
            BoundingBoxFace(color="green",
                            center=bb_center_world + half_height * y_axis_world,
                            orient_a=x_axis_world,
                            orient_b=z_axis_world,
                            size_a=grasp_object.width,
                            size_b=grasp_object.depth),
            BoundingBoxFace(color="blue",
                            center=bb_center_world - half_height * y_axis_world,
                            orient_a=x_axis_world,
                            orient_b=z_axis_world,
                            size_a=grasp_object.width,
                            size_b=grasp_object.depth),
            BoundingBoxFace(color="grey",
                            center=bb_center_world + half_depth * z_axis_world,
                            orient_a=x_axis_world,
                            orient_b=y_axis_world,
                            size_a=grasp_object.width,
                            size_b=grasp_object.height),
            BoundingBoxFace(color="light_blue",
                            center=bb_center_world - half_depth * z_axis_world,
                            orient_a=x_axis_world,
                            orient_b=y_axis_world,
                            size_a=grasp_object.width,
                            size_b=grasp_object.height)
        ]
        # Publish the bounding box face center points for visualization in RVIZ
        self.publish_face_centers(faces_world)

        # find the top face
        faces_world = sorted(faces_world, key=lambda x: x.center[2])
        faces_world[-1].is_top = True

        # Delete the bottom face
        del faces_world[0]

        # Sort along the x axis and delete the face furthes away (robot can't comfortably reach it)
        faces_world = sorted(faces_world, key=lambda x: x.center[0])
        del faces_world[-1]

        # Sort along y axis and delete the face furthest away (no normals in this area)
        faces_world = sorted(faces_world, key=lambda x: x.center[1])
        del faces_world[-1]

        # Publish the bounding box face center points for visualization in RVIZ
        self.publish_points(faces_world)

        # If the object is too short, only select top grasps.
        rospy.loginfo('##########################')
        rospy.loginfo('Obj_height: %s' % grasp_object.height)
        if self.VISUALIZE:
            points_array = np.array([bb.center for bb in faces_world])
            self.visualize(points_array)
        if grasp_object.height < 0.04:
            rospy.loginfo('Object is short, only use top grasps!')
            return [faces_world[0]]

        return faces_world

    def sample_hithand_preshape_joint_state(self):
        """ Sample a joint state for the hithand
        """
        # Generate the mixing weights
        weights = np.random.uniform(0, 1., 4)
        # Eigengrasp comp thumb abduction
        thumb_abd = THUMB_ABD_MIN + weights[0] * (THUMB_ABD_MAX - THUMB_ABD_MIN)

        # Eigengrasp comp finger spread
        finger_spread = SPREAD_MIN + weights[1] * (SPREAD_MAX - SPREAD_MIN)

        # Eigengrasp comp MCP flex
        mcp_flex = MCP_MIN + weights[2] * (MCP_MAX - MCP_MIN)

        # Eigengrasp comp PIP flex
        pip_flex = PIP_MIN + weights[3] * (PIP_MAX - PIP_MIN)

        # Sum up individual contributions
        joint_pos_np = thumb_abd + finger_spread + mcp_flex + pip_flex

        hithand_joint_state = JointState()
        hithand_joint_state.position = list(joint_pos_np)
        return hithand_joint_state

    ################################################
    ############ Part III The service ##############
    ################################################
    def sample_6D_palm_pose_from_face(self, face, from_center):
        """ Given a bounding box face face constructs a 6 D pose either from a point normal close to the center or in the vicinity
            From_center determines if the 6D palm pose comes from object point normal close to bounding box center.
        """
        # Two cases I need to handle: 1. Is top or is side grasp + 2. object point from center or around center
        if from_center:
            if face.is_top:
                point = face.center
                normal = face.center - self.bounding_box_center
                normal /= np.linalg.norm(normal)
            else:
                point, normal = self.nearest_neighbor(self.segmented_object_points, face.center)
        else:
            vec_a = face.size_a / 2 * face.orient_a
            vec_b = face.size_b / 2 * face.orient_b
            point_face = face.center + np.random.uniform((-1) * vec_a, vec_a) + np.random.uniform(
                (-1) * vec_b, vec_b)
            point, normal = self.nearest_neighbor(self.segmented_object_points, point_face)

        # Normal should not point too much into negative Z. Hand won't reach it.
        if normal[2] < 0.3:
            normal[2] = 0

        # Find palm position by going "outward"
        palm_pos = point + np.random.uniform(self.palm_obj_min, self.palm_obj_max) * normal

        # Find approach pos
        approach_pos = palm_pos + self.approach_dist * normal
        approach_pos[2] = point[2] + 0.05

        # Get orientation of longer face side. The y-axis of palm link should be oriented towards this location
        orient = face.get_longer_side_orient()

        # Find orientation of y-axis of palm link by projecting the orient into tanget plane of normal
        palm_q = self.full_quat_from_normal_tangent_projection(normal, orient)

        return (palm_pos, palm_q)

    def handle_generate_hithand_preshape(self, req):
        # Get new information on segmented object from rostopics/disk and store in instance attributes
        self.update_object_information()
        bounding_box_faces = self.get_oriented_bounding_box_faces(req.object)
        # Lists to store generated desired poses
        self.palm_approach_poses_world = []
        self.palm_goal_poses_world = []
        # The grasp sampling now is threefold.
        # 1. Sample N grasps from the 3 bounding box faces closest to the camera
        for face in bounding_box_faces:
            for _ in xrange(self.num_samples_from_bb_face_center):
                palm_pose = self.sample_6D_palm_pose_from_face(face, from_center=True)
                self.palm_goal_poses_world.append(palm_pose)

            for _ in xrange(self.num_samples_around_bb_face_center):
                pass

        for k in xrange(self.num_samples_from_random_object_points):
            pass

        # Finally return the poses
        res = GraspPreshapeResponse()
        res.palm_approach_poses_world = self.palm_approach_poses_world
        res.palm_goal_poses_world = self.palm_goal_poses_world

    def create_hithand_preshape_server(self):
        rospy.Service('generate_hithand_preshape', GraspPreshape,
                      self.handle_generate_hithand_preshape)
        rospy.loginfo('Service generate_hithand_preshape:')
        rospy.loginfo('Ready to generate the grasp preshape.')


if __name__ == '__main__':
    ghp = GenerateHithandPreshape()

    ghp.create_hithand_preshape_server()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        ghp.broadcast_palm_poses()
        rate.sleep()