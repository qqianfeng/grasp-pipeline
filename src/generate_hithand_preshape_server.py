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


class GenerateHithandPreshape():
    """ Generates Preshapes for the Hithand via sampling or geometric considerations.

    It stores the object_size, bounding_box_corner_points and objct_pcd in instance variables
    A service for sampling a Hithand preshape is offered which takes in the object information and samples the palm 6D pose and finger configuration.
    """
    def __init__(self):
        # Init node and publishers
        rospy.init_node("generate_hithand_preshape_node")
        self.service_is_called = False

        self.bounding_box_center_pub = rospy.Publisher(
            '/box_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points
        self.bounding_box_face_centers_pub = rospy.Publisher(
            '/box_face_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points

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

        self.tf_broadcaster_palm_poses = tf.TransformBroadcaster()
        # Get parameters from the ROS parameter server
        self.palm_dist_to_top_face_mean = rospy.get_param(
            'generate_hithand_preshape_server_node/palm_dist_to_top_face_mean', 0.1)
        self.palm_dist_to_side_face_mean = rospy.get_param(
            'generate_hithand_preshape_server_node/palm_dist_to_side_face_mean', 0.07)
        self.palm_dist_normal_to_obj_var = rospy.get_param(
            'generate_hithand_preshape_server_node/palm_dist_normal_to_obj_var', 0.03)
        self.palm_position_3D_sample_var = rospy.get_param(
            'generate_hithand_preshape_server_node/palm_position_3D_sample_var', 0.005)
        self.wrist_roll_orientation_var = rospy.get_param(
            'generate_hithand_preshape_server_node/wrist_roll_orientation_var', 0.005)
        self.min_object_height = rospy.get_param(
            'generate_hithand_preshape_server_node/min_object_height', 0.03)
        self.num_samples_per_preshape = rospy.get_param(
            'generate_hithand_preshape_server_node/num_samples_per_preshape', 50)
        self.object_pcd_path = rospy.get_param('object_pcd_path', '/home/vm/object.pcd')
        self.VISUALIZE = rospy.get_param('visualize', False)
        print(self.num_samples_per_preshape)
        self.use_bb_orient_to_determine_wrist_roll = True
        self.listener = tf.TransformListener()
        # Initialize object related instance variables
        self.segmented_object_pcd = None
        self.segmented_object_points = None
        self.segmented_object_normals = None
        self.object_size = None
        self.bounding_box_center = None
        self.object_bounding_box_corner_points = None
        self.bbp1, self.bbp2, self.bbp3, self.bbp4 = 4 * [None]
        self.bbp5, self.bbp6, self.bbp7, self.bbp8 = 4 * [None]

        # This stores the 6D palm pose in the world
        self.palm_goal_pose_world = []

        # Stores the approach pose
        self.palm_approach_pose_world = []
        self.approach_offset = 0.2

        self.palm_pose_lower_limit = None
        self.palm_pose_upper_limit = None

        self.palm_rand_pose_parameter_pos = rospy.get_param(
            "generate_hithand_preshape_server_node/palm_rand_pose_sample_pos")
        self.palm_rand_pose_parameter_orient = rospy.get_param(
            "generate_hithand_preshape_server_node/palm_rand_pose_sample_orient"
        )  # percentage of pi. 1 means sample +-180 degrees around initial palm pose

    # ++++++++++++++++++++++ PART I: Helper/initialization functions +++++++++++++++++++++++
    def set_palm_rand_pose_limits(self, palm_preshape_pose):
        """ Set the palm pose sample range for sampling grasp detection.
        """
        # Convert the pose into a format more amenable to subsequent task
        palm_preshape_euler = tft.euler_from_quaternion(
            (palm_preshape_pose.pose.orientation.x, palm_preshape_pose.pose.orientation.y,
             palm_preshape_pose.pose.orientation.z, palm_preshape_pose.pose.orientation.w))

        preshape_palm_pose_config = np.array([
            palm_preshape_pose.pose.position.x, palm_preshape_pose.pose.position.y,
            palm_preshape_pose.pose.position.z, palm_preshape_euler[0], palm_preshape_euler[1],
            palm_preshape_euler[2]
        ])

        # Add/ subtract these from the pose to get lower and upper limits
        pos_range = self.palm_rand_pose_parameter_pos  #0.05
        ort_range = self.palm_rand_pose_parameter_orient * np.pi  #0.05
        # upper_limit_range = np.array([
        #     pos_range, pos_range, pos_range, ort_range, ort_range, ort_range
        # ])
        upper_limit_range = np.array([pos_range, pos_range, pos_range, ort_range, 0, 0])
        lower_limit_range = (-1) * upper_limit_range
        self.palm_pose_lower_limit = preshape_palm_pose_config + lower_limit_range
        self.palm_pose_upper_limit = preshape_palm_pose_config + upper_limit_range

    # ++++++++++++++++++++++++++++ PART II: Publishers ++++++++++++++++++++++++++++++++++++
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

    def broadcast_palm_poses(self):
        if self.service_is_called:
            # Publish the palm goal tf
            for i, palm_pose_world in enumerate(self.palm_goal_pose_world):
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
        #return
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points)
        pcd_vis.paint_uniform_color([1, 0, 0])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        bb = self.segmented_object_pcd.get_oriented_bounding_box()
        bb.color = (0, 1, 0)

        #o3d.visualization.draw_geometries(
        #    [origin, bb, self.segmented_object_pcd, pcd_vis])

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

    # +++++++++++++++++++++ PART III: Sampling and Bounding Box +++++++++++++++++++++++++++
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

    def sample_uniformly_around_preshape_palm_pose(self, frame_id):
        ''' Get a random palm pose by sampling around the preshape palm pose
        for sampling grasp detection. 

        The palm pose limit instance variables define the sampling range (middle between upper and lower limits is the previously computed palm pose)
        '''
        sample_palm_pose_array = np.random.uniform(self.palm_pose_lower_limit,
                                                   self.palm_pose_upper_limit)
        sample_palm_pose = PoseStamped()
        sample_palm_pose.header.frame_id = frame_id
        sample_palm_pose.pose.position.x, sample_palm_pose.pose.position.y, \
                sample_palm_pose.pose.position.z = sample_palm_pose_array[:3]

        palm_euler = sample_palm_pose_array[3:]
        palm_quaternion = tft.quaternion_from_euler(palm_euler[0], palm_euler[1], palm_euler[2])
        sample_palm_pose.pose.orientation.x = palm_quaternion[0]
        sample_palm_pose.pose.orientation.y = palm_quaternion[1]
        sample_palm_pose.pose.orientation.z = palm_quaternion[2]
        sample_palm_pose.pose.orientation.w = palm_quaternion[3]

        return sample_palm_pose

    def find_full_palm_orientation(self, object_point_normal, bounding_box_orientation_vector):
        """ Finds the full palm orientation py projecting the bounding box orientation vector 
        (which gives the rough direction for the thumb, or more accurately the y-axis of the palm_link_hithand)
        by projecting into into the tangent space of the object point normal. 

        For the palm frame, x: palm normal, y: thumb, z: middle finger
        """
        if self.use_bb_orient_to_determine_wrist_roll:
            y_rand = bounding_box_orientation_vector
        else:
            roll = np.random.uniform(-np.pi, np.pi)
            y_rand = np.array([0.0, np.sin(roll), np.cos(roll)])

        # Project orientation vector into the tangent space of the normal x tp get the y component, then determine z from the cross-product of the two
        x_axis = -object_point_normal
        y_onto_x = y_rand.dot(x_axis) * x_axis
        y_axis = (y_rand - y_onto_x) / np.linalg.norm(y_rand - y_onto_x)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        rot_matrix = np.matrix([x_axis, y_axis, z_axis]).T

        # Compute quaternion from rpy
        trans_matrix = np.matrix(np.zeros((4, 4)))
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[3, 3] = 1.
        quaternion = tft.quaternion_from_matrix(trans_matrix)
        rospy.loginfo('Grasp orientation = ' + str(quaternion))
        return quaternion

    def find_palm_pose_from_bounding_box_face(self, bounding_box_face):
        """ Finds the object point cloud's point closest to the center of a given bounding box face
        and generates a Hithand palm pose given the closest point's surface normal. For top faces the normal is not taken from the object pointcloud but computed as the distance between bounding box center and top face center
        """
        palm_poses = []
        palm_approach_poses = []
        for i in range(3):
            if bounding_box_face.is_top:
                rospy.loginfo('***Top.')
                # Part I: Sample position of palm pose
                top_face_normal = bounding_box_face.center - self.bounding_box_center  # compute the normal for top faces as the vector pointing from the pointing box center to the top face center
                top_face_normal /= np.linalg.norm(top_face_normal)
                closest_point_normal = top_face_normal  # This variable is needed later, after if/else
                if i == 0:
                    palm_dist_top_in_normal_direction = np.random.normal(
                        self.palm_dist_to_top_face_mean, self.palm_dist_normal_to_obj_var
                    )  # sample palm distance from normal distribution centered around mean_dist_top
                else:
                    palm_dist_side_normal_direction = np.random.uniform(
                        self.palm_dist_to_side_face_mean - self.palm_dist_normal_to_obj_var,
                        self.palm_dist_to_side_face_mean + self.palm_dist_normal_to_obj_var)
                palm_position = bounding_box_face.center + palm_dist_top_in_normal_direction * top_face_normal

                # Also compute an approach position which is 20cm further away from the palm position outward in object normal direction
                # palm_approach_position = palm_position + self.approach_offset * top_face_normal
                palm_approach_position = copy.deepcopy(palm_position)
                palm_approach_position[2] += 0.2

                # Part II: Compute vector to determine orientation of palm pose
                # For our setup and with the chosen palm frame a successful top grasp should roughly align the palm link y axis (green in RViz) with the long side of the bounding box orientation vector
                if bounding_box_face.size_a > bounding_box_face.size_b:
                    bounding_box_orientation_vector = bounding_box_face.orient_a
                elif bounding_box_face.size_a < bounding_box_face.size_b:
                    bounding_box_orientation_vector = bounding_box_face.orient_b
                else:  # Account for fact that size_a and size_b could be same
                    bb_extent = np.ones(4)
                    bb_extent[:3] = np.array(
                        self.segmented_object_pcd.get_oriented_bounding_box().extent)
                    (_, quat) = self.listener.lookupTransform('object_pose', 'object_pose_aligned',
                                                              rospy.Time())
                    aligned_T_pose = tft.quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])
                    bb_extent_aligned = np.abs(aligned_T_pose.dot(bb_extent))
                    if bb_extent_aligned[0] > bb_extent_aligned[1]:
                        bounding_box_orientation_vector = bounding_box_face.orient_a
                    else:
                        bounding_box_orientation_vector = bounding_box_face.orient_b

                # the x component of palm frame orientation should be positive
                # if bounding_box_orientation_vector[0] < 0:
                #     bounding_box_orientation_vector = (-1) * bounding_box_orientation_vector
            else:
                # Part I: Sample the position of the palm pose
                #find the closest point in the point cloud
                m = self.segmented_object_points.shape[0]
                center_aug = np.tile(bounding_box_face.center, (m, 1))
                squared_dist = np.sum(np.square(self.segmented_object_points - center_aug), axis=1)
                min_idx = np.argmin(squared_dist)
                closest_point = self.segmented_object_points[min_idx, :]
                rospy.loginfo('Found closest point ' + str(closest_point) + ' to obj_pt = ' +
                              str(bounding_box_face.center) + ' at dist = ' +
                              str(squared_dist[min_idx]))

                # Get normal of closest point
                closest_point_normal = self.segmented_object_normals[min_idx, :]
                rospy.loginfo('Associated normal n = ' + str(closest_point_normal))
                # make sure the normal actually points outwards
                center_to_face = bounding_box_face.center - self.bounding_box_center
                if np.dot(closest_point_normal, center_to_face) < 0.:
                    closest_point_normal = (-1.) * closest_point_normal
                # Sample hand position
                palm_dist_side_normal_direction = np.random.normal(
                        self.palm_dist_to_side_face_mean, self.palm_dist_normal_to_obj_var)
                else:
                    palm_dist_side_normal_direction = np.random.uniform(
                        self.palm_dist_to_side_face_mean - self.palm_dist_normal_to_obj_var,
                        self.palm_dist_to_side_face_mean + self.palm_dist_normal_to_obj_var)
                palm_position = closest_point + palm_dist_side_normal_direction * closest_point_normal

                # Also compute an approach position which is 20cm further away from the palm position outward in object normal direction
                palm_approach_position = closest_point + (palm_dist_side_normal_direction +
                                                          0.15) * closest_point_normal
                palm_approach_position[2] += 0.2

                # Part II: Compute vector to determine the palm orientation
                # Pick the vertical vector with larger z value to be the palm link y axis (thumb)
                # in order to remove the orientation that runs into the table
                if abs(bounding_box_face.orient_a[2]) > abs(bounding_box_face.orient_b[2]):
                    bounding_box_orientation_vector = bounding_box_face.orient_a[:]
                else:
                    bounding_box_orientation_vector = bounding_box_face.orient_b[:]

                if bounding_box_orientation_vector[2] < 0:
                    bounding_box_orientation_vector = (-1.) * bounding_box_orientation_vector

            # Add some extra 3D noise on the palm position
            if i == 0:
                position_noise_3d = np.random.normal(0, self.palm_position_3D_sample_var, 3)
                palm_position = palm_position + position_noise_3d
                palm_approach_position = palm_approach_position + position_noise_3d

            # Add some noise to the palm orientation
            bounding_box_orientation_vector = bounding_box_orientation_vector + np.random.normal(
                0., self.wrist_roll_orientation_var, 3)
            bounding_box_orientation_vector /= np.linalg.norm(bounding_box_orientation_vector)

            # Find the full palm orientation from the bounding box orientation vector (which is the desired vector for the palm link y-direction) and object point normal
            palm_orientation_quat = self.find_full_palm_orientation(
                closest_point_normal, bounding_box_orientation_vector)

            # Put this into a Stamped Pose for publishing
            palm_pose = PoseStamped()
            palm_pose.header.frame_id = 'world'
            # palm_pose.header.stamp = rospy.Time.now()
            palm_pose.pose.position.x = palm_position[0]
            palm_pose.pose.position.y = palm_position[1]
            palm_pose.pose.position.z = palm_position[2]
            palm_pose.pose.orientation.x = palm_orientation_quat[0]
            palm_pose.pose.orientation.y = palm_orientation_quat[1]
            palm_pose.pose.orientation.z = palm_orientation_quat[2]
            palm_pose.pose.orientation.w = palm_orientation_quat[3]
            rospy.loginfo('Chosen palm position is ' + str(palm_pose.pose.position))

            # Construct the approach pose which is 20cm outward from palm goal pose in normal direction
            approach_pose = copy.deepcopy(palm_pose)
            approach_pose.pose.position.x = palm_approach_position[0]
            approach_pose.pose.position.y = palm_approach_position[1]
            approach_pose.pose.position.z = palm_approach_position[2]

            palm_poses.append(palm_pose)
            palm_approach_poses.append(approach_pose)

        return palm_poses, palm_approach_poses

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
        if grasp_object.height < self.min_object_height:
            rospy.loginfo('Object is short, only use top grasps!')
            return [faces_world[0]]

        return faces_world

    # ++++++++++++++++++ PART IV: Main service logic +++++++++++++++++++++++
    def sample_grasp_preshape(self, grasp_object):
        """ Grasp preshape service callback for sampling Hithand grasp preshapes.
        """
        rospy.loginfo('Sampling grasp preshapes')
        response = GraspPreshapeResponse()
        # Compute bounding box faces
        bounding_box_faces = self.get_oriented_bounding_box_faces(grasp_object)

        # Save the goal poses for the palm and an approach pose
        self.palm_goal_pose_world = []
        self.palm_approach_pose_world = []
        for i in xrange(len(bounding_box_faces)):
            # Get the desired palm pose given the point from the bounding box
            palm_poses_world, palm_approach_poses_world = self.find_palm_pose_from_bounding_box_face(
                bounding_box_faces[i])  # Tested, seems to work fine, only visualized position
            for k in xrange(len(palm_poses_world)):
                palm_pose_world = palm_poses_world[k]
                palm_approach_pose_world = palm_approach_poses_world[k]
                if self.VISUALIZE:
                    point = np.zeros([3, 2])
                    point[0, 1] = palm_pose_world.pose.position.x
                    point[1, 1] = palm_pose_world.pose.position.y
                    point[2, 1] = palm_pose_world.pose.position.z
                    self.visualize(point.T)
                self.palm_goal_pose_world.append(palm_pose_world)
                response.palm_goal_pose_world.append(palm_pose_world)

                self.palm_approach_pose_world.append(palm_approach_pose_world)
                response.palm_approach_pose_world.append(palm_approach_pose_world)

                response.hithand_joint_state.append(self.sample_hithand_preshape_joint_state())
                response.is_top_grasp.append(bounding_box_faces[i].is_top)
                # Set the rand pose limits for subsequent uniform sampling around the previously found palm_pose_world as initial pose.
                self.set_palm_rand_pose_limits(palm_pose_world)
                if self.VISUALIZE:
                    point = np.zeros([3, self.num_samples_per_preshape])

                for j in xrange(self.num_samples_per_preshape):
                    sampled_palm_pose = self.sample_uniformly_around_preshape_palm_pose(
                        palm_pose_world.header.frame_id)
                    response.palm_goal_pose_world.append(sampled_palm_pose)
                    self.palm_goal_pose_world.append(sampled_palm_pose)
                    response.palm_approach_pose_world.append(palm_approach_pose_world)
                    self.palm_approach_pose_world.append(palm_approach_pose_world)

                    if self.VISUALIZE:
                        point[0, j] = sampled_palm_pose.pose.position.x
                        point[1, j] = sampled_palm_pose.pose.position.y
                        point[2, j] = sampled_palm_pose.pose.position.z
                    # Sample remaining joint values
                    hithand_joint_state = self.sample_hithand_preshape_joint_state()
                    response.hithand_joint_state.append(hithand_joint_state)
                    response.is_top_grasp.append(bounding_box_faces[i].is_top)
                if self.VISUALIZE:
                    self.visualize(point.T)
        self.service_is_called = True

        return response

    def handle_generate_hithand_preshape(self, req):
        """ Handler for the advertised service. Decides whether the hand preshape should be sampled or generated
        """
        rospy.loginfo('I received the service call to generate hithand preshapes.')
        # Get new information on segmented object from rostopics/disk and store in instance attributes
        self.update_object_information()
        if req.sample:
            return self.sample_grasp_preshape(req.object)
        else:
            raise NotImplementedError

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