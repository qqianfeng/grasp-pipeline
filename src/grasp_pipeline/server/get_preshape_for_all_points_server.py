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

from grasp_pipeline.utils import utils
from grasp_pipeline.utils.eigengrasps_hithand import *
from grasp_pipeline.utils.utils import get_pose_stamped_from_trans_and_quat


class BoundingBoxFace():
    """Simple class to store properties of a bounding box face.
    """
    def __init__(self,
                 color,
                 center,
                 orient_a,
                 orient_b,
                 size_a,
                 size_b,
                 is_top=False,
                 face_id=''):
        self.center = np.array(center)
        self.orient_a = orient_a
        self.orient_b = orient_b
        self.size_a = size_a
        self.size_b = size_b
        self.color = color
        self.is_top = is_top
        self.face_id = face_id

    def get_longer_side_orient(self):
        if self.size_a > self.size_b:
            orient = self.orient_a
        elif self.size_a <= self.size_b:
            orient = self.orient_b
        return orient

    def get_shorter_side_size(self):
        if self.size_a <= self.size_b:
            return self.size_a
        elif self.size_a > self.size_b:
            return self.size_b


class GetPreshapeForAllPoints():
    def __init__(self):
        rospy.init_node("get_preshape_for_all_points_node")

        # Publish information about bounding box
        self.bounding_box_center_pub = rospy.Publisher(
            '/box_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points
        self.bounding_box_face_centers_pub = rospy.Publisher(
            '/box_face_center_points', MarkerArray, queue_size=1,
            latch=True)  # publishes the bounding box center points

        # Ros params / hyperparameters
        self.VISUALIZE = rospy.get_param('visualize', False)
        self.palm_obj_min = 0.045  # min dist to object point
        self.palm_obj_max = 0.115  # max dist to object point
        self.approach_dist = 0.1
        self.min_obj_height = 0.05
        self.max_hand_spread = 0.135  # true max spread more like 9 or 9.5cm, object must not be bigger than this
        self.pos_delta_3D = 0.01  # 1cm noise in 3D position
        self.roll_sample_delta = 0.7
        self.pitch_sample_delta = 0.3
        self.yaw_sample_delta = 0.3
        # Publish goal poses over TF
        self.tf_broadcaster_palm_poses = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        # Segmented object vars
        self.object_pcd_path = rospy.get_param('object_pcd_path', '/home/vm/object.pcd')
        self.object_pcd = None
        self.object_points = None
        self.object_normals = None
        self.object_size = None
        self.bounding_box_center = None
        self.object_bounding_box_corner_points = None
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
        self.palm_goal_poses = []
        self.palm_approach_poses = []

        self.service_is_called = False

    def nearest_neighbor(self, object_points, point):
        """ Find closest point to "point" in object_points.
            Return closest point and it's normal.
        """
        m = self.object_points.shape[0]
        center_aug = np.tile(point, (m, 1))
        squared_dist = np.sum(np.square(self.object_points - center_aug), axis=1)
        min_idx = np.argmin(squared_dist)
        closest_point = self.object_points[min_idx, :]
        normal = self.object_normals[min_idx, :]
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
        T = np.matrix(np.zeros((4, 4)))
        T[:3, :3] = R
        T[3, 3] = 1
        # Get the euler angles
        r, p, y = tft.euler_from_matrix(T, 'sxyz')

        # Sample uniformly around euler angles
        r += np.random.uniform(-self.roll_sample_delta,
                               self.roll_sample_delta)  # +-40 degree for roll
        p += np.random.uniform(-self.pitch_sample_delta,
                               self.pitch_sample_delta)  # +-18 degree pitch
        y += np.random.uniform(-self.yaw_sample_delta, self.yaw_sample_delta)  # +-18 degree yaw

        # Compute quaternion from matrix and return
        q = tft.quaternion_from_euler(r, p, y, 'sxyz')

        return q

    def find_face_for_each_point(self, points, faces):
        dists = np.zeros((points.shape[0], len(faces)))

        # Find the distance between each face and each point
        for i, face in enumerate(faces):
            query_points = points - np.tile(face.center, (points.shape[0], 1))
            normal = np.cross(face.orient_a, face.orient_b)
            normal /= np.linalg.norm(normal)
            dists[:, i] = np.abs(query_points.dot(normal))

        # The idx of the closest face for each point gets stored
        closest_face_idxs = np.argmin(dists, axis=1)

        return closest_face_idxs

    def show_closest_face(self, faces, closest_face_idxs):
        if not self.VISUALIZE:
            return
        print(faces[0].face_id)
        idxs = [i for i, x in enumerate(closest_face_idxs) if x == 0]
        self.visualize(self.object_points[idxs, :])

        print(faces[1].face_id)
        idxs = [i for i, x in enumerate(closest_face_idxs) if x == 1]
        self.visualize(self.object_points[idxs, :])

        print(faces[2].face_id)
        idxs = [i for i, x in enumerate(closest_face_idxs) if x == 2]
        self.visualize(self.object_points[idxs, :])

        print(faces[3].face_id)
        idxs = [i for i, x in enumerate(closest_face_idxs) if x == 3]
        self.visualize(self.object_points[idxs, :])

    def show_closest_face_validity(self, pose_pos, faces, closest_face_idx):
        points = np.zeros((2, 3))
        points[0, :] = faces[closest_face_idx[0]].center
        points[1, :] = pose_pos
        self.visualize(points)

    # PART I: Visualize and broadcast information on bounding box and palm poses
    def broadcast_palm_poses(self):
        if self.service_is_called:
            # Publish the palm goal tf
            for i, palm_pose_world in enumerate(self.palm_goal_poses):
                self.tf_broadcaster_palm_poses.sendTransform(
                    (palm_pose_world.pose.position.x, palm_pose_world.pose.position.y,
                     palm_pose_world.pose.position.z),
                    (palm_pose_world.pose.orientation.x, palm_pose_world.pose.orientation.y,
                     palm_pose_world.pose.orientation.z, palm_pose_world.pose.orientation.w),
                    rospy.Time.now(), 'heu_' + str(i), palm_pose_world.header.frame_id)

    def publish_points(self, faces_world, publisher, color=(1., 0., 0.)):
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
        if publisher == 'box_center':
            self.bounding_box_center_pub.publish(markerArray)
        elif publisher == 'face_centers':
            self.bounding_box_face_centers_pub.publish(markerArray)

    def visualize(self, points):
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points)
        pcd_vis.paint_uniform_color([1, 0, 0])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        bb = self.object_pcd.get_oriented_bounding_box()
        bb.color = (0, 1, 0)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(origin)
        vis.add_geometry(bb)
        vis.add_geometry(self.object_pcd)
        vis.add_geometry(pcd_vis)
        vis.get_render_option().load_from_json("/home/vm/hand_ws/src/grasp-pipeline/save.json")
        vis.run()
        #vis.get_render_option().save_to_json("save.json")

    def update_object_information(self):
        """ Update instance variables related to the object of interest

        This is intended to 1.) receive a single message from the object topics and store them in instance attributes and 
        2.) read the segmented object point cloud from disk
        """
        # Bounding box corner points and center
        # The 1. and 5. point of bounding_box_corner points are cross-diagonal
        obbcp = np.array(
            rospy.wait_for_message('/segmented_object_bounding_box_corner_points',
                                   Float64MultiArray,
                                   timeout=5).data)
        self.object_bounding_box_corner_points = np.reshape(obbcp, (8, 3))

        bbp1 = self.object_bounding_box_corner_points[0, :]
        bbp5 = self.object_bounding_box_corner_points[4, :]
        self.bounding_box_center = np.array(0.5 * (bbp1 + bbp5))

        # Object pcd, store points and normals
        self.object_pcd = o3d.io.read_point_cloud(self.object_pcd_path)
        self.object_pcd.normalize_normals()  # normalize the normals
        self.object_points = np.asarray(self.object_pcd.points)  # Nx3 shape
        self.object_normals = np.asarray(self.object_pcd.normals)

    ##################################################
    ############ Part II Sampling grasps #############
    ##################################################
    def get_oriented_bounding_box_faces(self, grasp_object, is_check_validity=False):
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
        self.publish_points(faces_world, publisher='face_centers')

        # find the top face
        faces_world = sorted(faces_world, key=lambda x: x.center[2])
        faces_world[-1].is_top = True
        faces_world[-1].face_id = 'top'

        # Delete the bottom face
        del faces_world[0]

        # Sort along y axis and delete the face furthest away (no normals in this area)
        faces_world = sorted(faces_world, key=lambda x: x.center[1])
        face_back = copy.deepcopy(faces_world[-1])
        del faces_world[-1]

        # Label front face with 'side1'
        face_idx_min_y = np.argmin([face.center[1] for face in faces_world])
        faces_world[face_idx_min_y].face_id = 'side1'

        # Label left side face with 'side2'
        face_idx_min_x = np.argmin([face.center[0] for face in faces_world])
        faces_world[face_idx_min_x].face_id = 'side2'

        # Publish the bounding box face center points for visualization in RVIZ
        self.publish_points(faces_world, publisher='box_center')

        if is_check_validity:
            faces_world.append(face_back)

        if self.VISUALIZE:
            points_array = np.array([bb.center for bb in faces_world])
            self.visualize(points_array)

        return faces_world

    def sample_hithand_joint_state(self, max_weight=1.):
        """ Sample a joint state for the hithand
        """
        # Generate the mixing weights
        weights = np.random.uniform(0.25, max_weight, 4)

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
    def sample_6D_palm_pose_from_point(self, point, normal, face):
        # Make sure the normal is normalized
        normal /= np.linalg.norm(normal)

        # Add distance outward in normal direction
        palm_pos = point + np.random.uniform(self.palm_obj_min, self.palm_obj_max) * normal

        # Add small 3D noise
        palm_pos += np.random.uniform(-self.pos_delta_3D, self.pos_delta_3D, 3)

        # Find the palm orientation as the longer side
        orient = face.get_longer_side_orient()

        # Thumb gets aligned to orient, should not point down too much
        if orient[2] < -0.2:
            orient = (-1) * orient

        # Find orientation of y-axis of palm link by projecting the orient into tanget plane of normal
        palm_q = self.full_quat_from_normal_tangent_projection(normal, orient)

        # For approach pos normal should not point down too much
        if normal[2] < -0.2:
            normal[2] = 0

        # Find approach pos
        approach_pos = palm_pos + self.approach_dist * normal
        approach_pos[2] = point[2] + 0.05

        # Approach orient should be same as palm
        approach_q = palm_q

        # Transform pos and q to pose stamped
        palm_pose = get_pose_stamped_from_trans_and_quat(palm_pos, palm_q)
        approach_pose = get_pose_stamped_from_trans_and_quat(approach_pos, approach_q)

        return (palm_pose, approach_pose)

    def handle_get_preshape_for_all_points(self, req):
        # Get new information on segmented object from rostopics/disk and store in instance attributes
        self.update_object_information()

        # Get all bounding box faces
        faces = self.get_oriented_bounding_box_faces(req.object)

        # For each point in pointcloud find closest bounding box face
        closest_face_idxs = self.find_face_for_each_point(self.object_points, faces)

        # Visualize the closest face
        self.show_closest_face(faces, closest_face_idxs)

        # Store the index of the face furthest away in x direction. Corresponding points are excluded since not reachable.
        face_idx_max_x = np.argmax([face.center[0] for face in faces])

        # Lists to store relevant information
        self.palm_approach_poses = []
        self.palm_goal_poses = []
        is_tops = []
        joint_states = []
        face_ids = []

        ob = o3d.io.read_point_cloud('/home/vm/object.pcd')

        # Sample 6D pose for each point
        for i, (point, normal) in enumerate(zip(self.object_points, self.object_normals)):
            # Get the face closest to point
            face = faces[closest_face_idxs[i]]

            # Exclude imporbable points
            if (closest_face_idxs[i] == face_idx_max_x) or (
                    point[2] < self.min_obj_height
                    and face.face_id != 'top') or ((face.size_a > self.max_hand_spread) and
                                                   (face.size_b > self.max_hand_spread)):
                continue

            # If face is top append
            is_tops.append(face.is_top)

            # Get 6D pose for palm and approach pose from point and normal, sample along normal, sample 3D orientation, sample 3D position
            (palm_pose, approach_pose) = self.sample_6D_palm_pose_from_point(point, normal, face)

            # Finally also sample a hithand joint state
            max_weight = -0.08 * face.get_shorter_side_size() + 1.16
            joint_state = self.sample_hithand_joint_state(max_weight=max_weight)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            hom = utils.hom_matrix_from_pose_stamped(palm_pose)
            frame.transform(hom)
            o3d.visualization.draw_geometries([frame, ob])

            # Append
            self.palm_goal_poses.append(palm_pose)
            self.palm_approach_poses.append(approach_pose)
            joint_states.append(joint_state)
            face_ids.append(face.face_id)

        # Finally return the poses
        res = GraspPreshapeResponse()
        res.palm_approach_poses_world = self.palm_approach_poses
        res.palm_goal_poses_world = self.palm_goal_poses
        res.hithand_joint_states = joint_states
        res.is_top_grasp = is_tops
        res.face_ids = face_ids

        # Set true to enable publishing
        self.service_is_called = True

        return res

    def handle_check_pose_validity(self, req):
        # Extract only the position of the query pose
        grasp_pos = np.array([[
            req.pose.pose.position.x, req.pose.pose.position.y, req.pose.pose.position.z
        ]])

        # Get new information on segmented object from rostopics/disk and store in instance attributes
        self.update_object_information()

        # Get all bounding box faces
        faces = self.get_oriented_bounding_box_faces(req.object, is_check_validity=True)

        # Remove the top face
        faces = sorted(faces, key=lambda x: x.center[2])
        del faces[-1]

        closest_pcd_point, _ = self.nearest_neighbor(self.object_points, grasp_pos[0, :])

        # Bring point to correct format
        cpd = np.zeros((1, 3))
        cpd[0, :] = closest_pcd_point

        # Find closest face to point cloud point closest to grasp pos
        closest_face_idx = self.find_face_for_each_point(cpd, faces)

        # Visualize the closest face
        self.show_closest_face_validity(cpd, faces, closest_face_idx)

        # Return
        res = CheckPoseValidityResponse()
        if faces[closest_face_idx[0]].face_id in ['side1', 'side2']:
            res.is_valid = True
        else:
            res.is_valid = False

        return res

    def create_get_preshape_for_all_points_server(self):
        rospy.Service('get_preshape_for_all_points', GraspPreshape,
                      self.handle_get_preshape_for_all_points)
        rospy.loginfo('Service get_preshape_for_all_points:')
        rospy.loginfo('Ready to generate the grasp preshape.')

    def create_check_pose_validity_utah_server(self):
        rospy.Service('check_pose_validity_utah', CheckPoseValidity,
                      self.handle_check_pose_validity)
        rospy.loginfo('Service check_pose_validity_utah:')
        rospy.loginfo('Ready to check_pose_validity_utah')


if __name__ == '__main__':
    ghp = GetPreshapeForAllPoints()

    ghp.create_get_preshape_for_all_points_server()
    ghp.create_check_pose_validity_utah_server()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        ghp.broadcast_palm_poses()
        rate.sleep()