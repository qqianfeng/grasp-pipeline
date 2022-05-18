#!/usr/bin/env python
import numpy as np
import rospy
import tf.transformations as tft
import torch

from FFHNet.models.ffhnet import FFHNet
from FFHNet.config.eval_config import EvalConfig
from geometry_msgs.msg import PoseStamped
from grasp_pipeline.srv import *
from grasp_pipeline.utils import utils
from sensor_msgs.msg import JointState


class GraspInference():
    def __init__(self):
        rospy.init_node('grasp_inference_node')
        cfg = EvalConfig().parse()
        self.load_path = rospy.get_param('ffhnet_load_path')
        self.load_epoch = rospy.get_param('ffhnet_load_epoch')
        self.FFHNet = FFHNet(cfg)
        # self.FFHNet.load_ffhgenerator(epoch=self.load_epoch, load_path=self.load_path)
        self.FFHNet.load_ffhgenerator(epoch=self.load_epoch, load_path='/home/vm/hand_ws/src/ffhnet/models/ffhgenerator'
)

        self.FFHNet.load_ffhevaluator(
            epoch=30,
            load_path='/home/vm/hand_ws/src/ffhnet/models/ffhevaluator')

    def build_pose_list(self, rot_matrix, transl, frame_id='object_centroid_vae'):
        assert rot_matrix.shape[1:] == (3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_matrix.shape[0] == transl.shape[
            0], "Batch dimension of rot and trans not equal."

        poses = []

        for i in range(rot_matrix.shape[0]):
            rot_hom = utils.hom_matrix_from_rot_matrix(rot_matrix[i, :, :])
            quat = tft.quaternion_from_matrix(rot_hom)
            t = transl[i, :]

            pose_st = PoseStamped()
            pose_st.header.frame_id = frame_id
            pose_st.pose.position.x = t[0]
            pose_st.pose.position.y = t[1]
            pose_st.pose.position.z = t[2]
            pose_st.pose.orientation.x = quat[0]
            pose_st.pose.orientation.y = quat[1]
            pose_st.pose.orientation.z = quat[2]
            pose_st.pose.orientation.w = quat[3]

            poses.append(pose_st)

        return poses

    def build_joint_conf_list(self, joint_conf):
        joint_confs = []
        for i in range(joint_conf.shape[0]):
            jc = JointState()
            jc.position = joint_conf[i, :]
            joint_confs.append(jc)
        return joint_confs

    def handle_infer_grasp_poses(self, req):
        # reshape
        bps_object = np.load('/home/vm/pcd_enc.npy')
        n_samples = req.n_poses
        results = self.FFHNet.generate_grasps(bps_object, n_samples=n_samples, return_arr=True)

        # prepare response
        res = InferGraspPosesResponse()
        res.palm_poses = self.build_pose_list(results['rot_matrix'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

    def to_grasp_dict(self, palm_poses, joint_confs):
        """Take the palm_poses and joint_confs in ros-format and convert them to a dict with two 3D arrays in order to
        use as input for FFHEvaluator 

        Args:
            palm_poses (List of PoseStamped): List of Palm poses in object frame
            joint_confs (List of JointState): List of joint states belonging to grasp

        Returns:
            grasp_dict (dict): k1 is 
        """
        # prepare
        batch_n = len(palm_poses)
        joint_arr = np.zeros((batch_n, 15))
        rot_matrix_arr = np.zeros((batch_n, 3, 3))
        transl_arr = np.zeros((batch_n, 3))

        # convert
        for i, (palm_pose, joint_conf) in enumerate(zip(palm_poses, joint_confs)):
            q = palm_pose.pose.orientation
            t = palm_pose.pose.position

            rot_matrix_arr[i, :, :] = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            transl_arr[i, :] = [t.x, t.y, t.z]
            joint_arr[i, :] = np.array(joint_conf.position)

        # Build grasp dict
        grasp_dict = {}
        grasp_dict['rot_matrix'] = rot_matrix_arr
        grasp_dict['transl'] = transl_arr
        grasp_dict['joint_conf'] = joint_arr

        return grasp_dict

    def handle_evaluate_and_filter_grasp_poses(self, req):
        bps_object = np.load('/home/vm/pcd_enc.npy')
        grasp_dict = self.to_grasp_dict(req.palm_poses, req.joint_confs)
        results = self.FFHNet.filter_grasps(bps_object, grasp_dict, thresh=req.thresh)

        # prepare response
        res = EvaluateAndFilterGraspPosesResponse()
        res.palm_poses = self.build_pose_list(results['rot_matrix'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

    def handle_evaluate_grasp_poses(self, req):
        bps_object = np.load('/home/vm/pcd_enc.npy')
        # Build a dict with all the grasps
        p_success = self.FFHNet.evaluate_grasps(bps_object, grasps, return_arr=True)

    def create_infer_grasp_poses_server(self):
        rospy.Service('infer_grasp_poses', InferGraspPoses, self.handle_infer_grasp_poses)
        rospy.loginfo('Service infer_grasp_poses')
        rospy.loginfo('Ready to sample grasps from FFHGenerator.')

    def create_evaluate_grasp_poses_server(self):
        rospy.Service('evaluate_grasp_poses', EvaluateGraspPoses, self.handle_evaluate_grasp_poses)
        rospy.loginfo('Service evaluate_grasp_poses')
        rospy.loginfo('Ready to evaluate grasps with the FFHEvaluator.')

    def create_evaluate_and_filter_grasp_poses_server(self):
        rospy.Service('evaluate_and_filter_grasp_poses', EvaluateAndFilterGraspPoses,
                      self.handle_evaluate_and_filter_grasp_poses)
        rospy.loginfo('Service evaluate_and_filter_grasp_poses')
        rospy.loginfo('Ready to evaluate and filter grasps with the FFHEvaluator.')


if __name__ == '__main__':
    gi = GraspInference()
    gi.create_infer_grasp_poses_server()
    gi.create_evaluate_and_filter_grasp_poses_server()
    rospy.spin()
