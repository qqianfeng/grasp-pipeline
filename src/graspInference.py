#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import os
import sys
import copy

# Add FFHNet to the path
sys.path.append(os.environ['FFHNET_PATH'])

from FFHNet.models.ffhnet import FFHNet
from FFHNet.config.eval_config import EvalConfig
from FFHNet.utils import visualization

from data_types import *

# TODO: add visualization to the inference step with module from FFHNet

class GraspInference():
    def __init__(self):
        cfg = EvalConfig().parse()
        self.VISUALIZE = os.environ.get('VISUALIZE')
        self.pcd_path = os.environ['OBJECT_PCD_PATH']

        self.FFHNet = FFHNet(cfg)
        ffhnet_path = os.environ['FFHNET_PATH']
        self.FFHNet.load_ffhgenerator(epoch=10, load_path=os.path.join(ffhnet_path, 'models/ffhgenerator'))
        self.FFHNet.load_ffhevaluator(
            epoch=30,
            load_path=os.path.join(ffhnet_path, 'models/ffhevaluator'))

    def build_pose_list(self, rot_matrix, transl, frame_id='object_centroid_vae'):
        assert rot_matrix.shape[1:] == (
            3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_matrix.shape[0] == transl.shape[
            0], "Batch dimension of rot and trans not equal."

        poses = []

        for i in range(rot_matrix.shape[0]):
            r = R.from_matrix(rot_matrix[i, :, :])
            quat = r.as_quat()
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

            poses.append(copy.deepcopy(pose_st))

        return poses

    def build_joint_conf_list(self, joint_conf):
        joint_confs = []
        for i in range(joint_conf.shape[0]):
            jc = JointState()
            jc.position = joint_conf[i, :]
            joint_confs.append(jc)
        return joint_confs

    def handle_infer_grasp_poses(self, n_poses):
        # reshape
        bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
        bps_object = bps_object_center[:, 6:]
        n_samples = n_poses
        results = self.FFHNet.generate_grasps(
            bps_object, n_samples=n_samples, return_arr=True)

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        # prepare response
        palm_poses = self.build_pose_list(
            results['rot_matrix'], results['transl'])
        joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return palm_poses, joint_confs

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

            r = R.from_quat(np.array([q.x, q.y, q.z, q.w]))
            rot_matrix_arr[i, :, :] = r.as_matrix()
            transl_arr[i, :] = [t.x, t.y, t.z]
            joint_arr[i, :] = np.array(joint_conf.position)

        # Build grasp dict
        grasp_dict = {}
        grasp_dict['rot_matrix'] = rot_matrix_arr
        grasp_dict['transl'] = transl_arr
        grasp_dict['joint_conf'] = joint_arr

        return grasp_dict

    def handle_evaluate_and_filter_grasp_poses(self, palm_poses, joint_confs, thresh):
        bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
        bps_object = bps_object_center[:, 6:]
        grasp_dict = self.to_grasp_dict(palm_poses, joint_confs)
        results = self.FFHNet.filter_grasps(
            bps_object, grasp_dict, thresh=thresh)

        n_grasps_filt = results['rot_matrix'].shape[0]

        n_samples = len(palm_poses)
        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" %
              (float(n_grasps_filt) / float(n_samples)))

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        # prepare response
        #res = EvaluateAndFilterGraspPosesResponse()
        palm_poses = self.build_pose_list(
            results['rot_matrix'], results['transl'])
        joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return palm_poses, joint_confs

    def handle_evaluate_grasp_poses(self, req):
        bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
        bps_object = bps_object_center[:, 6:]
        # Build a dict with all the grasps
        p_success = self.FFHNet.evaluate_grasps(
            bps_object, grasps, return_arr=True)
