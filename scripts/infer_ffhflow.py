""" This server only runs under python 3. It reads a segmented pcd from disk and also a BPS.
When it is called it will compute the BPS encoding of the object.
"""
import numpy as np
import roslibpy as rlp
import rospy
import open3d as o3d
import torch
from time import time
import torch
import sys
import os
import transforms3d
import pickle

# Add FFHNet to the path
# sys.path.append(rospy.get_param('ffhflow_path'))
sys.path.append('/home/yb/workspace/FFHFlow-dpf')
sys.path.append('/home/yb/workspace/normalizing-flows')

use_ffhflow_lvm = True
add_joint_conf = True

from ffhflow.configs import get_config
# TODO: Change the import package
# from ffhflow.ffhflow_pos_enc_with_transl import FFHFlowPosEncWithTransl
from ffhflow.ffhflow_cnf import FFHFlowCNF
if use_ffhflow_lvm:
    from ffhflow.ffhflow_lvm import FFHFlowLVM

# from ffhflow.normflows_ffhflow_pos_enc_with_transl import NormflowsFFHFlowPosEncWithTransl
# if use_ffhflow_lvm:
#     from ffhflow.normflows_ffhflow_pos_enc_with_transl import NormflowsFFHFlowPosEncWithTransl_LVM

from geometry_msgs.msg import PoseStamped
from grasp_pipeline.srv import *
from grasp_pipeline.utils import utils
from sensor_msgs.msg import JointState

class InferFFHFlow():
    def __init__(self, client):
        self.client = client
        service = rlp.Service(client, '/infer_ffhflow', 'std_srvs/SetBool')
        service.advertise(self.handle_infer_ffhflow)
        model_cfg = os.path.join(rospy.get_param('ffhflow_path'),rospy.get_param('model_cfg'))
        ckpt_path = os.path.join(rospy.get_param('ffhflow_path'),rospy.get_param('ckpt_path'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.VISUALIZE = False

        # Set up cfg
        cfg = get_config(model_cfg)
        # TODO: Use the correct imported module
        if use_ffhflow_lvm:
            self.model = FFHFlowLVM.load_from_checkpoint(ckpt_path, cfg=cfg)
        else:
            self.model = FFHFlowCNF.load_from_checkpoint(ckpt_path, cfg=cfg)
        self.model.eval()

    def check_quat_validity(self, quat):
        q_norm_error = abs(quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3] + quat[0] * quat[0] - 1.0)
        if q_norm_error > 0.1:
            print('found invalid quat in infer_ffhflow::check_quat_validity')
            return False
        else:
            return True

    def build_pose_list(self, rot_matrix, transl, frame_id='object_centroid_vae'):
        assert rot_matrix.shape[1:] == (
            3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_matrix.shape[0] == transl.shape[
            0], "Batch dimension of rot and trans not equal."

        poses = []

        for i in range(rot_matrix.shape[0]):
            quat = transforms3d.quaternions.mat2quat(rot_matrix[i, :, :])
            t = transl[i, :]

            pose_st = PoseStamped()
            pose_st.header.frame_id = frame_id
            pose_st.pose.position.x = t[0]
            pose_st.pose.position.y = t[1]
            pose_st.pose.position.z = t[2]
            pose_st.pose.orientation.x = quat[1]
            pose_st.pose.orientation.y = quat[2]
            pose_st.pose.orientation.z = quat[3]
            pose_st.pose.orientation.w = quat[0]

            poses.append(pose_st)

        return poses

    def build_joint_conf_list(self, joint_conf=False):
        joint_confs = []
        offset = 0.2
        joint_offset = [0,offset,offset,0,offset,offset,0,offset,offset,0,offset,offset,0,offset,offset]

        for i in range(joint_conf.shape[0]):
            jc = JointState()
            jc.position = joint_conf[i, :]
            if add_joint_conf:
                jc.position += np.asarray(joint_offset)

            # jc.position = 0.2*np.zeros(20)
            joint_confs.append(jc)
        print(joint_confs)
        return joint_confs

    def build_pose_and_joint_conf_list(self, rot_matrix, transl, frame_id='object_centroid_vae', joint_conf=False):
        """Add quaternion check validity

        Args:
            rot_matrix (_type_): _description_
            transl (_type_): _description_
            frame_id (str, optional): _description_. Defaults to 'object_centroid_vae'.
            joint_conf (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert rot_matrix.shape[1:] == (
            3, 3), "Assumes palm rotation is 3*3 matrix."
        assert rot_matrix.shape[0] == transl.shape[
            0], "Batch dimension of rot and trans not equal."

        poses = []
        joint_confs = []

        for i in range(rot_matrix.shape[0]):
            quat = transforms3d.quaternions.mat2quat(rot_matrix[i, :, :])
            if not self.check_quat_validity(quat):
                continue
            t = transl[i, :]

            pose_st = PoseStamped()
            pose_st.header.frame_id = frame_id
            pose_st.pose.position.x = t[0]
            pose_st.pose.position.y = t[1]
            pose_st.pose.position.z = t[2]
            pose_st.pose.orientation.x = quat[1]
            pose_st.pose.orientation.y = quat[2]
            pose_st.pose.orientation.z = quat[3]
            pose_st.pose.orientation.w = quat[0]

            poses.append(pose_st)

            jc = JointState()
            jc.position = joint_conf[i, :]
            joint_confs.append(jc)

        return poses, joint_confs

    def handle_infer_ffhflow(self, req, res):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        bps_tensor = torch.from_numpy(bps_object).to(self.device)
        n_samples = 100
        # Go over the images in the dataset.
        with torch.no_grad():
            # For normflow ffhflow-lvm model
            if use_ffhflow_lvm:
                grasps = self.model.sample_in_experiment(bps_tensor, num_samples=n_samples, posterior_score="neg_kl")
            else:
                grasps = self.model.sample_in_experiment(bps_tensor, num_samples=n_samples)
            # # self.model.show_grasps(pcd_path=rospy.get_param('object_pcd_path'), samples=grasps, i=-1)
            grasps = self.model.sort_and_filter_grasps(grasps, perc=0.99,return_arr=True)
            # print('after filter', grasps['log_prob'])
            # if no sort_and_filter
            # grasps = {k: v.cpu().detach().numpy() for k, v in grasps.items()}
            # i = -1 then no images will be saved in show_grasps
            # self.model.show_gt_grasps(batch['pcd_path'][0], batch, i)
        palm_poses, joint_confs = self.build_pose_and_joint_conf_list(grasps['rot_matrix'], grasps['transl'],joint_conf=grasps['joint_conf'])
        print('after build_pose_and_joint_conf_list, we have palm_poses of:', len(palm_poses))

        # palm_poses = self.build_pose_list(grasps['rot_matrix'], grasps['transl'])
        # joint_confs = self.build_joint_conf_list(grasps['joint_conf'])
        # joint_confs = self.build_joint_conf_list()

        with open(rospy.get_param('grasp_save_path'), 'wb') as fp:
            pickle.dump([palm_poses, joint_confs, grasps['log_prob']], fp, protocol=2)

        return True

    # def to_grasp_dict(self, palm_poses, joint_confs, probs=False):
    #     """Take the palm_poses and joint_confs in ros-format and convert them to a dict with two 3D arrays in order to
    #     use as input for FFHEvaluator

    #     Args:
    #         palm_poses (List of PoseStamped): List of Palm poses in object frame
    #         joint_confs (List of JointState): List of joint states belonging to grasp

    #     Returns:
    #         grasp_dict (dict): k1 is
    #     """
    #     # prepare
    #     batch_n = len(palm_poses)
    #     joint_arr = np.zeros((batch_n, 15))
    #     rot_matrix_arr = np.zeros((batch_n, 3, 3))
    #     transl_arr = np.zeros((batch_n, 3))
    #     prob_arr = np.zeros((batch_n, 1))
    #     # convert
    #     for i, (palm_pose, joint_conf, prob) in enumerate(zip(palm_poses, joint_confs, probs)):
    #         q = palm_pose.pose.orientation
    #         t = palm_pose.pose.position

    #         rot_matrix_arr[i, :, :] = tft.quaternion_matrix(
    #             [q.x, q.y, q.z, q.w])[:3, :3]
    #         transl_arr[i, :] = [t.x, t.y, t.z]
    #         joint_arr[i, :] = np.array(joint_conf.position)
    #         prob_arr[i] = prob

    #     # Build grasp dict
    #     grasp_dict = {}
    #     grasp_dict['rot_matrix'] = rot_matrix_arr
    #     grasp_dict['transl'] = transl_arr
    #     grasp_dict['joint_conf'] = joint_arr
    #     grasp_dict['log_prob'] = prob_arr
    #     return grasp_dict

    # def handle_evaluate_and_filter_grasp_poses(self, req):
    #     bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
    #     grasp_dict = self.to_grasp_dict(req.palm_poses, req.joint_confs, req.probs)
    #     print('before filter')
    #     results = self.FFHNet.filter_grasps(
    #         bps_object, grasp_dict, thresh=req.thresh, filter_grasps=True)
    #     print('after filter')

    #     n_grasps_filt = results['rot_matrix'].shape[0]

    #     palm_poses = req.palm_poses
    #     n_samples = len(palm_poses)
    #     print("n_grasps after filtering: %d" % n_grasps_filt)
    #     print("This means %.2f of grasps pass the filtering" %
    #           (n_grasps_filt / n_samples))

    #     if self.VISUALIZE:
    #         visualization.show_generated_grasp_distribution(
    #             self.pcd_path, results)

    #     # prepare response
    #     res = EvaluateAndFilterGraspPosesResponse()
    #     res.palm_poses = self.build_pose_list(
    #         results['rot_matrix'], results['transl'])
    #     res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

    #     return res

    def handle_evaluate_grasp_poses(self, req):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        # Build a dict with all the grasps
        p_success = self.FFHNet.evaluate_grasps(
            bps_object, grasps, return_arr=True)


if __name__ == '__main__':
    client = rlp.Ros(host='localhost', port=9090)
    bpsenc = InferFFHFlow(client)
    client.run_forever()
    client.terminate()
