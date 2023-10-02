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
sys.path.append(rospy.get_param('ffhflow_path'))
from ffhflow.configs import get_config
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc
from ffhflow.ffhflow_pos_enc_with_transl import FFHFlowPosEncWithTransl
from ffhflow.ffhflow_pos_enc_neg_grasp import FFHFlowPosEncNegGrasp
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

        self.model = FFHFlowPosEncWithTransl.load_from_checkpoint(ckpt_path, cfg=cfg)
        self.model.eval()

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
        for i in range(joint_conf.shape[0]):
            jc = JointState()
            jc.position = joint_conf[i, :]
            joint_confs.append(jc)
        return joint_confs

    def handle_infer_ffhflow(self, req, res):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        bps_tensor = torch.from_numpy(bps_object).to(self.device)
        n_samples = 100
        # Go over the images in the dataset.
        with torch.no_grad():
            grasps = self.model.sample(bps_tensor, num_samples=n_samples)
            # self.model.show_grasps(pcd_path=rospy.get_param('object_pcd_path'), samples=grasps, i=-1)
            grasps = self.model.sort_and_filter_grasps(grasps, perc=0.99,return_arr=True)
            # i = -1 then no images will be saved in show_grasps
            # self.model.show_gt_grasps(batch['pcd_path'][0], batch, i)

        palm_poses = self.build_pose_list(grasps['rot_matrix'], grasps['transl'])
        joint_confs = self.build_joint_conf_list(grasps['joint_conf'])
        # joint_confs = self.build_joint_conf_list()

        with open(rospy.get_param('grasp_save_path'), 'wb') as fp:
            pickle.dump([palm_poses, joint_confs], fp, protocol=2)

        return True

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

            rot_matrix_arr[i, :, :] = tft.quaternion_matrix(
                [q.x, q.y, q.z, q.w])[:3, :3]
            transl_arr[i, :] = [t.x, t.y, t.z]
            joint_arr[i, :] = np.array(joint_conf.position)

        # Build grasp dict
        grasp_dict = {}
        grasp_dict['rot_matrix'] = rot_matrix_arr
        grasp_dict['transl'] = transl_arr
        grasp_dict['joint_conf'] = joint_arr

        return grasp_dict

    def handle_evaluate_and_filter_grasp_poses(self, req):
        bps_object = np.load(rospy.get_param('object_pcd_enc_path'))
        grasp_dict = self.to_grasp_dict(req.palm_poses, req.joint_confs)
        results = self.FFHNet.filter_grasps(
            bps_object, grasp_dict, thresh=req.thresh)

        n_grasps_filt = results['rot_matrix'].shape[0]

        palm_poses = req.palm_poses
        n_samples = len(palm_poses)
        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" %
              (n_grasps_filt / n_samples))

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        # prepare response
        res = EvaluateAndFilterGraspPosesResponse()
        res.palm_poses = self.build_pose_list(
            results['rot_matrix'], results['transl'])
        res.joint_confs = self.build_joint_conf_list(results['joint_conf'])

        return res

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
