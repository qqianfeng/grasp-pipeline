#!/usr/bin/env python
import numpy as np
import torch
import os
import sys
import copy
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as T

# Add FFHNet to the path
sys.path.append(os.environ['FFHNET_PATH'])

from FFHNet.models.ffhnet import FFHNet
from FFHNet.config.eval_config import EvalConfig
from FFHNet.utils import visualization

class GraspInference():
    def __init__(self):
        cfg = EvalConfig().parse()
        self.VISUALIZE = int(os.environ.get('VISUALIZE'))
        self.pcd_path = os.environ['OBJECT_PCD_PATH']

        self.FFHNet = FFHNet(cfg)
        ffhnet_path = os.environ['FFHNET_PATH']
        self.FFHNet.load_ffhgenerator(epoch=10, load_path=os.path.join(ffhnet_path, 'models/ffhgenerator'))
        self.FFHNet.load_ffhevaluator(
            epoch=30,
            load_path=os.path.join(ffhnet_path, 'models/ffhevaluator'))

    def handle_infer_grasp_poses(self, n_poses):
        # Make sure the file is not currently locked for writing
        loaded = False
        while not loaded:
            try:
                 bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
                 loaded = True
            except Exception as e:
                print("\033[93m" + "[Warning] ", {e}, "\nRetrying... \033[0m")

        center_transf = bps_object_center[:, :6]
        bps_object = bps_object_center[:, 6:]
        n_samples = n_poses
        results = self.FFHNet.generate_grasps(
            bps_object, n_samples=n_samples, return_arr=False)

        # Shift grasps back to original coordinate frame
        r_axis_angle = center_transf[:, 3:6]
        r = torch.tensor(R.from_rotvec(r_axis_angle).as_matrix()).cuda()
        transf = T.Rotate(r)
        results['transl'] = transf.transform_points(results['transl'])
        results['transl'] += torch.from_numpy(center_transf[:, :3]).cuda()

        """if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)"""

        return results

    def handle_evaluate_and_filter_grasp_poses(self, grasp_dict, thresh=0.5):
        # Make sure the file is not currently locked for writing
        loaded = False
        while not loaded:
            try:
                 bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
                 loaded = True
            except Exception as e:
                print("\033[93m" + "[Warning] ", {e}, "\nRetrying... \033[0m")

        center_transf = bps_object_center[:, :6]
        bps_object = bps_object_center[:, 6:]

        n_samples = len(grasp_dict['transl'])

        # Shift grasps back to center
        grasp_dict['transl'] -= torch.from_numpy(center_transf[:, :3]).cuda()
        print(grasp_dict['transl'])
        print(grasp_dict['transl'].shape)
        r_axis_angle = center_transf[:, 3:6]
        r_numpy = R.from_rotvec(r_axis_angle).as_matrix()
        r = torch.tensor(r_numpy).cuda()
        transf = T.Rotate(r)
        print(transf.get_matrix().detach().cpu().numpy())
        grasp_dict['transl'] = transf.inverse().transform_points(grasp_dict['transl'])
        print(grasp_dict['transl'])
        results = self.FFHNet.filter_grasps(
            bps_object, grasp_dict, thresh=thresh)

        n_grasps_filt = results['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" %
              (float(n_grasps_filt) / float(n_samples)))

        # Shift grasps back to original coordinate frame
        results['transl'] = results['transl'] @ r_numpy[0].T
        results['transl'] += center_transf[:, :3]

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        return results

    def handle_evaluate_grasp_poses(self, grasp_dict):
        # Make sure the file is not currently locked for writing
        loaded = False
        while not loaded:
            try:
                 bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
                 loaded = True
            except Exception as e:
                print("\033[93m" + "[Warning] ", {e}, "\nRetrying... \033[0m")

        center_transf = bps_object_center[:, :6]
        bps_object = bps_object_center[:, 6:]

        # Shift grasps back to center
        grasp_dict['transl'] -= torch.from_numpy(center_transf[:, :3]).cuda()
        r_axis_angle = center_transf[:, 3:6]
        r = torch.tensor(R.from_rotvec(r_axis_angle).as_matrix()).cuda()
        transf = T.Rotate(r)
        grasp_dict['transl'] = transf.inverse().transform_points(grasp_dict['transl'])

        # Get score for all grasps
        p_success = self.FFHNet.evaluate_grasps(
            bps_object, grasp_dict, return_arr=True)

        # Shift grasps back to original coordinate frame
        grasp_dict['transl'] = transf.transform_points(grasp_dict['transl'])
        grasp_dict['transl'] += torch.from_numpy(center_transf[:, :3]).cuda()

        return p_success
