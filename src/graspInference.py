#!/usr/bin/env python
from distutils.log import error
import numpy as np
import torch
import os
import sys
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as T

# Add FFHNet to the path
sys.path.append(os.environ['FFHNET_PATH'])

from FFHNet.models.ffhnet import FFHNet
from FFHNet.config.eval_config import EvalConfig
from FFHNet.utils import visualization

# DDS imports
import ar_dds as dds

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

        # Initialize array for DDS data
        self.center_transf = np.zeros((1, 6))

        # Initialize DDS Domain Participant
        participant = dds.DomainParticipant(domain_id=0)

        # Create subscriber using the participant
        self.listener = participant.create_subscriber_listener("ar::dds::pcd_enc::Msg",
                                                          'pcd_enc_msg', self.listener_callback)
        print("Subscriber is ready. Waiting for data ...")

    def interrupt_signal_handler(self, _signal_number, _frame):
        """SIGINT/SIGTSTP handler for gracefully stopping an application."""
        print("Caught interrupt signal. Stop application!")
        self.shutdown_requested
        self.shutdown_requested = True

    def listener_callback(self, msg):
        """
        retrieve the message field of the data received
        """
        data = np.array(msg["message"])
        """
        Assign data
        """
        self.center_transf = data[:6].reshape(1, -1)
        self.bps_object = data[6:].reshape(1, -1)

    def handle_infer_grasp_poses(self, n_poses):
        if self.VISUALIZE:
            bps_object_center = np.load(os.environ['OBJECT_PCD_ENC_PATH'])
            self.center_transf = bps_object_center[:, :6]
            self.bps_object = bps_object_center[:, 6:]
        else:
            # Wait for data to be published
            while not any(self.center_transf[0]):
                pass

        n_samples = n_poses
        results = self.FFHNet.generate_grasps(
            self.bps_object, n_samples=n_samples, return_arr=False)

        # Shift grasps back to original coordinate frame
        r_axis_angle = self.center_transf[:, 3:6]
        r = torch.tensor(R.from_rotvec(r_axis_angle).as_matrix()).cuda()
        transf = T.Rotate(r)
        results['transl'] = transf.transform_points(results['transl'])
        results['transl'] += torch.from_numpy(self.center_transf[:, :3]).cuda()

        """if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)"""

        return results, self.center_transf

    def handle_evaluate_and_filter_grasp_poses(self, grasp_dict, thresh=0.5):
        
        n_samples = len(grasp_dict['transl'])

        # Shift grasps back to center
        grasp_dict['transl'] -= torch.from_numpy(self.center_transf[:, :3]).cuda()
        r_axis_angle = self.center_transf[:, 3:6]
        r_numpy = R.from_rotvec(r_axis_angle).as_matrix()
        r = torch.tensor(r_numpy).cuda()
        transf = T.Rotate(r)
        grasp_dict['transl'] = transf.inverse().transform_points(grasp_dict['transl'])
        results = self.FFHNet.filter_grasps(
            self.bps_object, grasp_dict, thresh=thresh)

        n_grasps_filt = results['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" %
              (float(n_grasps_filt) / float(n_samples)))

        # Shift grasps back to original coordinate frame
        results['transl'] = results['transl'] @ r_numpy[0].T
        results['transl'] += self.center_transf[:, :3]

        if self.VISUALIZE:
            visualization.show_generated_grasp_distribution(
                self.pcd_path, results)

        return results

    def handle_evaluate_grasp_poses(self, grasp_dict):
        # Wait for data to be published
        while not any(self.center_transf[0]):
            pass

        # Shift grasps back to center
        grasp_dict['transl'] -= torch.from_numpy(self.center_transf[:, :3]).cuda()
        r_axis_angle = self.center_transf[:, 3:6]
        r = torch.tensor(R.from_rotvec(r_axis_angle).as_matrix()).cuda()
        transf = T.Rotate(r)
        grasp_dict['transl'] = transf.inverse().transform_points(grasp_dict['transl'])

        # Get score for all grasps
        p_success = self.FFHNet.evaluate_grasps(
            self.bps_object, grasp_dict, return_arr=True)

        # Shift grasps back to original coordinate frame
        grasp_dict['transl'] = transf.transform_points(grasp_dict['transl'])
        grasp_dict['transl'] += torch.from_numpy(self.center_transf[:, :3]).cuda()

        return p_success
