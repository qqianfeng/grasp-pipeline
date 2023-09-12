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

from ffhflow.configs import get_config
from ffhflow.datasets import FFHDataModule
from ffhflow.ffhflow_pos_enc import FFHFlowPosEnc

class InferFFHFlow():
    def __init__(self, client):
        self.client = client
        service = rlp.Service(client, '/infer_ffhflow', 'std_srvs/SetBool')
        service.advertise(self.handle_infer_ffhflow)
        model_cfg = rospy.get_param('model_cfg')
        ckpt_path = rospy.get_param('ckpt_path')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.VISUALIZE = False

        # Set up cfg
        cfg = get_config(model_cfg)

        # configure dataloader
        ffh_datamodule = FFHDataModule(cfg)

        self.model = FFHFlowPosEnc.load_from_checkpoint(ckpt_path, cfg=cfg)
        self.model.eval()

    def handle_infer_ffhflow(self, req, res):
        bps = req.bps
                # Go over the images in the dataset.
        with torch.no_grad():
            out = self.model.sample(bps, num_samples=100)
            grasp = self.model.filter_grasps(out)
            self.model.show_grasps(bps, out)
            # self.model.show_gt_grasps(batch['pcd_path'][0], batch, i)

            print(grasp[0])
        return grasp[0]


if __name__ == '__main__':
    client = rlp.Ros(host='localhost', port=9091)
    bpsenc = InferFFHFlow(client)
    client.run_forever()
    client.terminate()
