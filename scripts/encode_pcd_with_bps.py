""" This server only runs under python 3. It reads a segmented pcd from disk and also a BPS.
When it is called it will compute the BPS encoding of the object.
"""
# for FFHNet ICRA2022 work
import bps_torch.bps as b_torch
import numpy as np
import roslibpy as rlp
import rospy
import open3d as o3d
import torch
import os

bps_path = '/data/hdd1/qf/hithand_data/models/basis_point_set.npy'
# bps_path = '/data/hdd1/qf/hithand_data/ffhnet-data/basis_point_set.npy'
class BPSEncoder():
    def __init__(self, client, bps_path=bps_path):
        self.client = client
        service = rlp.Service(client, '/encode_pcd_with_bps', 'std_srvs/SetBool')
        service.advertise(self.handle_encode_pcd_with_bps)

        self.bps_np = np.load(bps_path)
        self.bps = b_torch.bps_torch(custom_basis=self.bps_np)

        self.pcd_path = rospy.get_param('object_pcd_path')
        self.enc_path = rospy.get_param('object_pcd_enc_path')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.VISUALIZE = False

    def handle_encode_pcd_with_bps(self, req, res):
        obj_pcd = o3d.io.read_point_cloud(self.pcd_path)
        obj_tensor = torch.from_numpy(np.asarray(obj_pcd.points))
        obj_tensor.to(self.device)

        enc_dict = self.bps.encode(obj_tensor)
        enc_np = enc_dict['dists'].cpu().detach().numpy()
        np.save(self.enc_path, enc_np)

        # if all dists are greater, pcd is too far from bps
        assert enc_np.min() < 0.1, 'The pcd might not be in centered in origin!'

        if self.VISUALIZE:
            bps_pcd = o3d.geometry.PointCloud()
            bps_pcd.points = o3d.utility.Vector3dVector(self.bps_np)
            bps_pcd.colors = o3d.utility.Vector3dVector(0.3 * np.ones(self.bps_np.shape))
            o3d.visualization.draw_geometries([obj_pcd, bps_pcd])

        return True


if __name__ == '__main__':
    client = rlp.Ros(host='localhost', port=9090)
    bpsenc = BPSEncoder(client)
    client.run_forever()
    client.terminate()