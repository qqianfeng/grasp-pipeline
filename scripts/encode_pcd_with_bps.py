""" This server only runs under python 3. It reads a segmented pcd from disk and also a BPS. 
When it is called it will compute the BPS encoding of the object. 
"""
import bps_torch.bps as b_torch
import numpy as np
import roslibpy as rlp
import open3d as o3d
import torch


class BPSEncoder():
    def __init__(self, client, bps_path='/home/vm/data/vae-grasp/basis_point_set.npy'):
        self.client = client
        service = rlp.Service(client, '/encode_pcd_with_bps', 'std_srvs/SetBool')
        service.advertise(self.handle_encode_pcd_with_bps)

        self.bps_np = np.load(bps_path)
        self.bps = b_torch.bps_torch(custom_basis=self.bps_np)

        self.pcd_path = '/home/vm/object.pcd'
        self.enc_path = '/home/vm/pcd_enc.npy'

        self.VISUALIZE = False

    def handle_encode_pcd_with_bps(self, req, res):
        obj_pcd = o3d.io.read_point_cloud(self.pcd_path)
        obj_np = np.asarray(obj_pcd.points)

        enc_dict = self.bps.encode(obj_np)
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