import open3d as o3d
from time import sleep
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

def draw_with_time_out(pcd_in, time_out_in_seconds=5, num_frames=100):
    angle = (2 * np.pi / num_frames)
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(np.array([angle, angle, angle]))
    pcd = deepcopy(pcd_in)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for i in range(num_frames):
        pcd.rotate(rot_mat)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        sleep(time_out_in_seconds / float(num_frames))

    vis.destroy_window()

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('./object.pcd')
    draw_with_time_out(pcd)