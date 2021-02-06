#!/usr/bin/env python
import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
from grasp_planner.srv import *

SHOW_PCD = False


class VoxelGenerator():
    def __init__(self):
        rospy.init_node("generate_voxel_from_pcd_node")

    def show_pcd(self, pcd):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    def generate_voxel_grid_from_pcd(self, pcd, voxel_size, voxel_dim):
        # Voxel_grid: consists of a python list of arrays. Initialize with empty list
        voxel_grid = []
        # Initialize voxel grid of size voxel_dim with all zeros
        voxel_grid_bi = np.zeros((voxel_dim[0], voxel_dim[1], voxel_dim[2]), dtype=np.int8)
        # Create voxel_min_location variable
        voxel_min_loc = np.zeros(3, dtype=np.float64)
        # Compute voxel_min_loc from voxel_dim and voxel_size
        voxel_min_loc[0] = (-1.) * voxel_dim[0] / 2 * voxel_size[0]
        voxel_min_loc[1] = (-1.) * voxel_dim[1] / 2 * voxel_size[1]
        voxel_min_loc[2] = (-1.) * voxel_dim[2] / 2 * voxel_size[2]
        # Make sure that pcd contains no NANs
        pcd.remove_none_finite_points()
        if SHOW_PCD:
            self.show_pcd(pcd)
        # Iterate over all points in the point cloud
        for iter_point in np.asarray(pcd.points):
            voxel_loc_x = int((iter_point[0] - voxel_min_loc[0]) / voxel_size[0])
            voxel_loc_y = int((iter_point[1] - voxel_min_loc[1]) / voxel_size[1])
            voxel_loc_z = int((iter_point[2] - voxel_min_loc[2]) / voxel_size[2])
            if (voxel_loc_x >= 0 and voxel_loc_x < voxel_dim[0] \
                and voxel_loc_y >= 0 and voxel_loc_y < voxel_dim[1] \
                    and voxel_loc_z >= 0 and voxel_loc_z < voxel_dim[2]):
                if (voxel_grid_bi[voxel_loc_x][voxel_loc_y][voxel_loc_z] != 1):
                    voxel_grid_bi[voxel_loc_x][voxel_loc_y][voxel_loc_z] = 1
                    voxel_grid.append(np.array([voxel_loc_x, voxel_loc_y, voxel_loc_z]))
        return voxel_grid

    def handle_generate_voxel_from_pcd(self, req):
        object_pcd = o3d.io.read_point_cloud(req.object_pcd_path)
        # Display point cloud for debugging purposes
        if show_PCD:
            self.show_pcd(object_pcd)
        # Transform point cloud to object centric frame by translating the object point cloud by it's center, which is also the origin of the object centric frame. The object centric frame is aligned with the world frame which is why no rotation is applied
        object_pcd.translate((-1.) * object_pcd.get_center())
        if SHOW_PCD:
            self.show_pcd(object_pcd)
        # Generate voxel grid representation from pcd
        voxel_grid = self.generate_voxel_grid_from_pcd(object_pcd, req.voxel_size, req.voxel_dim)
        voxels_num = len(voxel_grid)
        voxel_grid_1d = (-1) * np.ones([3 * voxels_num], dtype=np.int16)
        for i in range(voxels_num):
            #Translate the voxel grid from partial frame to full view frame
            voxel_grid_1d[3 * i] = voxel_grid[i][0] + req.voxel_translation_dim[0]
            voxel_grid_1d[3 * i + 1] = voxel_grid[i][1] + req.voxel_translation_dim[1]
            voxel_grid_1d[3 * i + 2] = voxel_grid[i][2] + req.voxel_translation_dim[2]
        print("Number of filled voxels:")
        print(voxels_num)
        res = GenVoxelFromPcdResponse()
        res.voxel_grid = voxel_grid_1d
        return res

    def create_generate_voxel_from_pcd_server(self):
        rospy.Service('generate_voxel_from_pcd', GenVoxelFromPcd,
                      self.handle_generate_voxel_from_pcd)
        rospy.loginfo('Service generate_voxel_from_pcd:')
        rospy.loginfo('Ready to generate Voxels from point clouds.')


if __name__ == '__main__':
    vg = VoxelGenerator()
    vg.create_generate_voxel_from_pcd_server()
    rospy.spin()