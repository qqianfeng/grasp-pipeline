import open3d as o3d
import os


def show_pcd(pcd):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    o3d.visualization.draw_geometries([pcd, origin])


if __name__ == '__main__':
    base_path = '/home/vm/data/vae-grasp/point_clouds'
    for obj in os.listdir(base_path):
        obj_path = os.path.join(base_path, obj)
        for pcd in os.listdir(obj_path):
            pcd_path = os.path.join(obj_path, pcd)
            show_pcd(o3d.io.read_point_cloud(pcd_path))