import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/vm/scene.pcd")

#o3d.visualization.draw_geometries([pcd])

intrinsics = o3d.camera.PinholeCameraIntrinsic(width=1280,
                                               height=720,
                                               fx=695.9951171875,
                                               fy=695.9951171875,
                                               cx=640.0,
                                               cy=360.0)
depth_image = o3d.io.read_image("/home/vm/depth.png")
# pcd_from_depth = o3d.geometry.PointCloud().create_from_depth_image(depth=depth_image,
#                                                                    intrinsic=intrinsics)
pcd_from_depth = o3d.geometry.PointCloud().create_from_depth_image(
    depth=depth_image, intrinsic=o3d.camera.PinholeCameraIntrinsic())
o3d.visualization.draw_geometries([pcd_from_depth])
