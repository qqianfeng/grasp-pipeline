import open3d as o3d
import numpy as np


def visualize_normals(pcd):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(origin)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("/home/vm/hand_ws/src/grasp-pipeline/save.json")
    vis.run()


origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

pcd_scene = o3d.io.read_point_cloud("/home/vm/scene.pcd")
o3d.visualization.draw_geometries([pcd_scene, origin])

# no panda
points = np.asarray(pcd_scene.points)
colors = np.asarray(pcd_scene.colors)
mask = points[:, 0] > 0.1

pcd_scene = o3d.geometry.PointCloud()
pcd_scene.points = o3d.utility.Vector3dVector(points[mask])
o3d.visualization.draw_geometries([pcd_scene, origin])

# segment plane
_, inliers = pcd_scene.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
object_pcd = pcd_scene.select_down_sample(inliers, invert=True)

# Downsample regularly
k = np.asarray(object_pcd.points).shape[0] // 256
#object_pcd_down_reg = object_pcd.voxel_down_sample(0.015)
#o3d.visualization.draw_geometries([object_pcd_down_reg, origin])
#print(np.asarray(object_pcd_down_reg.points).shape)

# Estimate normals
object_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=60))
visualize_normals(object_pcd)
object_pcd_down_reg = object_pcd.voxel_down_sample(0.015)

# object_pcd_down_reg.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
visualize_normals(object_pcd_down_reg)

# o3d.visualization.draw_geometries([pcd])
# pcd1 = pcd.voxel_down_sample(0.015)
# print(np.asarray(pcd.points).shape)
# o3d.visualization.draw_geometries([pcd1])
# print(np.asarray(pcd1.points).shape)

# k = np.asarray(pcd.points).shape[0] // 256

# pcd2 = pcd.uniform_down_sample(k)
# print(np.asarray(pcd2.points).shape)
# o3d.visualization.draw_geometries([pcd2])

# depth_image = o3d.io.read_image("/home/vm/depth.png")

# intrinsics = o3d.camera.PinholeCameraIntrinsic(width=1280,
#                                                height=720,
#                                                fx=695.9951171875,
#                                                fy=695.9951171875,
#                                                cx=640.0,
#                                                cy=360.0)

# pcd_from_depth = o3d.geometry.PointCloud().create_from_depth_image(depth=depth_image,
#                                                                    intrinsic=intrinsics)

# from scipy import misc

# a = misc.imread("/home/vm/depth.png")

# image = np.interp(a, (a.min(), a.max()), (458, 1500))

# world_T_camera = np.array([[1.00000000e+00, -1.44707624e-12, -2.18525541e-13, 8.42500000e-01],
#                            [-2.18874845e-13, -2.95520207e-01, 9.55336489e-01, -9.96305997e-01],
#                            [-1.44702345e-12, -9.55336489e-01, -2.95520207e-01, 3.61941706e-01],
#                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# def deproject_to_pcd_image(depth_image, intrinsic_matrix):
#     """
#     deproject depth image into a point cloud image
#     :param depth_image: numpy array shape(height, width, 1) or shape(height, width)
#     :return: point cloud image: numpy array shape(width, height, 3)
#     """
#     row_indices = np.arange(depth_image.shape[0])
#     col_indices = np.arange(depth_image.shape[1])
#     pixel_grid = np.meshgrid(col_indices, row_indices)
#     pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
#     pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
#     depth_arr = np.tile(depth_image.flatten(), [3, 1])
#     # deproject
#     points_3d = depth_arr * np.linalg.inv(intrinsic_matrix).dot(pixels_homog)
#     pcd_image = points_3d.T.reshape(depth_image.shape[0], depth_image.shape[1], 3)
#     return points_3d

# int_vec = np.array([695.9951171875, 0.0, 640.0, 0.0, 695.9951171875, 360.0, 0.0, 0.0, 1.0])
# intrinsics_matrix = np.reshape(int_vec, (3, 3))

# xyz = deproject_to_pcd_image(image, intrinsics_matrix)
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# xyz = xyz * 0.001
# cloud = o3d.geometry.PointCloud()
# cloud.points = o3d.utility.Vector3dVector(xyz.T)
# o3d.visualization.draw_geometries([cloud, origin])
# cloud.transform(world_T_camera)
# o3d.visualization.draw_geometries([cloud, origin])

# o3d.visualization.draw_geometries([cloud, origin, pcd])
