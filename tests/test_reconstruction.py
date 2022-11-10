import open3d as o3d
import time
""" https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
"""
pcd = o3d.io.read_point_cloud(
    "/home/ffh/grasp_data/recording_sessions/recording_session_0001/grasp_000005/pre_grasp/band_aid_clear_strips.pcd"
)

#pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
start = time.time()
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                         depth=8,
                                                                         width=0,
                                                                         scale=1.1,
                                                                         linear_fit=False)[0]
print("Took: " + str(time.time() - start))

bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)

o3d.visualization.draw_geometries([p_mesh_crop, pcd])

o3d.io.write_triangle_mesh("/home/ffh/triangle_mesh.ply", p_mesh_crop)
