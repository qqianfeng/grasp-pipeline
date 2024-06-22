import open3d as o3d
import os
import numpy as np

path = '/data/hdd1/qf/sim_exp_ffhflow/grasp_data/recording_sessions/recording_session_0201/bigbird_3m_high_tack_spray_adhesive/grasp_0001/pre_grasp'
object_pcd = o3d.io.read_point_cloud(os.path.join(path,'object.pcd'))
scene_pcd = o3d.io.read_point_cloud(os.path.join(path,'scene.pcd'))
segmented_obj_pcd = o3d.io.read_point_cloud(os.path.join(path,'segmented_obj.pcd'))
obstacles_pcd = o3d.io.read_point_cloud(os.path.join(path,'obstacles.pcd'))
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

print(np.asarray(object_pcd.points).shape)
print(np.asarray(scene_pcd.points).shape)
print(np.asarray(segmented_obj_pcd.points).shape)
print(np.asarray(obstacles_pcd.points).shape)

o3d.visualization.draw_geometries([object_pcd,origin])
o3d.visualization.draw_geometries([scene_pcd,origin])
o3d.visualization.draw_geometries([segmented_obj_pcd,origin])
o3d.visualization.draw_geometries([obstacles_pcd,origin])