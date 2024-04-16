import numpy as np
import urdfpy
import open3d as o3d
from copy import deepcopy

sim_cam_pose_origin = [0.5,-0.75,0.35,0,0.3,1.57]
sim_cam_pose_30dge = [0.5,-0.75,0.39,0,0.36,1.57]
# sim_cam_pose_30dge = [0.5,-0.75,0.55,0,0.52,1.57]

def xyzrpy_to_mat(xyzrpy):
    mat_rot = urdfpy.rpy_to_matrix(xyzrpy[3:])
    mat = np.eye(4)
    mat[:3,:3] = mat_rot
    mat[:3,-1] = np.asarray(xyzrpy[:3])
    # print(mat)
    return mat

sim_cam_pose_origin_mat = xyzrpy_to_mat(sim_cam_pose_origin)
sim_cam_pose_30dge = xyzrpy_to_mat(sim_cam_pose_30dge)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

obj_pcd = o3d.io.read_point_cloud('/home/yb/object.pcd')
pcd_center = np.load('/home/yb/obj_center.npy')
obj_pcd_new_cam = obj_pcd.translate(pcd_center)
# o3d.visualization.draw_geometries([origin,obj_pcd])

origin_cam = deepcopy(origin).transform(sim_cam_pose_origin_mat)
new_cam = deepcopy(origin).transform(sim_cam_pose_30dge)
# o3d.visualization.draw_geometries([origin,origin_cam,new_cam])

# origin_cam = deepcopy(origin).transform(world_T_cam)
# new_cam = deepcopy(origin).transform(world_T_new_cam)
# o3d.visualization.draw_geometries([origin,origin_cam,new_cam])

robot_T_origin = sim_cam_pose_origin_mat
robot_T_new = sim_cam_pose_30dge
origin_T_new = np.matmul(np.linalg.inv(robot_T_origin),robot_T_new)
print('origin_T_new',repr(origin_T_new))
print('new_T_origin',repr(np.linalg.inv(origin_T_new)))

# new_T_origin_frame = deepcopy(origin).transform(origin_T_new)
# o3d.visualization.draw_geometries([origin,new_T_origin_frame])

obj_pcd_origin_cam = deepcopy(obj_pcd_new_cam).transform(origin_T_new)
# o3d.visualization.draw_geometries([origin,obj_pcd_origin_cam])
# o3d.visualization.draw_geometries([origin,obj_pcd_origin_cam])
np.save('/home/yb/obj_center_origin_cam.npy', obj_pcd_origin_cam.get_center())
obj_pcd_origin_cam.translate(-1*obj_pcd_origin_cam.get_center())

# o3d.visualization.draw_geometries([origin,obj_pcd_origin_cam])

o3d.io.write_point_cloud('/home/yb/object.pcd', obj_pcd_origin_cam)
