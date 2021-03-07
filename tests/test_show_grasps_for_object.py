import open3d as o3d
import numpy as np
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandlerVae
from grasp_pipeline.utils.utils import hom_matrix_from_pos_quat_list
# read mesh and convert to pc
mesh = o3d.io.read_triangle_mesh(
    '/home/vm/gazebo-objects/objects_gazebo/kit/NutCandy/NutCandy_25k_tex.obj')

pc = mesh.sample_points_uniformly(number_of_points=2000)
pc.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pc.colors).shape) + 0.2)

# Get all the grasps
obj = 'kit_NutCandy'
path = '/home/vm/Music/vae-grasp/grasp_data_vae.h5'
data_handler = GraspDataHandlerVae(file_path=path)

palm_poses, _, num_succ = data_handler.get_all_positives_for_object(obj)

print("Successful ones:", num_succ)

frames = []
for i in range(len(palm_poses)):
    palm_hom = hom_matrix_from_pos_quat_list(palm_poses[i])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01).transform(palm_hom)
    frames.append(frame)

#visualize
orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
frames.append(pc)
frames.append(orig)
o3d.visualization.draw_geometries(frames)
