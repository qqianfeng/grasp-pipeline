import open3d as o3d
import numpy as np
import os
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandlerVae
from grasp_pipeline.utils.utils import hom_matrix_from_pos_quat_list
# read mesh and convert to pc
data_folder = '/home/ffh/Documents/grasp_data/2021-03-08/grasp_data/recording_sessions/recording_session_0001'
grasp_data_path = '/home/ffh/Documents/grasp_data/2021-03-08/grasp_data_vae.h5'

for obj in os.listdir(data_folder):
    if obj == 'bigbird_aunt_jemima_original_syrup':
        continue
    obj_split = obj.split('_')
    dset = obj_split[0]
    obj_name = '_'.join(obj_split[1:])
    if dset == 'kit':
        file_name = obj_name + '_25k_tex.obj'
    elif dset == 'bigbird':
        file_name = 'optimized_tsdf_texture_mapped_mesh.obj'
    elif dset == 'ycb':
        file_name = ''
    else:
        raise Exception('Unknown dataset name.')
    path = os.path.join('/home/ffh/gazebo-objects/objects_gazebo', dset, obj_name, file_name)
    mesh = o3d.io.read_triangle_mesh(path)

    pc = mesh.sample_points_uniformly(number_of_points=2000)
    pc.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pc.colors).shape) + 0.2)

    # Get all the grasps
    data_handler = GraspDataHandlerVae(file_path=grasp_data_path)

    for outcome in ['all', 'positive', 'negative']:
        palm_poses, _, num_succ = data_handler.get_grasps_for_object(obj, outcome=outcome)

        print("Successful ones:", num_succ)

        frames = []
        max_poses = 2000
        if len(palm_poses) > 2000:
            step = int(len(palm_poses) / max_poses)
        else:
            step = 1
        for i in range(0, len(palm_poses), step):
            palm_hom = hom_matrix_from_pos_quat_list(palm_poses[i])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01).transform(palm_hom)
            frames.append(frame)

        #visualize
        orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
        frames.append(pc)
        frames.append(orig)
        o3d.visualization.draw_geometries(frames)
