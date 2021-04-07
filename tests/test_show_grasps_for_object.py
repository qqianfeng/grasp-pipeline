import open3d as o3d
import numpy as np
import os
from grasp_pipeline.utils.grasp_data_handler import GraspDataHandlerVae
from grasp_pipeline.utils.utils import hom_matrix_from_pos_quat_list
from grasp_pipeline.utils.object_names_in_datasets import KIT_OBJECTS
# Input needed for this script
outcomes = ['positive']  # which outcomes to visualize
grasp_data_path = '/home/vm/data/ffhnet-data/ffhnet-grasp.h5'  # where is the grasp data file
obj_names = ['kit_' + obj for obj in KIT_OBJECTS]  # which objects to visualize

for obj in obj_names:
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
    path = os.path.join('/home/vm/gazebo-objects/objects_gazebo', dset, obj_name, file_name)
    mesh = o3d.io.read_triangle_mesh(path)
    verts_shape = np.asarray(mesh.vertices).shape
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.zeros(verts_shape) + np.tile(np.array([0.25, 0.25, 0.25]), (verts_shape[0], 1)))

    pc = mesh.sample_points_uniformly(number_of_points=10000)
    pc.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pc.colors).shape) + 0.2)

    # Get all the grasps
    data_handler = GraspDataHandlerVae(file_path=grasp_data_path)

    for outcome in outcomes:
        palm_poses, _, num_succ = data_handler.get_grasps_for_object(obj, outcome=outcome)

        print("Object name: ", obj_name, "Successful ones:", num_succ)

        frames = []
        max_poses = 2000
        if len(palm_poses) > 2000:
            step = int(len(palm_poses) / max_poses)
        else:
            step = 1
        for i in range(0, len(palm_poses), step):
            palm_hom = hom_matrix_from_pos_quat_list(palm_poses[i])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02).transform(palm_hom)
            frames.append(frame)

        #visualize
        orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.03)
        frames.append(mesh)
        #frames.append(orig)
        o3d.visualization.draw_geometries(frames)
