# This code only runs with python 3
import pymeshlab
import os

dataset = 'bigbird'

objects_raw_folder = '/home/vm/object_datasets/objects_raw'

if __name__ == "__main__":
    objects_raw_folder = os.path.join(objects_raw_folder, dataset)
    ms = pymeshlab.MeshSet()
    folder_names = sorted(os.listdir(objects_raw_folder))
    if dataset == 'kit':
        for i in range(0, 300):
            model_name = folder_names[i]
            print('processing model: ' + model_name + '. Iteration: ' + str(i))
            mesh_load_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                          model_name + '_25k.obj')
            mesh_save_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                          model_name + '_convex_hull.obj')
            ms.load_new_mesh(mesh_load_path)

            ms.apply_filter('convex_hull')
            ms.save_current_mesh(mesh_save_path)

    elif dataset == 'bigbird':
        #for model_name in folder_names:
        for i in range(0, 300):
            model_name = folder_names[i]
            print('processing model: ' + model_name + '. Iteration: ' + str(i))
            mesh_load_path = os.path.join(objects_raw_folder, model_name, model_name, 'meshes',
                                          'tsdf.ply')
            mesh_save_path = os.path.join(objects_raw_folder, model_name, model_name, 'meshes',
                                          model_name + '_convex_hull.obj')

            ms.load_new_mesh(mesh_load_path)

            ms.apply_filter('convex_hull')
            ms.save_current_mesh(mesh_save_path)

    elif dataset == 'ycb':
        objects_raw_folder = os.path.join(objects_raw_folder, 'models')
        folder_names = sorted(os.listdir(objects_raw_folder))
        #for model_name in folder_names:
        for i in range(0, 300):
            model_name = folder_names[i]
            print('processing model: ' + model_name + '. Iteration: ' + str(i))
            if os.path.exists(os.path.join(objects_raw_folder, model_name, 'google_16k')):
                mesh_load_path = os.path.join(objects_raw_folder, model_name, 'google_16k',
                                              'nontextured.stl')
                mesh_save_path = os.path.join(objects_raw_folder, model_name, 'google_16k',
                                              model_name + '_convex_hull.stl')

            elif os.path.exists(os.path.join(objects_raw_folder, model_name, 'poisson')):
                mesh_load_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                              'nontextured.stl')
                mesh_save_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                              model_name + '_convex_hull.stl')
            elif os.path.exists(os.path.join(objects_raw_folder, model_name, 'tsdf')):
                mesh_load_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                              'nontextured.stl')
                mesh_save_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                              model_name + '_convex_hull.stl')

            ms.load_new_mesh(mesh_load_path)

            ms.apply_filter('convex_hull')
            ms.save_current_mesh(mesh_save_path)
