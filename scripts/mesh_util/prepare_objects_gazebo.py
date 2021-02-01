import os
import trimesh
import numpy as np
import shutil

base_folder = '/home/vm/object_datasets'

#datasets = ['kit', 'ycb', 'bigbird']
datasets = ['kit']

if __name__ == "__main__":
    # Iterate over all datasets
    for dataset in datasets:
        objects_raw_folder = base_folder + '/objects_raw/' + dataset
        objects_gazebo_folder = base_folder + '/objects_gazebo/' + dataset
        template_folder = base_folder + '/templates/'

        # Set dataset specific variables
        if dataset == 'ycb':
            objects_raw_folder += '/models'

        # Create folder under base_folder/objects_gazebo for dataset
        if not os.path.exists(objects_gazebo_folder):
            os.mkdir(objects_gazebo_folder)

        # Get a list of all objects in objects_raw/dataset
        folder_names = sorted(os.listdir(objects_raw_folder))

        # Get the template files
        config_template_file = os.path.join(template_folder, "model.config")
        model_template_file = os.path.join(template_folder, "template.sdf")
        material_template_file = os.path.join(template_folder, "template.material")
        with open(config_template_file, "r") as f:
            config_template_text = f.read()
        with open(model_template_file, "r") as f:
            model_template_text = f.read()
        with open(material_template_file, "r") as f:
            material_template_text = f.read()

        # Now loop through all the folders under the folder_raw_data/dataset
        for model_name in folder_names:
            print("Creating Gazebo files for {} ...".format(model_name))

            # Create path where gazebo model file should be placed
            model_dest_folder = os.path.join(objects_gazebo_folder, model_name)
            if not os.path.exists(model_dest_folder):
                os.mkdir(model_dest_folder)

            object_folder = os.path.join(objects_raw_folder, model_name)

            # Set some dataset specific names
            if dataset == 'ycb':
                if os.path.exists(os.path.join(objects_raw_folder, model_name, 'google_16k')):
                    visual_mesh_path = os.path.join(objects_raw_folder, model_name, 'google_16k',
                                                    'textured.obj')
                    collision_mesh_path = os.path.join(objects_raw_folder, model_name,
                                                       'google_16k',
                                                       model_name + '_250_collision.stl')
                    texture_mesh_path = os.path.join(objects_raw_folder, model_name, 'google_16k',
                                                     'texture_map.png')
                    inertia_mesh_path = os.path.join(objects_raw_folder, model_name, 'google_16k',
                                                     model_name + '_convex_hull.stl')
                elif os.path.exists(os.path.join(objects_raw_folder, model_name, 'poisson')):
                    visual_mesh_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                                    'textured.obj')
                    collision_mesh_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                                       model_name + '_250_collision.stl')
                    texture_mesh_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                                     'textured.png')
                    inertia_mesh_path = os.path.join(objects_raw_folder, model_name, 'poisson',
                                                     model_name + '_convex_hull.stl')
                elif os.path.exists(os.path.join(objects_raw_folder, model_name, 'tsdf')):
                    visual_mesh_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                                    'textured.obj')
                    collision_mesh_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                                       model_name + '_250_collision.stl')
                    texture_mesh_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                                     'textured.png')
                    inertia_mesh_path = os.path.join(objects_raw_folder, model_name, 'tsdf',
                                                     model_name + '_convex_hull.stl')
                else:
                    raise ValueError

            elif dataset == 'kit':
                visual_mesh_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                                model_name + '_25k_tex.obj')
                collision_mesh_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                                   model_name + '_250_collision.obj')
                texture_mesh_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                                 model_name + '_25k_tex.png')
                inertia_mesh_path = os.path.join(objects_raw_folder, model_name, 'meshes',
                                                 model_name + '_convex_hull.obj')
            elif dataset == 'bigbird':
                visual_mesh_path = os.path.join(objects_raw_folder, model_name, model_name,
                                                'textured_meshes',
                                                'optimized_poisson_texture_mapped_mesh.obj')
                collision_mesh_path = os.path.join(objects_raw_folder, model_name, model_name,
                                                   'meshes', model_name + '_250_collision.obj')
                texture_mesh_path = os.path.join(objects_raw_folder, model_name, model_name,
                                                 'textured_meshes',
                                                 'optimized_poisson_texture_mapped_mesh.png')
                inertia_mesh_path = os.path.join(objects_raw_folder, model_name, model_name,
                                                 'meshes', model_name + '_convex_hull.obj')

            # Copy the material, visual and collision meshes to their new destinations
            shutil.copyfile(texture_mesh_path,
                            os.path.join(model_dest_folder,
                                         texture_mesh_path.split('/')[-1]))
            visual_mesh = trimesh.load(visual_mesh_path)
            if dataset == 'ycb' or dataset == 'bigbird':
                shutil.copyfile(visual_mesh_path,
                                os.path.join(model_dest_folder,
                                             visual_mesh_path.split('/')[-1]))
                shutil.copyfile(
                    collision_mesh_path,
                    os.path.join(model_dest_folder,
                                 collision_mesh_path.split('/')[-1]))
            else:
                collision_mesh = trimesh.load(collision_mesh_path)

                visual_mesh.apply_scale(0.001)
                collision_mesh.apply_scale(0.001)

                visual_mesh.export(os.path.join(model_dest_folder,
                                                visual_mesh_path.split('/')[-1]))
                collision_mesh.export(
                    os.path.join(model_dest_folder,
                                 collision_mesh_path.split('/')[-1]))
            # Load the visual mesh as this is used for computing inertia and mass.
            if visual_mesh.is_watertight:
                pass
            else:
                visual_mesh = trimesh.load(inertia_mesh_path)
                assert visual_mesh.is_watertight
                if dataset == 'kit':
                    visual_mesh.apply_scale(0.001)

            # Set the density to 100kg/m3 (1/10th of water density )
            visual_mesh.density = 100.
            # Mass and moment of inertia
            mass_text = str(visual_mesh.mass)
            print("Object mass is: " + mass_text)
            tf = visual_mesh.principal_inertia_transform
            inertia = trimesh.inertia.transform_inertia(tf, visual_mesh.moment_inertia)
            #print("Inertia is: " + str(inertia))
            # Center of mass
            com_vec = visual_mesh.center_mass.tolist()
            eul = trimesh.transformations.euler_from_matrix(np.linalg.inv(tf), axes="sxyz")
            com_vec.extend(list(eul))
            com_text = str(com_vec)
            com_text = com_text.replace("[", "")
            com_text = com_text.replace("]", "")
            com_text = com_text.replace(",", "")

            # Set config text
            config_text = config_template_text.replace('$MODEL_NAME', model_name)
            with open(os.path.join(model_dest_folder, "model.config"), "w") as f:
                f.write(config_text)

            # Copy and modify the model file template
            model_text = model_template_text.replace('$MODEL_NAME', model_name)
            model_text = model_text.replace('$DATASET', dataset)
            model_text = model_text.replace('$COLLISION_MESH_NAME',
                                            collision_mesh_path.split('/')[-1])
            model_text = model_text.replace('$VISUAL_MESH_NAME', visual_mesh_path.split('/')[-1])
            model_text = model_text.replace('$MATERIAL_NAME', model_name + '.material')
            model_text = model_text.replace("$MASS", mass_text)
            model_text = model_text.replace("$COM_POSE", com_text)
            model_text = model_text.replace("$IXX", str(inertia[0][0]))
            model_text = model_text.replace("$IYY", str(inertia[1][1]))
            model_text = model_text.replace("$IZZ", str(inertia[2][2]))
            model_text = model_text.replace("$IXY", str(inertia[0][1]))
            model_text = model_text.replace("$IXZ", str(inertia[0][2]))
            model_text = model_text.replace("$IYZ", str(inertia[1][2]))
            with open(os.path.join(model_dest_folder, model_name + ".sdf"), "w") as f:
                f.write(model_text)

            # Change material template
            material_text = material_template_text.replace("$MODEL_NAME", model_name)
            material_text = material_text.replace("$TEXTURE_FILE",
                                                  texture_mesh_path.split('/')[-1])
            with open(os.path.join(model_dest_folder, model_name + ".material"), "w") as f:
                f.write(material_text)
