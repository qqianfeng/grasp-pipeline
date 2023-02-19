import pymeshlab as ml
import os

in_folder = "/home/vm/meshes_inertia/"
out_folder = in_folder + "output/"
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
for filename in os.listdir(in_folder):
    ms = ml.MeshSet()
    in_full_path = in_folder + filename
    # Load mesh
    ms.load_new_mesh(in_full_path)
    # Apply scale filter
    ms.apply_filter('transform_scale_normalize', axisx=1000, axisy=1000, axisz=1000)
    # Compute inertia
    out_dict = ms.apply_filter('compute_geometric_measures', )

    print(out_dict)
