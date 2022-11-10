#!/usr/bin/env python
# mass = float(input("Object Mass is : "))
# volume = float(input("Mesh Volume is : "))
# print("Inertia Tensor is :")
# mass = 1.1485
# volume = 1177043.500000
# j = [list(map(float, input("  ").strip().split(" "))) for _ in range(3)]
# print(mass)
# print(volume)
# print(j)

# for i in range(3):
#     for k in range(3):
#         # Change density and convert millimeter to meter
#         j[i][k] *= mass/volume*0.000001

# print(f"""\n<inertia ixx="{j[0][0]:.16f}" ixy="{j[0][1]:.16f}" ixz="{j[0][2]:.16f}" iyy="{j[1][1]:.16f}" iyz="{j[1][2]:.16f}" izz="{j[2][2]:.16f}" />""")

import trimesh
mesh_path = '/home/ffh/ffh_ws/src/hithand-ros/hithand_description/meshes/collision/finger/hit-hand-2-finger-phadist.dae'
mesh = trimesh.load(mesh_path)

# Set the density to 100kg/m3 (1/10th of water density )
mesh.density = 100.
# Mass and moment of inertia
mass_text = str(mesh.mass)
print(mesh.moment_inertia)
tf = mesh.principal_inertia_transform
inertia = trimesh.inertia.transform_inertia(tf, mesh.moment_inertia)
print(inertia)
print(mesh.moment_inertia)