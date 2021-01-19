#!/usr/bin/env python
# mass = float(input("Object Mass is : "))
# volume = float(input("Mesh Volume is : "))
# print("Inertia Tensor is :")
mass = 1.1485
volume = 1177043.500000
j = [list(map(float, input("  ").strip().split(" "))) for _ in range(3)]
print(mass)
print(volume)
print(j)

for i in range(3):
    for k in range(3):
        # Change density and convert millimeter to meter
        j[i][k] *= mass/volume*0.000001

print(f"""\n<inertia ixx="{j[0][0]:.16f}" ixy="{j[0][1]:.16f}" ixz="{j[0][2]:.16f}" iyy="{j[1][1]:.16f}" iyz="{j[1][2]:.16f}" izz="{j[2][2]:.16f}" />""")