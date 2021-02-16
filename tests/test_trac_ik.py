#!/usr/bin/env python

# from trac_ik_python.trac_ik import IK

# ik_solver = IK("world", "panda_link8")

# seed_state = [0.0] * ik_solver.number_of_joints


# ik = ik_solver.get_ik(
#     seed_state,
#     0.45,
#     0.1,
#     0.3,  # X, Y, Z
#     0.0,
#     0.0,
#     0.0,
#     1.0)  # QX, QY, QZ, QW
# print(ik)
class Hi():
    def __init__(self, value):
        self.value = 1


l = [Hi(6), Hi(3)]

print(l.index(min(l, key=lambda x: x.value), key=lambda x: x.value))