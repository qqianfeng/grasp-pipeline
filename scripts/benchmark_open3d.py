import open3d as o3d
import numpy as np
import time

a = np.array([[0, 0, 0.5], [1, 1, 1]])
b = np.array([[0, 0, 0.5], [0, 0, 0.5]])

distances = np.sum(np.square(a - b), axis=1)
min_ix = np.argmin(distances)
min_vec = a[min_ix, :]

obj_cloud = o3d.io.read_point_cloud('/home/vm/object.pcd')
obj_points = np.asarray(obj_cloud.points)
ref = o3d.geometry.PointCloud()
p = np.transpose(np.array([[0], [0], [0]]))
ref.points = o3d.utility.Vector3dVector(p)

# first with open3d
start = time.time()
dist = obj_cloud.compute_point_cloud_distance(ref)
diff = time.time() - start
nearest_neighbour = p + np.asarray(dist)
print(diff)