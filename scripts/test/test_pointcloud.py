import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/vm/test_cloud.pcd")

box = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
box.paint_uniform_color([1, 0, 0])

box_cam = o3d.geometry.TriangleMesh.create_box(width=0.01,
                                               height=0.01,
                                               depth=0.01)
box_cam.paint_uniform_color([0, 1, 0])
box_cam.translate([0.8275, -0.996, 0.361])

o3d.visualization.draw_geometries([pcd, box, box_cam])