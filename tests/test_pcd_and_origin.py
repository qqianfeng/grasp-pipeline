import open3d as o3d

obj = o3d.io.read_point_cloud('/home/vm/object.pcd')
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

o3d.visualization.draw_geometries([origin, obj])

obj = o3d.io.read_point_cloud(
    '/home/vm/data/ffhnet-data/point_clouds/kit_CoughDropsBerries/kit_CoughDropsBerries_pcd017.pcd'
)
o3d.visualization.draw_geometries([origin, obj])
