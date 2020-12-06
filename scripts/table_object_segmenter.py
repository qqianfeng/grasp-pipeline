from __future__ import print_function
import roslibpy
import open3d as o3d
import os
import time

DEBUG = True


class TableObjectSegmenter():
    def __init__(self, client):
        if not DEBUG:
            table_object_segmentation_service = roslibpy.Service(
                client, '/table_object_segmentation_start_server',
                'std_srvs/SetBool')
            table_object_segmentation_service.advertise(
                self.handle_table_object_segmentation)
        print("Service advertised: /table_object_segmentation_start_server")

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def custom_draw_scene(self, pcd):
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.24,
                                          front=[0., 0.24, -1],
                                          lookat=[-0.013, -1.1, 5.42],
                                          up=[0., -0.97, -0.24])

    def custom_draw_object(self, pcd, bounding_box=None, show_normal=False):
        if bounding_box == None:
            o3d.visualization.draw_geometries([pcd],
                                              zoom=1.,
                                              front=[0., 0.1245, -0.977],
                                              lookat=[0.04479, -0.2049, 1.347],
                                              up=[0.02236, -0.9787, -0.1245],
                                              point_show_normal=show_normal)
        else:
            bounding_box.color = (1, 0, 0)
            o3d.visualization.draw_geometries([pcd, bounding_box],
                                              zoom=1.,
                                              front=[0., 0.1245, -0.977],
                                              lookat=[0.04479, -0.2049, 1.347],
                                              up=[0.02236, -0.9787, -0.1245],
                                              point_show_normal=show_normal)

    # +++++++++++++++++ Part II: Main business logic ++++++++++++++++++++++++
    def handle_table_object_segmentation(self, req, res):
        print("handle_table_object_segmentation received the service call")
        pcd = o3d.io.read_point_cloud("/home/vm/test_cloud.pcd")
        #self.custom_draw_scene(pcd)
        #start = time.time()

        # downsample point cloud
        down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # downsample
        #self.custom_draw_scene(down_pcd)

        # segment plane
        plane_model, inliers = down_pcd.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=10)
        object_pcd = down_pcd.select_by_index(inliers, invert=True)
        #print(time.time() - start)
        #self.custom_draw_object(object_pcd)

        # compute bounding box
        object_bounding_box = object_pcd.get_axis_aligned_bounding_box()
        size = object_bounding_box.get_extent()
        self.custom_draw_object(object_pcd, object_bounding_box)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,
                                                              max_nn=50))
        self.custom_draw_object(object_pcd, object_bounding_box, True)

        # orient normals towards camera
        object_pcd.orient_normals_towards_camera_location()
        self.custom_draw_object(object_pcd, object_bounding_box, True)

        o3d.io.write_point_cloud("/home/vm/object.pcd", object_pcd)
        print("Object.pcd saved successfully")
        res['success'] = True

        return True


if __name__ == "__main__":
    # client = None
    client = roslibpy.Ros(host='localhost', port=9090) if not DEBUG else None
    # client.is_connected

    tos = TableObjectSegmenter(client)

    if not DEBUG:
        client.run_forever()
        client.terminate()
    else:
        tos.handle_table_object_segmentation(None, dict({"success": True}))