from __future__ import print_function
import roslibpy
import open3d as o3d
import os
import time


class TableObjectSegmenter():
    def __init__(self, client):
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

    def custom_draw_object(self, pcd):
        o3d.visualization.draw_geometries([pcd],
                                          zoom=1.,
                                          front=[0., 0.1245, -0.977],
                                          lookat=[0.04479, -0.2049, 1.347],
                                          up=[0.02236, -0.9787, -0.1245])

    # +++++++++++++++++ Part II: Main business logic ++++++++++++++++++++++++
    def handle_table_object_segmentation(self, req, res):
        print("handle_table_object_segmentation received the service call")
        pcd = o3d.io.read_point_cloud("/home/vm/test_cloud.pcd")
        #self.custom_draw_scene(pcd)
        #start = time.time()
        downpcd = pcd.voxel_down_sample(voxel_size=0.005)  # downsample
        #self.custom_draw_scene(downpcd)
        # segment plane
        plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=10)
        object_cloud = downpcd.select_by_index(inliers, invert=True)
        #print(time.time() - start)
        #self.custom_draw_geometry(object_cloud)
        #self.custom_draw_object(object_cloud)
        o3d.io.write_point_cloud("/home/vm/object.pcd", object_cloud)
        print("Object.pcd saved successfully")
        res['success'] = True
        return True


if __name__ == "__main__":
    # client = None
    client = roslibpy.Ros(host='localhost', port=9090)
    # client.is_connected

    tos = TableObjectSegmenter(client)

    client.run_forever()
    client.terminate()