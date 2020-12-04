from __future__ import print_function
import roslibpy
import open3d as o3d


class TableObjectSegmenter():
    def __init__(self, client):
        table_object_segmentation_service = roslibpy.Service(
            client, '/table_object_segmentation_start', 'std_srvs/SetBool')
        table_object_segmentation_service.advertise(
            self.table_object_segmentation_callback)

    def table_object_segmentation_callback(self, req):
        # read in the necessary data e.g. open3d pointcloud
        pcd = o3d.io.read_point_cloud("/home/vm/pointcloud.pcd")


if __name__ == "__main__":
    print("It worked")

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    client.is_connected

    tos = TableObjectSegmenter(client)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Received interrupt")
        client.terminate()