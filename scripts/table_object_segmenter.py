from __future__ import print_function
import roslibpy
import open3d as o3d
import os
import time
import numpy as np

DEBUG = False

#todo: Use published transform to bring pointcloud to different realm, then


class TableObjectSegmenter():
    def __init__(self, client):
        if not DEBUG:
            table_object_segmentation_service = roslibpy.Service(
                client, '/table_object_segmentation_start_server',
                'std_srvs/SetBool')
            table_object_segmentation_service.advertise(
                self.handle_table_object_segmentation)
            self.object_size_publisher = roslibpy.Topic(
                client,
                '/segmented_object_size',
                'std_msgs/Float32MultiArray',
                latch=True)
            self.bounding_box_corner_publisher = roslibpy.Topic(
                client,
                '/segmented_object_bounding_box_corner_points',
                'std_msgs/Float32MultiArray',
                latch=True)
            self.camera_tf_listener = roslibpy.Topic(
                client, '/camera_color_optical_frame_in_world',
                'geometry_msgs/PoseStamped')
            self.camera_tf_listener.subscribe(self.callback_camera_tf)

        print("Service advertised: /table_object_segmentation_start_server")
        print(
            "Publisher for object dimensions and bounding box corner coordinates started"
        )
        self.bounding_box_corner_points = None
        self.colors = np.array(
            [
                [0, 0, 0],  #black,       left/front/up
                [1, 0, 0],  #red          right/front/up
                [0, 1, 0],  #green        left/front/down
                [0, 0, 1],  #blue         left/back/up
                [0.5, 0.5, 0.5],  #grey   right/back/down
                [0, 1, 1],  # light blue   left/back/down
                [1, 1, 0],  # yellow      right/back/up
                [1, 0, 1],
                [0.6, 0.2, 0.4]  #purple/ red-ish
            ]
        )  # this was just to understand the corner numbering logic, point 0 and point 4 in the list are cross diagonal, points 1,2,3 are attached to 0 in right handed sense, same for 5,6,7
        self.camera_color_frame_T = None

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def custom_draw_scene(self, pcd):
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.24,
                                          front=[0., 0.24, -1],
                                          lookat=[-0.013, -1.1, 5.42],
                                          up=[0., -0.97, -0.24])

    def construct_nine_box_objects(self):
        boxes = []
        for i in range(9):
            box = o3d.geometry.TriangleMesh.create_box(width=0.01,
                                                       height=0.01,
                                                       depth=0.01)
            box.translate(self.bounding_box_corner_points[
                i, :]) if i != 8 else box.translate([0, 0, 0])
            box.paint_uniform_color(self.colors[i, :])
            boxes.append(box)
        return boxes

    def custom_draw_object(self,
                           pcd,
                           bounding_box=None,
                           show_normal=False,
                           draw_box=False):
        if bounding_box == None:
            o3d.visualization.draw_geometries([pcd],
                                              zoom=1.,
                                              front=[0., 0.1245, -0.977],
                                              lookat=[0.04479, -0.2049, 1.347],
                                              up=[0.02236, -0.9787, -0.1245],
                                              point_show_normal=show_normal)
        else:
            boxes = self.construct_nine_box_objects()
            o3d.visualization.draw_geometries([
                pcd, bounding_box, boxes[0], boxes[1], boxes[2], boxes[3],
                boxes[4], boxes[5], boxes[6], boxes[7], boxes[8]
            ],
                                              zoom=1.,
                                              front=[0., 0.1245, -0.977],
                                              lookat=[0.04479, -0.2049, 1.347],
                                              up=[0.02236, -0.9787, -0.1245],
                                              point_show_normal=show_normal)

    def pick_points(self, pcd, bounding_box):
        print("")
        print("1) Please pick at least 3 corresp. using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(bounding_box)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    # +++++++++++++++++ Part II: Main business logic ++++++++++++++++++++++++
    def callback_camera_tf(self, msg):
        orientation = msg['pose']['orientation']
        position = msg['pose']['position']
        quat = np.array([
            orientation['x'], orientation['y'], orientation['z'],
            orientation['w']
        ])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(
            quat)
        self.camera_tf_listener.unsubscribe()
        transformation_matrix = np.zeros([4, 4])
        transformation_matrix[3, 3] = 1
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = np.array(
            [position['x'], position['y'], position['z']])
        self.camera_color_frame_T = transformation_matrix

    def handle_table_object_segmentation(self, req, res):
        print("handle_table_object_segmentation received the service call")
        pcd = o3d.io.read_point_cloud("/home/vm/test_cloud.pcd")

        # Transform PCD into world frame/origin
        pcd = pcd.transform(self.camera_color_frame_T)

        #self.custom_draw_scene(pcd)
        #start = time.time()

        # downsample point cloud
        down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # downsample
        #self.custom_draw_scene(down_pcd)

        # segment plane
        plane_model, inliers = down_pcd.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=30)
        object_pcd = down_pcd.select_by_index(inliers, invert=True)
        #print(time.time() - start)
        #self.custom_draw_object(object_pcd)

        # compute bounding box and size
        object_bounding_box = object_pcd.get_axis_aligned_bounding_box()
        object_size = object_bounding_box.get_extent()
        # self.custom_draw_object(object_pcd, object_bounding_box)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2,
                                                              max_nn=50))
        #self.custom_draw_object(object_pcd, object_bounding_box, True)

        # orient normals towards camera
        object_pcd.orient_normals_towards_camera_location()
        #self.custom_draw_object(object_pcd, object_bounding_box, True)

        # get the 8 corner points, from these you can compute the bounding box face center points from which you can get the nearest neighbour from the point cloud
        self.bounding_box_corner_points = np.asarray(
            object_bounding_box.get_box_points())
        print(self.bounding_box_corner_points)

        # Draw object, bounding box and colored corners
        self.custom_draw_object(object_pcd, object_bounding_box, False, True)

        # In the end the object pcd, bounding box corner points, bounding box size information need to be stored to disk
        # Could also be sent over a topic
        o3d.io.write_point_cloud("/home/vm/object.pcd", object_pcd)
        print(
            "Object.pcd saved successfully with normals oriented towards camera"
        )

        # Publish and latch newly computed dimensions and bounding box points
        if not DEBUG:
            self.object_size_publisher.publish(
                roslibpy.Message({'data': np.ndarray.tolist(object_size)}))
            self.bounding_box_corner_publisher.publish(
                roslibpy.Message({
                    'data':
                    np.ndarray.tolist(self.bounding_box_corner_points)
                }))

        res['success'] = True

        return True


if __name__ == "__main__":
    client = None
    client = roslibpy.Ros(host='localhost', port=9090) if not DEBUG else None
    tos = TableObjectSegmenter(client)

    if DEBUG:
        tos.handle_table_object_segmentation(None, dict({"success": True}))
    else:
        client.run_forever()
        print(client.is_connected)

        client.terminate()