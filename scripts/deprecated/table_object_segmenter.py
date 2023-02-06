from __future__ import print_function
import roslibpy
import open3d as o3d
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
import copy

DEBUG = True


class TableObjectSegmenter():
    def __init__(self, client, scene_pcd_path=None, object_pcd_path=None):
        if not DEBUG:
            self.client = client
            table_object_segmentation_service = roslibpy.Service(
                client, '/table_object_segmentation_start_server', 'std_srvs/SetBool')
            table_object_segmentation_service.advertise(self.handle_table_object_segmentation)
            self.object_size_publisher = roslibpy.Topic(client,
                                                        '/segmented_object_size',
                                                        'std_msgs/Float32MultiArray',
                                                        latch=True)
            self.bounding_box_corner_publisher = roslibpy.Topic(
                client,
                '/segmented_object_bounding_box_corner_points',
                'std_msgs/Float32MultiArray',
                latch=True)
            # for the camera position in 3D space w.r.t world
            self.camera_tf_listener = roslibpy.Topic(
                client, '/camera_color_optical_frame_in_world', 'geometry_msgs/PoseStamped')
            self.camera_tf_listener.subscribe(self.callback_camera_tf)
            # Where to save the object point cloud and where to load it from
            self.scene_pcd_path_listener = roslibpy.Topic(client, '/scene_pcd_path',
                                                          'std_msgs/String')
            self.scene_pcd_path_listener.subscribe(self.callback_scene_pcd_path)
            self.object_pcd_path_listener = roslibpy.Topic(client, '/object_pcd_path',
                                                           'std_msgs/String')
            self.object_pcd_path_listener.subscribe(self.callback_object_pcd_path)

        print("Service advertised: /table_object_segmentation_start_server")
        print("Publisher for object dimensions and bounding box corner coordinates started")
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
        self.world_R_cam = None
        self.world_t_cam = None

        self.object_pcd_save_path = object_pcd_path
        self.pcd_read_path = scene_pcd_path

    # +++++++++++++++++ Part I: Helper functions ++++++++++++++++++++++++
    def custom_draw_scene(self, pcd):
        o3d.visualization.draw_geometries([pcd],
                                          front=[0.236, -0.97, 0.0435],
                                          lookat=[-0.244, 3.89, 0.0539],
                                          up=[0., 0.045, 0.998],
                                          zoom=0.2399)

    def construct_nine_box_objects(self):
        boxes = []
        for i in range(9):
            box = o3d.geometry.TriangleMesh.create_box(width=0.01, height=0.01, depth=0.01)
            box.translate(self.bounding_box_corner_points[i, :]) if i != 8 else box.translate(
                [0, 0, 0])
            box.paint_uniform_color(self.colors[i, :])
            boxes.append(box)
        return boxes

    def custom_draw_object(self, pcd, bounding_box=None, show_normal=False, draw_box=False):
        if bounding_box == None:
            o3d.visualization.draw_geometries([pcd], point_show_normal=show_normal)
        else:
            boxes = self.construct_nine_box_objects()
            o3d.visualization.draw_geometries([
                pcd, bounding_box, boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5],
                boxes[6], boxes[7], boxes[8]
            ],
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
        print("Entered callback camera tf")
        orientation = msg['pose']['orientation']
        position = msg['pose']['position']
        quat = np.array([orientation['x'], orientation['y'], orientation['z'], orientation['w']])
        rotation_matrix = Rotation(quat=quat).as_matrix()
        self.camera_tf_listener.unsubscribe()
        self.world_R_cam = rotation_matrix
        self.world_t_cam = np.array([position['x'], position['y'], position['z']])
        print("Unsubscribed camera_tf_listener")

    def callback_object_pcd_path(self, msg):
        print("Entered callback object pcd path")
        self.object_pcd_save_path = msg['data']
        print("I heard: %s" % msg['data'])
        self.object_pcd_path_listener.unsubscribe()
        print("Unsubscribed object_pcd_path_listener")

    def callback_scene_pcd_path(self, msg):
        print("Entered callback scene pcd path")
        self.pcd_read_path = msg['data']
        print("I heard: %s" % msg['data'])
        self.scene_pcd_path_listener.unsubscribe()
        print("Unsubscribedscene_pcd_path_listener ")

    def handle_table_object_segmentation(self, req, res):
        while (self.object_pcd_save_path is None) or (self.pcd_read_path is None):
            print(
                "I haven't received the pointpcd paths from the corrsponding topic. Stuck in a while loop until they are received."
            )
            time.sleep(2)

        print("handle_table_object_segmentation received the service call")
        pcd = o3d.io.read_pcd(self.pcd_read_path)

        if self.world_t_cam is None:
            self.world_t_cam = [0.8275, -0.996, 0.36]
        if DEBUG:
            self.custom_draw_scene(pcd)
        #start = time.time()

        # downsample point pcd
        down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # downsample
        if DEBUG:
            self.custom_draw_scene(down_pcd)

        # segment plane
        _, inliers = down_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=30)
        object_pcd = down_pcd.select_by_index(inliers, invert=True)
        #print(time.time() - start)
        if DEBUG:
            self.custom_draw_object(object_pcd)

        # compute bounding box and size
        object_bounding_box = object_pcd.get_axis_aligned_bounding_box()
        object_size = object_bounding_box.get_extent()
        # get the 8 corner points, from these you can compute the bounding box face center points from which you can get the nearest neighbour from the point pcd
        self.bounding_box_corner_points = np.asarray(object_bounding_box.get_box_points())
        print(self.bounding_box_corner_points)
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box)

        # compute normals of object
        object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=50))
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # orient normals towards camera
        object_pcd.orient_normals_towards_camera_location(self.world_t_cam)
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, True)

        # Draw object, bounding box and colored corners
        if DEBUG:
            self.custom_draw_object(object_pcd, object_bounding_box, False, True)

        # In the end the object pcd, bounding box corner points, bounding box size information need to be stored to disk
        # Could also be sent over a topic
        if os.path.exists(self.object_pcd_save_path):
            os.remove(self.object_pcd_save_path)
        o3d.io.write_pcd("/home/vm/object.pcd", object_pcd)
        print("Object.pcd saved successfully with normals oriented towards camera")

        # Publish and latch newly computed dimensions and bounding box points
        if not DEBUG:
            print("I will publish the size now:")
            self.object_size_publisher.publish(
                roslibpy.Message({'data': np.ndarray.tolist(object_size)}))
            print("I will publish the corner points now:")
            # print(self.bounding_box_corner_points)
            # self.bounding_box_corner_publisher.publish(
            #     roslibpy.Message({
            #         'data':
            #         np.ndarray.tolist(self.bounding_box_corner_points)
            #     }))
            self.bounding_box_corner_publisher.publish(
                roslibpy.Message({'data': np.ndarray.tolist(object_size)}))
            print("I published them")

        res['success'] = True

        return True


if __name__ == "__main__":
    client = None
    if not DEBUG:
        client = roslibpy.Ros(host='localhost', port=9090)
        tos = TableObjectSegmenter(client)

    if DEBUG:
        tos = TableObjectSegmenter(client,
                                   scene_pcd_path='/home/ffh/test_pcd.pcd',
                                   object_pcd_path='/home/ffh/object.pcd')

        tos.handle_table_object_segmentation(None, dict({"success": True}))
    else:
        client.run_forever()
        client.terminate()