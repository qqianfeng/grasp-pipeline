import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time

# 3D processing imports
import pyrealsense2 as rs
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy

# Basis-Point-Set-encoder imports
import bps_torch.bps as b_torch

# DDS imports
import ar_dds as dds


class Segmenter():
    def __init__(self):
        self.use_diana7 = int(os.environ["USE_DIANA7"])
        self.simulation = int(os.environ["SIMULATION"])

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 0.000001

        # Transformations
        # Flange rotation
        self.flange_rot = np.identity(3)
        if not self.use_diana7:
            self.flange_rot[0, 0] = -1
            self.flange_rot[2, 2] = -1

        # Rotation and translation from camera frame to flange frame
        # self.cam_R_ee = np.zeros((3, 3))
        # self.cam_R_ee[0,1] = -1
        # self.cam_R_ee[1,0] = -1
        # self.cam_R_ee[2,2] = -1
        # self.cam_R_ee = cam_R_ee @ R.from_euler('xyz', [0, 0, math.pi/8]).as_matrix()

        # from hand-in-eye-calibration (camera frame)
        self.cam_R_ee = np.array(
            [[-0.3423919549517217, -0.939541935997535, 0.005357208820077931],
             [-0.9395565782222526, 0.34237955165001827, -0.0031110980737669783],
             [0.001088808353391471, -0.006098615739338168, -0.9999808105070985]])
        self.cam_R_ee = self.cam_R_ee @ self.flange_rot

        # Rotation from base frame to camera frame which FFHNet was trained with
        self.ee_home_R_cam_ffh = R.from_euler('zyx', [0.0, 0.0, 1.87079632679]).as_matrix()

    # DDS subscriber functions
    def interrupt_signal_handler(self, _signal_number, _frame):
        """SIGINT/SIGTSTP handler for gracefully stopping an application."""
        print("Caught interrupt signal. Stop application!")
        global shutdown_requested
        shutdown_requested = True

    # Read robot telemetry
    def poll_telemetry_data(self):

        received_data = self.listener.take(True)
        for sample in received_data:
            if not sample.info.valid:
                continue

            T = sample.data[self.dds_telemetry_topic_name]

            pos = T["translation"]
            robot_pos = self.flange_rot @ np.array([pos['x'], pos['y'], pos['z']])

            quat = T["rotation"]
            quat_arr = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
            robot_rot = self.flange_rot @ R.from_quat(quat_arr).as_matrix()

        return robot_pos, robot_rot

    def initialize_camera(self):
        # Enable data stream of Realsense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = depth_profile.get_intrinsics()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Init BPS-Encoder
        bps_path = os.path.join(os.environ['FFHNET_PATH'], 'models/basis_point_set.npy')
        bps_np = np.load(bps_path)
        self.bps = b_torch.bps_torch(custom_basis=bps_np)

        # Init path to save pcd
        self.pcd_path = os.environ['OBJECT_PCD_PATH']
        self.enc_path = os.environ['OBJECT_PCD_ENC_PATH']

    def initialize_dds(self):
        # Initialize DDS Domain Participant
        self.participant = dds.DomainParticipant(domain_id=0)

        if not self.simulation:
            # Create subscriber using the participant
            self.dds_telemetry_topic_name = "T_flange2base"
            if self.use_diana7:
                self.listener = self.participant.create_subscriber_listener(
                    "ar::interfaces::dds::robot::generic::telemetry_v1",
                    "diana7.generic_telemetry", None)
                print("[Info] Using Diana 7.")
            else:
                self.listener = self.participant.create_subscriber_listener(
                    "ar::interfaces::dds::robot::generic::telemetry_v1", "diana.telemetry_v1",
                    None)
                print("[Info] Using Diana X1.")
            print("[Info] Telemetry subscriber is ready. Waiting for data ...")

            # Test connection
            sample = False
            while not sample:
                received_data = self.listener.take(False)
                samples = iter(received_data)
                sample = next(samples, False)

        # Create Domain Participant and a publisher to publish data
        self.publisher = self.participant.create_publisher("ar::dds::pcd_enc::Msg", 'pcd_enc_msg')

    def get_realsense_data(self):
        # Wait until new frame is received
        frames = self.pipeline.poll_for_frames()
        while not frames:
            frames = self.pipeline.poll_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Update depth intrinsics
        self.depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        # Convert images to np.array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def segment_object(self):

        # Get camera data
        color_image, depth_image = self.get_realsense_data()

        # Create mask
        mask = np.zeros((color_image.shape[0], color_image.shape[1]), np.uint8)

        # GrabCut arrays
        bgdModel = np.zeros((1, 65), np.float64)
        fgbModel = np.zeros((1, 65), np.float64)

        # Select ROI
        reselect = True
        while reselect:

            # Get camera data
            color_image, _ = self.get_realsense_data()

            cv2.namedWindow("Seg", cv2.WND_PROP_FULLSCREEN)
            try:
                init_rect = cv2.selectROI('Seg', color_image, False, False)
            except:
                init_rect = [0]
            if not any(init_rect):
                print("No area selected. Press 'c' to abort or anything else to reselect")
                if cv2.waitKey(0) == ord('c'):
                    exit()
            else:
                reselect = False

        # Close window
        cv2.destroyWindow("Seg")

        # Run GrabCut
        cv2.grabCut(color_image, mask, init_rect, bgdModel, fgbModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        masked_image = color_image * mask2[:, :, np.newaxis]
        plt.imshow(masked_image), plt.colorbar, plt.show()

        # Get camera data
        color_image, depth_image = self.get_realsense_data()

        # Set area outside of the segmentation mask to zero
        depth_image *= mask2

        # Remove data with large depth offset from segmented object's median
        median = np.median(depth_image[depth_image > self.eps])
        depth_image = np.where(abs(depth_image - median) < 100, depth_image, 0)

        # Load depth image as o3d.Image
        depth_image_o3d = o3d.geometry.Image(depth_image)

        # Generate point cloud from depth image
        depth_intrinsics = self.depth_intrinsics
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.fx,
            depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d,
                                                             pinhole_camera_intrinsic)

        # Remove outliers
        self.draw_pc(pc)

        # Center point cloud
        pc_center = pc.get_center()
        pc.translate(-pc_center)

        # Get robot's rotation
        if not self.simulation:
            _, robot_rot = self.poll_telemetry_data()
        else:
            # For simulation point cloud has static rotation (stationary camera)
            robot_rot = self.flange_rot

        # Rotate point cloud to coincide with camera frame from FFHNet training
        ee_R_cam_ffh = self.ee_home_R_cam_ffh @ robot_rot @ self.cam_R_ee.T
        cart_pose_angle_axis = R.from_matrix(ee_R_cam_ffh.T).as_rotvec()
        pc_ffh = copy.deepcopy(pc).rotate(ee_R_cam_ffh)

        # Encode point cloud with BPS
        pc_tensor = torch.from_numpy(np.asarray(pc_ffh.points))
        pc_tensor.to(self.device)
        enc_dict = self.bps.encode(pc_tensor)

        # Convert encoded point cloud to numpy
        enc_np = enc_dict['dists'].cpu().detach().numpy()

        # if all dists are greater, pcd is too far from bps
        if enc_np.min() > 0.1:
            print("\033[93m" + "[Warning] The pcd might not be in centered in origin!" + "\033[0m")

        # Publish BPS encoded point cloud and centering data
        self.publisher.message["pose"] = np.concatenate((pc_center, cart_pose_angle_axis),
                                                        dtype=np.float32)
        self.publisher.message["pcd_enc"] = enc_np[0]
        self.publisher.message["active"] = True
        self.publisher.publish()
        print("[Info] BPS encoded point cloud published.")

        # Invert transformation of point cloud (for visualization)
        pc.translate(pc_center)

        # Prepare np-array to save BPS encoding
        self.pcd_enc_with_center = np.insert(enc_np, 0,
                                             np.concatenate((pc_center, cart_pose_angle_axis)))

        # Make point cloud accessable for visualization
        self.pc = pc

    def draw_pc(self, pc=None):
        # Visualize result
        if pc == None:
            pc = self.pc
        o3d.visualization.draw_geometries([pc])

    def save(self, rec_path):
        # Save point cloud
        timestamp = str(int(time.time()))
        file_path = os.path.join(rec_path, "o3d_point_cloud_baseline_" + timestamp + ".pcd")
        o3d.io.write_point_cloud(file_path, self.pc)
        print("[Info] Open3d point cloud saved to", file_path)

        # Save BPS encoding and centering data
        file_path = os.path.join(rec_path,
                                 "bps_encoding_with_center_baseline_" + timestamp + ".npy")
        np.save(file_path, self.pcd_enc_with_center[np.newaxis, :])
        print("[Info] BPS encoded point cloud saved to", file_path)


def main():
    # Load environment variables
    rec_path = os.environ["DATA_RECORDING_PATH"]

    # Instantiate segmenter
    seg = Segmenter()

    # Initialize camera
    seg.initialize_camera()

    # Initialze DDS subscriber and publisher
    seg.initialize_dds()

    # Run segmentation
    seg.segment_object()

    # Save data
    seg.save(rec_path)

    # Show result
    #seg.draw_pc()


if __name__ == "__main__":
    main()
