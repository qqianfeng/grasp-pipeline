import cv2
import numpy as np
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
import open3d as o3d
import rospy
import os
import math

N_POSES = 400
FILTER_THRESH = 0.5  # set to -1 if no filtering desired, default 0.9
FILTER_NUM_GRASPS = 5
NUM_TRIALS_PER_OBJ = 20
NUM_OBSTACLE_OBJECTS = 3

pcd_center = None

def main():
    global pcd_center
    data_recording_path = rospy.get_param('data_recording_path')
    gc = GraspClient(grasp_data_recording_path=data_recording_path, is_rec_sess=True, is_eval_sess=True)
    color_img, depth_img = read_image('/home/ffh/ffh_grasp_visualization/1/BathDetergent_color.jpg', '/home/ffh/ffh_grasp_visualization/1/BathDetergent_depth.npy')
    pcd_center = segment_object_as_point_cloud(color_img, depth_img, select_ROI(color_img), gc.object_pcd_save_path)
    # gc.remove_ground_plane('/home/ffh/ffh_grasp_visualization/1/point_cloud_0000.pcd')
    gc.encode_pcd_with_bps()
    palm_poses_obj_frame, joint_confs = gc.infer_grasp_poses(n_poses=N_POSES, visualize_poses=True)

    gc.object_metadata = {'name' : 'test_eval_real_images'}
    # Evaluate the generated poses according to the FFHEvaluator
    palm_poses_obj_frame, joint_confs = gc.evaluate_and_remove_grasps(
        palm_poses_obj_frame, joint_confs, 
        thresh=FILTER_THRESH, 
        visualize_poses=True
    )
    # translate back to camera frame
    for i in range(len(palm_poses_obj_frame)):
        palm_poses_obj_frame[i].pose.position.x += pcd_center[0]
        palm_poses_obj_frame[i].pose.position.y += pcd_center[1]
        palm_poses_obj_frame[i].pose.position.z += pcd_center[2]
    
    np.save('/home/ffh/Downloads/DexFFHNet_test/set_1/grasp_poses_000.npy', palm_poses_obj_frame)
    np.save('/home/ffh/Downloads/DexFFHNet_test/set_1/joint_confs_000.npy', joint_confs)
    pcd_center


def segment_object_as_point_cloud(color_image, depth_image, ROI, pcd_save_path):
    # Create mask
    mask = np.zeros((color_image.shape[0], color_image.shape[1]), np.uint8)

    # GrabCut arrays
    bgdModel = np.zeros((1, 65), np.float64)
    fgbModel = np.zeros((1, 65), np.float64)

    # Run GrabCut
    init_rect = ROI
    cv2.grabCut(color_image, mask, init_rect, bgdModel, fgbModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    masked_image = color_image * mask2[:, :, np.newaxis]
    
    # Set area outside of the segmentation mask to zero
    depth_image *= mask2

    # Remove data with large depth offset from segmented object's median
    median = np.median(depth_image[depth_image > 0])
    depth_image = np.where(abs(depth_image - median) < 100, depth_image, 0)

    # Load depth image as o3d.Image
    depth_image_o3d = o3d.geometry.Image(depth_image)

    # Generate point cloud from depth image
    pinhole_camera_intrinsic = get_camera_intrinsics_sim()
    object_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, pinhole_camera_intrinsic)
    point_cloud_center = object_pcd.get_center()
    object_pcd.translate((-1) * point_cloud_center)
    o3d.io.write_point_cloud(pcd_save_path, object_pcd)
    return point_cloud_center

def read_image(color_path, depth_path):
    color_img = cv2.imread(color_path)
    depth_img = np.load(depth_path)
    return color_img, depth_img

def get_camera_intrinsics_real():
    image_width = 1280
    image_height = 720
    fx = 924.275
    fy = 923.591
    cx = 622.875
    cy = 349.164

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    return pinhole_camera_intrinsic

def get_camera_intrinsics_sim():
    image_width = 1280
    image_height = 720
    horizontal_fov = math.radians(64)
    fx = 0.5 * image_width / math.tan(0.5 * horizontal_fov)
    fy = fx
    cx = image_width * 0.5
    cy = image_height * 0.5

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    return pinhole_camera_intrinsic

def select_ROI(image, close_window=True):
    while True:
        cv2.namedWindow("Seg", cv2.WND_PROP_FULLSCREEN)
        try:
            roi = cv2.selectROI('Seg', image, False, False)
        except:
            roi = [0]

        if not any(roi):
            print("No area selected. Press 'c' to abort or anything else to reselect")
            if cv2.waitKey(0) == ord('c'):
                exit()
        else:
            # user selected something
            break
    if close_window:
        cv2.destroyWindow("Seg")
    return roi

def get_sim_world_T_camera_mat():
    return np.array([
        [ 1.00000000e+00, -1.44707624e-12, -2.18525541e-13, 4.80000000e-01],
        [-2.18874845e-13, -2.95520207e-01, 9.55336489e-01, -8.46601518e-01],
        [-1.44702345e-12, -9.55336489e-01, -2.95520207e-01, 3.60986370e-01],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

def get_sim_camera_T_world_mat():
    return np.linalg.inv(get_sim_world_T_camera_mat())

def visualize_grasp_poses(color_img, depth_img, palm_poses_obj_frame):
    scene_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False)
    scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(scene_rgbd, get_camera_intrinsics_real())
    grasp_poses_visual_frame = []
    for frame in palm_poses_obj_frame:
        origin = np.array([frame.pose.position.x, frame.pose.position.y, frame.pose.position.z])
        quat = np.array([frame.pose.orientation.w, frame.pose.orientation.x, frame.pose.orientation.y, frame.pose.orientation.z])
        rot_mat = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(quat)
        grasp_poses_visual_frame.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015, origin=origin).rotate(rot_mat))
    o3d.visualization.draw_geometries(grasp_poses_visual_frame + [scene_pcd])


def visualize_grasp_poses_grasp_dict(color_img, depth_img, grasp_dict):
    scene_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False, depth_scale=1)
    scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(scene_rgbd, get_camera_intrinsics_sim())
    grasp_poses_visual_frame = []
    
    global pcd_center
    print grasp_dict['transl'].shape[0]
    for i in range(grasp_dict['transl'].shape[0]):
        origin = grasp_dict['transl'][i] + pcd_center
        rot_mat = grasp_dict['rot_matrix'][i]
        grasp_poses_visual_frame.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015, origin=origin).rotate(rot_mat))
    o3d.visualization.draw_geometries(grasp_poses_visual_frame + [scene_pcd])

def helper_visualize():
    color_img, depth_img = read_image('/home/ffh/ffh_grasp_visualization/1/BathDetergent_color.jpg', '/home/ffh/ffh_grasp_visualization/1/BathDetergent_depth.npy')
    grasp_dicts = [] 
    grasp_dicts.append(np.load('/home/ffh/ffh_outputs/generator.npy', allow_pickle=True).item())
    grasp_dicts.append(np.load('/home/ffh/ffh_outputs/coll_detector.npy', allow_pickle=True).item())
    grasp_dicts.append(np.load('/home/ffh/ffh_outputs/evaluator.npy', allow_pickle=True).item())
    color_img = o3d.geometry.Image(color_img)
    depth_img = o3d.geometry.Image(depth_img)
    for grasp_dict in grasp_dicts:
        visualize_grasp_poses_grasp_dict(color_img, depth_img, grasp_dict)

if __name__ == '__main__':
    main()
    helper_visualize()