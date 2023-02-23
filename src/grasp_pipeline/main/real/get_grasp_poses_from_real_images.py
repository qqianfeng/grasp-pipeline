import cv2
import numpy as np
from grasp_pipeline.grasp_client.grasp_sim_client import GraspClient
import open3d as o3d
import rospy
import os

N_POSES = 400
FILTER_THRESH = -1  # set to -1 if no filtering desired, default 0.9
FILTER_NUM_GRASPS = 5
NUM_TRIALS_PER_OBJ = 20
NUM_OBSTACLE_OBJECTS = 3
path2grasp_data = os.path.join(os.path.expanduser("~"), 'grasp_data')
object_datasets_folder = rospy.get_param('object_datasets_folder')
gazebo_objects_path = os.path.join(object_datasets_folder, 'objects_gazebo')

def main():
    data_recording_path = rospy.get_param('data_recording_path')
    gc = GraspClient(grasp_data_recording_path=data_recording_path, is_rec_sess=True, is_eval_sess=True)
    color_img, depth_img = read_image('/home/ffh/Downloads/DexFFHNet_test/set_1/color_0000.png', '/home/ffh/Downloads/DexFFHNet_test/set_1/depth_0000.npy')
    pcd_center = segment_object_as_point_cloud(color_img, depth_img, select_ROI(color_img), gc.object_pcd_save_path)
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
    pinhole_camera_intrinsic = get_camera_intrinsics()
    object_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, pinhole_camera_intrinsic)
    point_cloud_center = object_pcd.get_center()
    object_pcd.translate((-1) * point_cloud_center)
    o3d.io.write_point_cloud(pcd_save_path, object_pcd)
    return point_cloud_center

def read_image(color_path, depth_path):
    color_img = cv2.imread(color_path)
    depth_img = np.load(depth_path)
    return color_img, depth_img

def get_camera_intrinsics():
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


if __name__ == '__main__':
    main()