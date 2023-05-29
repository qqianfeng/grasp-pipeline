from ctypes import *
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import open3d as o3d
import rospy
import pandas as pd
import pylab
import tf.transformations as tft

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, JointState
import sensor_msgs.point_cloud2 as pc2


def convert_to_full_voxel_grid(sparse_grid):
    full_grid = np.zeros([32, 32, 32])
    for i in xrange(len(sparse_grid)):
        ix1, ix2, ix3 = sparse_grid[i]
        full_grid[ix1, ix2, ix3] = 1

    return full_grid


def convert_to_sparse_voxel_grid(voxel_grid, threshold=0.5):
    sparse_voxel_grid = []
    voxel_dim = voxel_grid.shape
    for i in xrange(voxel_dim[0]):
        for j in xrange(voxel_dim[1]):
            for k in xrange(voxel_dim[2]):
                if voxel_grid[i, j, k] > threshold:
                    sparse_voxel_grid.append([i, j, k])
    return np.asarray(sparse_voxel_grid)


def full_joint_conf_from_vae_joint_conf(vae_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        vae_joint_conf (JointState): Output from vae with dim(vae_joint_conf.position) = 15

    Returns:
        full_joint_conf (JointState): Full joint state with dim(full_joint_conf.position) = 20
    """
    full_joint_pos = 20 * [0]
    ix_full_joint_pos = 0
    for i, val in enumerate(vae_joint_conf.position):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = val
            full_joint_pos[ix_full_joint_pos + 1] = val
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = val
            ix_full_joint_pos += 1

    full_joint_conf = JointState(position=full_joint_pos)
    return full_joint_conf


def get_pose_stamped_from_array(pose_array, frame_id='/world'):
    """Transforms an array pose into a ROS stamped pose.
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    # RPY to quaternion
    pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = \
        pose_array[0], pose_array[1], pose_array[2]
    pose_quaternion = tft.quaternion_from_euler(pose_array[3], pose_array[4], pose_array[5])
    pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, \
            pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = pose_quaternion[0], pose_quaternion[1], pose_quaternion[2], pose_quaternion[3]

    return pose_stamped


def get_rot_quat_list_from_array(pose_array):
    pose_quaternion = tft.quaternion_from_euler(pose_array[3], pose_array[4], pose_array[5])
    rot_quat_list = pose_array[:3]
    rot_quat_list += pose_quaternion.tolist()
    return rot_quat_list


def get_array_from_rot_quat_list(rot_quat_list):
    r, p, y = tft.euler_from_quaternion(rot_quat_list[3:])

    pose_array = rot_quat_list[:3].tolist()
    pose_array = pose_array + [r, p, y]
    return pose_array


def get_pose_stamped_from_rot_quat_list(pose_list, frame_id="world"):
    """ Transform a list of 3 position and 4 orientation/quaternion elements to a stamped pose
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    # RPY to quaternion
    pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = \
        pose_list[0], pose_list[1], pose_list[2]
    pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, \
            pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = pose_list[3], pose_list[4], pose_list[5], pose_list[6]

    return pose_stamped


def get_pose_stamped_from_trans_and_quat(trans, quat, frame_id='world'):
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'world'
    pose_stamped.pose.position.x = trans[0]
    pose_stamped.pose.position.y = trans[1]
    pose_stamped.pose.position.z = trans[2]

    pose_stamped.pose.orientation.x = quat[0]
    pose_stamped.pose.orientation.y = quat[1]
    pose_stamped.pose.orientation.z = quat[2]
    pose_stamped.pose.orientation.w = quat[3]

    return pose_stamped


def get_pose_array_from_stamped(pose_stamped):
    """Transforms a stamped pose into a 6D pose array.
        """
    r, p, y = tft.euler_from_quaternion([
        pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w
    ])
    x_p, y_p, z_p = pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z
    pose_array = [x_p, y_p, z_p, r, p, y]
    return pose_array


def get_rot_trans_list_from_pose_stamp(pose_stamped):
    return [
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w,
    ]


def hom_matrix_from_pos_quat_list(rot_quat_list):
    p = rot_quat_list[:3]
    q = rot_quat_list[3:]
    T = tft.quaternion_matrix(q)
    T[:3, 3] = p
    return T


def hom_matrix_from_pose_stamped(pose_stamped):
    q = pose_stamped.pose.orientation
    r = pose_stamped.pose.position
    hom_matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    hom_matrix[:, 3] = [r.x, r.y, r.z, 1]
    return hom_matrix


def hom_matrix_from_pose(pose):
    q = pose.orientation
    r = pose.position
    hom_matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    hom_matrix[:, 3] = [r.x, r.y, r.z, 1]
    return hom_matrix


def hom_matrix_from_ros_transform(transform_ros):
    """ Transform a ROS transform to a 4x4 homogenous numpy array
    """
    q = transform_ros.transform.rotation
    t = transform_ros.transform.translation
    hom_matrix = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    hom_matrix[:, 3] = [t.x, t.y, t.z, 1]
    return hom_matrix


# Convert the datatype of point pcd from o3d to ROS Pointpcd2 (XYZRGB only)
def hom_matrix_from_rot_matrix(rot_matrix):
    assert rot_matrix.shape == (3, 3)
    hom_matrix = np.eye(4)
    hom_matrix[:3, :3] = rot_matrix
    return hom_matrix


def hom_matrix_from_6D_pose(pos, ori):
    hom = tft.euler_matrix(ori[0], ori[1], ori[2])
    hom[:3, 3] = pos
    return hom


def list_of_objects_from_folder(folder_path):
    dirs = os.listdir(folder_path)
    objects = []
    for dir in dirs:
        obj_name = '_'.join(dir.split('_')[1:])
        if obj_name not in objects:
            objects.append(obj_name)
    return sorted(objects)


def pose_stamped_from_hom_matrix(hom_matrix, frame_id):
    q = tft.quaternion_from_matrix(hom_matrix)

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose.position.x = hom_matrix[0, 3]
    pose_stamped.pose.position.y = hom_matrix[1, 3]
    pose_stamped.pose.position.z = hom_matrix[2, 3]
    pose_stamped.pose.orientation.x = q[0]
    pose_stamped.pose.orientation.y = q[1]
    pose_stamped.pose.orientation.z = q[2]
    pose_stamped.pose.orientation.w = q[3]

    return pose_stamped


def plot_voxel(voxel, img_path=None, voxel_res=None, centroid=None, pca_axes=None):
    fig = pyplot.figure()

    ax = fig.add_subplot(111, projection='3d')

    if len(voxel) != 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('voxel')

    if voxel_res is not None:
        ax.set_xlim3d(0, voxel_res[0])
        ax.set_ylim3d(0, voxel_res[1])
        ax.set_zlim3d(0, voxel_res[2])

    # if centroid is not None and pca_axes is not None:
    #     for pca_ax in pca_axes:
    #         ax.plot([centroid[0], centroid[0] + pca_ax[0]], [centroid[1], centroid[1] + pca_ax[1]], [centroid[2], centroid[2] + pca_ax[2]], ax)

    pyplot.show()
    if img_path is not None:
        pyplot.savefig(img_path)


def trans_rot_list_from_ros_transform(ros_transform):
    t = ros_transform.transform.translation
    q = ros_transform.transform.rotation
    return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]


def wait_for_service(service_name):
    rospy.logdebug('Waiting for service ' + service_name)
    rospy.wait_for_service(service_name)
    rospy.logdebug('Calling service ' + service_name)


def get_objects_few_grasps(n_min, base_path='/home/vm/data/ffhnet-data'):
    """Get a list of objects with less positive grasps than threshold

    Args:
        n_min (int): Minimum number of successful grasps an object should have
        base_path (str): Base path to where the metadata.csv file lies
    """
    file_path = os.path.join(base_path, 'metadata.csv')
    df = pd.read_csv(file_path)
    objs = df[df['remove positive'] == 'X'].loc[:, 'Unnamed: 0']
    for obj in objs:
        print("'" + obj + "',")


def reduce_joint_conf(jc_full):
    """Turn the 20 DoF input joint array into 15 DoF by either dropping each 3rd or 4th joint value, depending on which is smaller.

    Args:
        jc_full (np array): 20 dimensional array of hand joint values

    Returns:
        jc_red (np array): 15 dimensional array of reduced hand joint values
    """
    idx = 0
    jc_red = np.zeros((15, ))
    for i, _ in enumerate(jc_red):
        if (i + 1) % 3 == 0:
            if jc_full[idx + 1] > jc_full[idx]:
                jc_red[i] = jc_full[idx + 1]
            else:
                jc_red[i] = jc_full[idx]
            idx += 2
        else:
            jc_red[i] = jc_full[idx]
            idx += 1
    return jc_red


if __name__ == '__main__':
    # folder_path = '/home/vm/Documents/grasp_data_generated_on_this_machine/2021-04-09_02/grasp_data/recording_sessions/recording_session_0001'
    # l = list_of_objects_from_folder(folder_path)
    # print(l)
    l = get_objects_few_grasps(150)

    #mesh = o3d.io.read_triangle_mesh()
    #pcd = o3d.io.read_point_cloud()
    #o3d.visualization.draw_geometries([mesh, pcd])