from sensor_msgs.msg import Image, PointCloud2, PointField, JointState
import open3d as o3d
from ctypes import *
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


def convert_rgbUint32_to_tuple(rgb_uint32):
    return ((rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8,
            (rgb_uint32 & 0x000000ff))


def convert_rgbFloat_to_tuple(rgb_float):
    return convert_rgbUint32_to_tuple(
        int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value))


def pcd_from_ros_to_o3d(ros_pcd):
    # Get pcd data from ros_pcd
    field_names = [field.name for field in ros_pcd.fields]
    pcd_data = list(pc2.read_points(ros_pcd, skip_nans=True, field_names=field_names))
    # Check empty
    o3d_pcd = o3d.geometry.PointCloud()
    if len(pcd_data) == 0:
        print("Converting an empty pcd")
        return None
    # Set o3d_pcd
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD = 3  # x, y, z, rgb
        # Get xyz
        # (why cannot put this line below rgb?)
        xyz = [(x, y, z) for x, y, z, rgb in pcd_data]
        # Get rgb
        # Check whether int or float
        # if float (from pcl::toROSMsg)
        if type(pcd_data[0][IDX_RGB_IN_FIELD]) == float:
            rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in pcd_data]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in pcd_data]
        # combine
        o3d_pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
        o3d_pcd.colors = o3d.utility.Vector3dVector(np.array(rgb) / 255.0)
    else:
        xyz = [(x, y, z) for x, y, z in pcd_data]  # get xyz
        o3d_pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    return o3d_pcd


# Convert the datatype of point pcd from o3d to ROS Pointpcd2 (XYZRGB only)


def pcd_from_o3d_to_ros(o3d_pcd, frame_id):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # Set "fields" and "pcd_data"
    points = np.asarray(o3d_pcd.points)
    if not o3d_pcd.colors:  # XYZ only
        fields = FIELDS_XYZ
        pcd_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(o3d_pcd.colors) * 255)  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + \
            colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        pcd_data = np.c_[points, colors]
    # create ros_pcd
    return pc2.create_cloud(header, fields, pcd_data)


def wait_for_service(service_name):
    rospy.loginfo('Waiting for service ' + service_name)
    rospy.wait_for_service(service_name)
    rospy.loginfo('Calling service ' + service_name)


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


def get_utah_grasp_config_from_pose_and_joints(palm_pose, joint_pos):
    # Get pos and quat
    pos = palm_pose.pose.position
    q = palm_pose.pose.orientation
    quat = [q.x, q.y, q.z, q.w]
    # "allocate" memory for full pose 6D and 10 joints
    pose_array = np.zeros(6 + 10)

    # Bring ori from quaternion to euler
    r, p, y = tft.euler_from_quaternion(quat)

    # Insert pos and euler in right places
    pose_array[:3] = [pos.x, pos.y, pos.z]
    pose_array[3:6] = [r, p, y]

    # Insert joint angles in the right place
    # Index, Little, Middle, Ring, Thumb
    des_joints = []
    for i in range(0, len(joint_pos), 4):
        des_joints.append(joint_pos[i])
        des_joints.append(joint_pos[i + 1])
    assert len(des_joints) == 10
    pose_array[6:] = des_joints

    return pose_array


if __name__ == '__main__':
    js = JointState()
    js.position = np.random.randint(0, 100, 20)

    palm_pose = PoseStamped()

    get_utah_grasp_config_from_pose_and_joints(palm_pose, js)