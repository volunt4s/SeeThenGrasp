import numpy as np
import geometry_msgs.msg
import rospy
import tf
import json

from sensor_msgs.msg import CameraInfo
from fr_msgs.srv import SavedPos
from fr_msgs.srv import PoseSer

from tf.transformations import quaternion_from_matrix
from tf.transformations import translation_matrix
from tf.transformations import translation_from_matrix
from tf.transformations import quaternion_multiply
from tf.transformations import quaternion_matrix

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def vector_to_rotation_matrix(vector, theta):
    vector = vector / np.linalg.norm(vector)

    I = np.eye(3)
    
    K = np.array([[0.0, -vector[2], vector[1]],
                  [vector[2], 0.0, -vector[0]],
                  [-vector[1], vector[0], 0.0]])
    
    # Rodrigues' formula
    R = I + (np.sin(theta) * K) + (1 - np.cos(theta)) * np.matmul(K, K)
    return R

def cal_target_pose(pos =[0.307,0,0.59], rot = [1.,0.,0.,0.]):
        target_pose = geometry_msgs.msg.Pose()

        target_pose.position.x=float(pos[0])
        target_pose.position.y=float(pos[1])
        target_pose.position.z=float(pos[2])

        target_pose.orientation.x = float(rot[0])
        target_pose.orientation.y = float(rot[1])
        target_pose.orientation.z = float(rot[2])
        target_pose.orientation.w = float(rot[3])

        return target_pose

def generate_cam_pose(r, theta, phi, z_axis_flip=False):
    point = spherical_to_cartesian(r, theta, phi)
    x, y, z = point
    rz = -np.array([x, y, z])
    rz = rz / np.linalg.norm(rz)
    rx = np.array([-y,x,0])
    rx = rx / np.linalg.norm(rx)
    ry = np.cross(rz,rx)
    tf_mat = np.eye(4)
    tf_mat[:3,0] = rx
    tf_mat[:3,1] = ry
    tf_mat[:3,2] = rz
    tf_mat[:3,3] = [x,y,z]
    print(tf_mat)
    return tf_mat


def generate_eef_pose(desired_frame,
                      object_frame_trans=None,
                      object_frame_quat=None,
                      form = "pose", do_y_axis_flip=False):
    y_axis_flip = np.array([0.0, 1.0, 0.0, 0.0])
    z_axis_half = np.array([0.0, np.sin(np.pi/2), 0.0, np.cos(np.pi/2)])

    # Calculate cam to eef transformation matrix
    target_frame = "calibed_optical_frame"
    source_frame = "panda_hand"
    listener = tf.TransformListener()
    listener.waitForTransform(target_frame, source_frame,
                              rospy.Time(), rospy.Duration(4.0))
    (cam_to_eef_trans, cam_to_eef_quat) = listener.lookupTransform(target_frame,
                                                                   source_frame,
                                                                   rospy.Time(0))
    cam_to_eef_mat_rot = quaternion_matrix(cam_to_eef_quat)
    cam_to_eef_mat_trans = translation_matrix(cam_to_eef_trans)
    cam_to_eef_mat = cam_to_eef_mat_rot + cam_to_eef_mat_trans - np.eye(4)


    # Calculate base to cam transformation matrix
    # if desired_frame is pose
    if form == "pose" :        
        
        base_to_obj_mat_rot = quaternion_matrix(object_frame_quat)
        base_to_obj_mat_trans = translation_matrix(object_frame_trans)
        base_to_obj_mat = base_to_obj_mat_trans + base_to_obj_mat_rot - np.eye(4)

        obj_to_cam_mat = desired_frame

        base_to_cam_mat = np.dot(base_to_obj_mat,obj_to_cam_mat)

    # Calculate base to eef transformation matrix
    base_to_eef_mat = np.dot(base_to_cam_mat,cam_to_eef_mat)
    base_to_eef_trans, base_to_eef_quat = translation_from_matrix(base_to_eef_mat), quaternion_from_matrix(base_to_eef_mat)
    if do_y_axis_flip:
        base_to_eef_quat = quaternion_multiply(base_to_eef_quat, y_axis_flip)

    # Calculate target pose
    pose_goal = cal_target_pose(pos=base_to_eef_trans, rot=base_to_eef_quat)
    return pose_goal

def generate_eef_pose2(obj_to_cam_mat):
    """Generate base to eef pose using a obj to cam2 matrix


    Parameters
    ----------
    obj_to_cam_mat : pose(h_mat)
        desired_frame information
    """

    get_saved_pos = rospy.ServiceProxy("FR_Robot/pub_saved_pose", SavedPos)

    base_to_cam = np.array(get_saved_pos().other_opt_frame).reshape(4,4)
    eef_to_obj = np.array(get_saved_pos().grasp_to_cent).reshape(4,4)

    base_to_eef_mat = base_to_cam @ np.linalg.inv(obj_to_cam_mat) @ np.linalg.inv(eef_to_obj)
    return base_to_eef_mat

def get_req_msg_pose_from_mat(mat):
    pose_request = PoseSer._request_class()
    quat = quaternion_from_matrix(mat)

    pose_request.position.x = mat[0,3]
    pose_request.position.y = mat[1,3]
    pose_request.position.z = mat[2,3]
    pose_request.orientation.x = quat[0]
    pose_request.orientation.y = quat[1]
    pose_request.orientation.z = quat[2]
    pose_request.orientation.w = quat[3]

    return pose_request

def save_as_json_init(file_path):
    data = {}
    cam_info_msg = rospy.wait_for_message("/camera/color/camera_info",CameraInfo)
    # Convert CameraInfo message to dictionary
    cam_info_dict = {
        'header': {
            'seq': cam_info_msg.header.seq,
            'stamp': {
                'secs': cam_info_msg.header.stamp.secs,
                'nsecs': cam_info_msg.header.stamp.nsecs
            },
            'frame_id': cam_info_msg.header.frame_id
        },
        'height': cam_info_msg.height,
        'width': cam_info_msg.width,
        'distortion_model': cam_info_msg.distortion_model,
        'D': list(cam_info_msg.D),
        'K': list(cam_info_msg.K),
        'R': list(cam_info_msg.R),
        'P': list(cam_info_msg.P),
        'binning_x': cam_info_msg.binning_x,
        'binning_y': cam_info_msg.binning_y,
        'roi': {
            'x_offset': cam_info_msg.roi.x_offset,
            'y_offset': cam_info_msg.roi.y_offset,
            'height': cam_info_msg.roi.height,
            'width': cam_info_msg.roi.width,
            'do_rectify': cam_info_msg.roi.do_rectify
        }
    }
    data["cam_info"] = cam_info_dict
    
    data["frames"] = []

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def save_as_json(json_file_path,file_name, tf_pose):
    tf_mat  = translation_matrix(tf_pose[0]) + quaternion_matrix(tf_pose[1]) - np.eye(4)

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    data["frames"].append({
        "file_name": f"./image/{file_name}",
        "mask_path": f"./mask/{file_name}",
        "transform_matrix": tf_mat.tolist()
    })

    with open(json_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)