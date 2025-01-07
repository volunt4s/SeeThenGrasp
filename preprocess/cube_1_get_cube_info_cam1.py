import numpy as np
import cv2
import os
import copy
from control_src.image_saver import ImageSaver
from control_src.control_util import generate_cam_pose
from control_src.control_util import generate_eef_pose
from fr_msgs.srv import PoseSer
import matplotlib.pyplot as plt
import tf
import rospy
from std_srvs.srv import Empty
from fr_msgs.srv import Image2Pos
from cv_bridge import CvBridge
import moveit_commander
from tf.transformations import translation_matrix
from tf.transformations import quaternion_matrix

bgr2rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def get_object_center_frame(tf_listener):
    tf_listener.waitForTransform(target_frame="panda_link0",
                                 source_frame="object_center_frame",
                                 time=rospy.Time(),
                                 timeout=rospy.Duration(1.0))
    object_trans, object_quat = tf_listener.lookupTransform(target_frame="panda_link0",
                                                            source_frame="object_center_frame",
                                                            time=rospy.Time())
    return copy.deepcopy(object_trans), copy.deepcopy(object_quat)

def pose_stamp_to_hmat(ros_pose):
    position = np.array([
        ros_pose.position.x,
        ros_pose.position.y,
        ros_pose.position.z
    ])
    orientation = np.array([
        ros_pose.orientation.x,
        ros_pose.orientation.y,
        ros_pose.orientation.z,
        ros_pose.orientation.w
    ])
    trans_mat = translation_matrix(position)
    rot_mat = quaternion_matrix(orientation)

    return trans_mat + rot_mat - np.eye(4)

rospy.init_node("only_get_images")

go_to_pose_goal = rospy.ServiceProxy('/FR_Robot/pose_goal', PoseSer)    
tf_listener = tf.TransformListener()

add_table = rospy.ServiceProxy('/FR_Robot/add_table', Empty)
remove_table = rospy.ServiceProxy('/FR_Robot/remove_table', Empty)

resp = add_table()

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

r = 0.25
theta = np.linspace(np.pi/7, np.pi/3.5, num=8)
phi = np.linspace((np.pi+np.pi/7), (2*np.pi-np.pi/7), num=8)

test_pose_set = []
for one_theta in theta:
    for one_phi in phi:
        test_pose_set.append(generate_cam_pose(r, one_theta, one_phi))
    phi = np.flip(phi)

object_frame_trans, object_frame_quat = get_object_center_frame(tf_listener)
test_eef_pose = [generate_eef_pose(one_test_pose, object_frame_trans, object_frame_quat) for one_test_pose in test_pose_set]

cube_img_path = os.path.join("pre_generated_data", "image_cube")
overlay_path = os.path.join("pre_generated_data", "overlay")
os.makedirs(cube_img_path, exist_ok=True)
os.makedirs(overlay_path, exist_ok=True)

img_to_extrinsic = rospy.ServiceProxy("/FR_Robot/get_ext_from_camera", Image2Pos)
cube_info = {}

cam1_joint_info = {}


# save initial pose
image_saver = ImageSaver()
current_image = image_saver.get_current_image("camera")
current_image_ros = CvBridge().cv2_to_imgmsg(current_image, "bgr8")
try:
    resp_april = img_to_extrinsic(current_image_ros)
    overlay = CvBridge().imgmsg_to_cv2(resp_april.overlay, "bgr8")
    hmat = pose_stamp_to_hmat(resp_april.pose)
except:
    print("Cannot get extrinsic")
    overlay = current_image
    hmat = np.eye(3)

cube_info[f"init_ext"] = hmat
cube_info[f"init_int"] = image_saver.get_intrinsic("camera")


## Actual moving
for i, one_eef_pose in enumerate(test_eef_pose):
    print(i)
    msg_goal = PoseSer._request_class()
    msg_goal.position = one_eef_pose.position
    msg_goal.orientation = one_eef_pose.orientation
    resp_pose = go_to_pose_goal(msg_goal)
    reached = resp_pose.success
    print(reached)        
    
    if reached:
        joint_value = move_group.get_current_joint_values()

        image_saver = ImageSaver()
        current_image = image_saver.get_current_image("camera")
        current_image_ros = CvBridge().cv2_to_imgmsg(current_image, "bgr8")
        try:
            resp_april = img_to_extrinsic(current_image_ros)
            overlay = CvBridge().imgmsg_to_cv2(resp_april.overlay, "bgr8")
            hmat = pose_stamp_to_hmat(resp_april.pose)
        except:
            print("Cannot get extrinsic")
            overlay = current_image
            hmat = np.eye(3)

        if i < 100 and i < 10:
            idx_name = f"00{i}"
        elif i < 100 and i >= 10:
            idx_name = f"0{i}"
        else:
            idx_name = f"{i}"

        cube_info[f"{idx_name}_ext"] = hmat
        cube_info[f"{idx_name}_int"] = image_saver.get_intrinsic("camera")

        cv2.imwrite(os.path.join(cube_img_path,f"{idx_name}.png"),
                    bgr2rgb(current_image))
        cv2.imwrite(os.path.join(overlay_path, f"{idx_name}_overlay.png"),
                    overlay)
        
        cam1_joint_info[f"{idx_name}_joint"] = joint_value
        print()
        
np.savez(os.path.join("pre_generated_data", "cam1_cube_info.npz"), **cube_info)
np.savez(os.path.join("pre_generated_data", "cam1_joint_info.npz"), **cam1_joint_info)
