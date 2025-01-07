import rospy
import numpy as np
from fr_msgs.srv import PoseSer
from fr_msgs.srv import EmptySer
from fr_msgs.srv import SavedPos
from fr_msgs.srv import Image2Pos
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import tf,os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tf.transformations import quaternion_from_matrix
from tf.transformations import translation_from_matrix
import moveit_commander
import control_src.control_util as cu
from control_src.image_saver import ImageSaver
import cv2
import glob
from tf.transformations import translation_matrix
from tf.transformations import quaternion_matrix

bgr2rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

if __name__ == '__main__':
    rospy.init_node('gen_cam2_pose')  # Initialize ROS node

    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    add_table = rospy.ServiceProxy('/FR_Robot/add_table', Empty)
    add_scene = rospy.ServiceProxy('/FR_Robot/add_scene', Empty)
    remove_table = rospy.ServiceProxy('/FR_Robot/remove_table', Empty)

    resp = add_table()

    go_to_start = rospy.ServiceProxy('/FR_Robot/go_to_start', EmptySer)
    go_to_pose_goal = rospy.ServiceProxy('/FR_Robot/pose_goal', PoseSer)
    check_plan = rospy.ServiceProxy('/FR_Robot/check_plan', PoseSer)
    check_grasp = rospy.ServiceProxy('/FR_Robot/check_grasp', PoseSer)
    move_grasp = rospy.ServiceProxy('/FR_Robot/move_grasp', PoseSer)
    move_release = rospy.ServiceProxy('/FR_Robot/move_release', PoseSer)

    # load grasp data
    package_path = "your panda ros package directory" # input your panda ros package
    savedpose = np.load(package_path + 'results/pos_for_cam2.npz')
    success_idx = savedpose['success_idx']
    eef_pose    = savedpose['eef_pose']
    obj_to_cam2 = savedpose['obj_to_cam2']

    cam1_image_cube_path = glob.glob("pre_generated_data/image_cube/*.png")
    cam1_image_cube_path = sorted(cam1_image_cube_path)
    cam2_image_cube_path = os.path.join("pre_generated_data", "image_cube2")
    os.makedirs(cam2_image_cube_path, exist_ok=True)

    img_to_extrinsic = rospy.ServiceProxy("/FR_Robot/get_ext_from_camera2", Image2Pos)
    cam2_cube_overlay_path = os.path.join("pre_generated_data", "overlay")
    os.makedirs(cam2_cube_overlay_path, exist_ok=True)
    cube_info = {}

    cam2_joint_info = {} 
    image_saver = ImageSaver()
    cam2_cnt = 0

    pass_ratio = 2

    print(eef_pose.shape)

    for eef_idx, one_eef_pose in enumerate(eef_pose):
        req_msg_pose = cu.get_req_msg_pose_from_mat(one_eef_pose)

        if eef_idx % pass_ratio != 0:
            continue
    
        if eef_idx in success_idx:
            plan_reached = check_plan(req_msg_pose).success
            
            if plan_reached: 
                reached = go_to_pose_goal(req_msg_pose).success
                
                if reached:
                    cam1_len = len(cam1_image_cube_path)
                    # print(f'{i} move success')
                    cam2_idx = cam1_len + cam2_cnt
                    print(f"{cam2_idx} moved [{eef_idx}/{len(eef_pose)}]")

                    if cam2_idx < 100:
                        idx_name = f"0{cam2_idx}"
                    else:
                        idx_name = f"{cam2_idx}"

                    current_image = image_saver.get_current_image("camera2")
                    current_image_ros = CvBridge().cv2_to_imgmsg(current_image, "bgr8")
                    try:
                        resp_april = img_to_extrinsic(current_image_ros)
                        overlay = CvBridge().imgmsg_to_cv2(resp_april.overlay, "bgr8")
                        hmat = pose_stamp_to_hmat(resp_april.pose)
                    except:
                        print("Cannot get extrinsic")
                        idx_name = idx_name + "_detect_failed"
                        overlay = current_image
                        hmat = np.eye(3)
                    
                    cube_info[f"{idx_name}_ext"] = hmat
                    cube_info[f"{idx_name}_int"] = image_saver.get_intrinsic("camera2")

                    joint_value = move_group.get_current_joint_values()
                    cam2_joint_info[f"{idx_name}_joint"] = joint_value

                    cv2.imwrite(os.path.join(cam2_image_cube_path, f"{idx_name}.png"),
                                bgr2rgb(current_image))
                    cv2.imwrite(os.path.join(cam2_cube_overlay_path, f"{idx_name}_overlay.png"),
                                overlay)
                    cam2_cnt = cam2_cnt + 1
            else:
                 print(f'{eef_idx} plan failed')

    print("success check fin")
    print(success_idx)
    
    np.savez(os.path.join("pre_generated_data", "cam2_cube_info.npz"), **cube_info)
    np.savez(os.path.join("pre_generated_data", "cam2_joint_info.npz"), **cam2_joint_info)
    # resp = remove_table()