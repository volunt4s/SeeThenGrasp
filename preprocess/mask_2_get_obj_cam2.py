import os
import cv2
import re
import rospy
import moveit_commander
import numpy as np
from cv_bridge import CvBridge
from fr_msgs.srv import JointSer
from fr_msgs.srv import Image2Pos
from std_srvs.srv import Empty
from control_src.image_saver import ImageSaver
from tf.transformations import translation_matrix
from tf.transformations import quaternion_matrix

bgr2rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

rospy.init_node('only_get_obj')

cam2_joint_info = np.load("pre_generated_data/cam2_joint_info.npz")

go_to_joint_target = rospy.ServiceProxy('/FR_Robot/go_to_joint_target',
                                        JointSer)

cam2_obj_img_path = os.path.join("pre_generated_data", "image_obj_cam2")
os.makedirs(cam2_obj_img_path, exist_ok=True)

group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

get_joint_error = lambda des, cur: np.linalg.norm(des - cur)

print(cam2_joint_info.files)

add_table = rospy.ServiceProxy('/FR_Robot/add_table', Empty)
remove_table = rospy.ServiceProxy('/FR_Robot/remove_table', Empty)

resp = add_table()


# actual move
for i, joint_key in enumerate(cam2_joint_info.files):
    idx_name = re.search(r'^\d+', joint_key).group(0)
    try:
        joint_state = cam2_joint_info[joint_key]
        msg_joint_goal = JointSer._request_class()
        msg_joint_goal.joint_angles = joint_state
        resp = go_to_joint_target(msg_joint_goal)
        
        current_joint = move_group.get_current_joint_values()
        print(f"[{i+1} / {len(cam2_joint_info.files)}] {idx_name} error : {get_joint_error(joint_state, current_joint):.5f}")

        image_saver = ImageSaver()
        current_image = image_saver.get_current_image("camera2")

        cv2.imwrite(os.path.join(cam2_obj_img_path, f"{idx_name}.png"),
                    bgr2rgb(current_image))
    except:
        print(f">>>>>>>>>> cannot go to : {joint_key}")

# resp = remove_table()