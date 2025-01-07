import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class ImageSaver:
    """
    Image saver class
    """
    def __init__(self):
        self.rgb_image = None

    def get_current_image(self, camera='camera'):
        """
        Get current state image using wait_for_message
        """
        msg = rospy.wait_for_message(f"/{camera}/color/image_raw", Image)
        self.rgb_image = CvBridge().imgmsg_to_cv2(msg, "rgb8")
        return self.rgb_image

    def get_intrinsic(self, camera='camera'):
        msg = rospy.wait_for_message(f"/{camera}/color/camera_info", CameraInfo)
        K_ros = np.array(msg.K).reshape((3, 3))
        intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        intrinsic[:3, :3] = K_ros
        return intrinsic

    def save_current_image(self, image_name, camera = 'camera'):
        """
        Save current state image
        """
        if self.rgb_image is None:
            self.rgb_image = self.get_current_image(camera)
        cv2.imwrite(image_name, self.rgb_image)

