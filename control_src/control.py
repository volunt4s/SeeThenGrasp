#!/usr/bin/env python
# Python 2/3 compatibility imports

from __future__ import print_function

import sys
import copy
import math
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import franka_gripper.msg
import actionlib

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class control(object):
    """control"""

    def __init__(self):
        super(control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface", anonymous=True)

        robot = moveit_commander.RobotCommander()
        # print(robot.get_current_state())

        scene = moveit_commander.PlanningSceneInterface()

        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        planning_frame = move_group.get_planning_frame()

        # We can also print the name of the end-effector link for this group:
        move_group.set_end_effector_link("panda_hand")
        eef_link = move_group.get_end_effector_link()

        group_names = robot.get_group_names()

        move_group.set_planner_id('RRTConnect')
        move_group.set_planning_time(5)

        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.fallback_joint_limits = [math.radians(90)] * 4 + [math.radians(90)] + [math.radians(180)] + [
            math.radians(350)]

        self.add_scene()

    
    def go_to_start(self):

        move_group = self.move_group

        print("[CONTROL] Execute go_to_start")

        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -0.785398163397
        joint_goal[2] = 0
        joint_goal[3] = -2.35619449019
        joint_goal[4] = 0
        joint_goal[5] = 1.57079632679
        joint_goal[6] = 0.785398163397

        move_group.go(joint_goal, wait=True)

        move_group.stop()

        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)


    def go_to_pose_goal(self, pose_goal, only_check_plan):

        move_group = self.move_group

        move_group.set_pose_target(pose_goal,self.eef_link)
        
        success, plan, planning_time, error_code = move_group.plan()
        if not only_check_plan:
            move_group.execute(plan, wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        print(f"Is plan success : {success}")

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)
    
    def cal_target_pose(self, pos =[0.307,0,0.59], rot = [1.,0.,0.,0.]):
        target_pose = geometry_msgs.msg.Pose()

        target_pose.position.x=float(pos[0])
        target_pose.position.y=float(pos[1])
        target_pose.position.z=float(pos[2])

        target_pose.orientation.x = float(rot[0])
        target_pose.orientation.y = float(rot[1])
        target_pose.orientation.z = float(rot[2])
        target_pose.orientation.w = float(rot[3])

        return target_pose

    def plan_and_execute_cartesian_path(self, pose_goal):

        move_group = self.move_group

        waypoints = []
        waypoints.append(pose_goal)

        
        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  
    
        move_group.execute(plan, wait=True)

        return plan, fraction

    def wait_for_state_update(
        self, box_is_known=False, box_is_attached=False, timeout=4
    ):
        box_name = self.box_name
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():

            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            is_known = box_name in scene.get_known_object_names()

            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL


    def grasp(self):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.GraspGoal()

        goal.width = 0.03
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        goal.speed = 0.1
        goal.force = 5

        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()

        # Prints out the result of executing the action
        return client.get_result()  # A GraspResult

    def release(self):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.GraspGoal()

        goal.width=0.08
        goal.speed = 0.1

        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()

        # Prints out the result of executing the action
        return client.get_result()  # A GraspResult
    
    def add_scene(self, timeout=4):
        scene = self.scene

        scene.clear()

        plane_pose = geometry_msgs.msg.PoseStamped()
        plane_pose.header.frame_id = "panda_link0"
        plane_pose.pose.orientation.w = 1.0
        plane_pose.pose.position.z = -0.01  
        plane_name = "plane_1"
        scene.add_plane(plane_name, plane_pose)

        plane_pose = geometry_msgs.msg.PoseStamped()
        plane_pose.header.frame_id = "panda_link0"
        plane_pose.pose.orientation.w = 1.0
        plane_pose.pose.position.y = 0.52
        plane_name = "plane_2"
        scene.add_plane(plane_name, plane_pose, normal=(0,1,0))

        self.plane_name = plane_name

    def is_crazy_plan(self, plan, max_rotation_per_joint):
        abs_rot_per_joint = self.rot_per_joint(plan)
        if (abs_rot_per_joint > max_rotation_per_joint).any():
            return True
        else:
            return False

    def rot_per_joint(self, plan, degrees=False):
        np_traj = np.array([p.positions for p in plan.joint_trajectory.points])
        if len(np_traj) == 0:
            raise ValueError
        np_traj_max_per_joint = np_traj.max(axis=0)
        np_traj_min_per_joint = np_traj.min(axis=0)
        ret = abs(np_traj_max_per_joint - np_traj_min_per_joint)
        if degrees:
            ret = [math.degrees(j) for j in ret]
        return ret

def main():
    try:
        test= control()
        test.go_to_start()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()