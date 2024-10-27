#!/usr/bin/env python3
import sys
#import rospy

import rclpy 
from rclpy.node import Node

from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseStamped

"""
The class of the pid controller.
"""
def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Convert a quaternion to yaw angle (in radians).
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def calculate_camera_pose(x_cam, z_cam, qx_cam, qy_cam, qz_cam, qw_cam,
                          x_tag_world, y_tag_world, theta_tag_world):
    # Step 1: Convert the detected quaternion to yaw
    yaw_cam = quaternion_to_yaw(qx_cam, qy_cam, qz_cam, qw_cam)

    # Step 2: Compute the camera's yaw in the world frame
    theta_cam_world = theta_tag_world - yaw_cam

    # Step 3: Rotate the detected AprilTag's position from the camera frame to the world frame
    x_cam_rotated = x_cam * np.cos(-yaw_cam) - z_cam * np.sin(-yaw_cam)
    z_cam_rotated = x_cam * np.sin(-yaw_cam) + z_cam * np.cos(-yaw_cam)

    # Step 4: Calculate the camera's world position
    x_cam_world = x_tag_world - x_cam_rotated
    y_cam_world = y_tag_world - z_cam_rotated

    return x_cam_world, y_cam_world, theta_cam_world

class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__('PID_Controller_NodePub')
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.1
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)
        self.current_state = np.array([0.0,0.0,0.0])
        
        self.tag_locations = {'0': np.array([-0.5, 0.0, np.pi])}
        self.subscription = self.create_subscription(
            PoseStamped,
            '/april_poses',  # Adjust to the actual topic name
            self.apriltag_callback,
            10  # Queue size
        )
        self.subscription
        
    def apriltag_callback(self, msg):
        print("callback activated")
        tag_id = msg.header.frame_id
        # Extract the position data from the PoseStamped message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Extract the quaternion orientation from the PoseStamped message
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w 
        
        updated_pose = calculate_camera_pose(x, z, qx, qy, qz, qw, self.tag_locations[tag_id][0], self.tag_locations[tag_id][1], self.tag_locations[tag_id][2])
        print(f"pose updated: {updated_pose}")
        self.current_state = updated_pose


    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the different between two states
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result 

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        self.update_camera_pose()
        e = self.getError(self.current_state, self.target)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        # scale down the twist if its norm is more than the maximum value. 
        resultNorm = np.linalg.norm(result)
        if(resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result

def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0] 
    twist_msg.linear.y = desired_twist[1] 
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg

def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)
    


if __name__ == "__main__":
    rclpy.init()

    import time
    #rospy.init_node("hw1")
    #pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)

    waypoint = np.array([[0.0,0.0,0.0], 
                         [-0.5,0.0,0.0],
                         [-0.5,0.5,np.pi/2.0],
                         [-1.0,0.5,0.0],
                         [-1.0,1.0,-np.pi/2.0],
                         [-0.5,0.5,-np.pi/4.0],
                         [0.0,0.0,0.0]]) 
    
    # init pid controller
    pid = PIDcontroller(0.02,0.005,0.005)

    # init current state

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update()
        # publish the twist
        pid.publisher_.publish(genTwistMsg(coord(update_value, pid.current_state)))
        #print(coord(update_value, current_state))
        time.sleep(0.05)
        # update the current state
        pid.current_state += update_value
        while(np.linalg.norm(pid.getError(pid.current_state, wp)) > 0.05): # check the error between current state and current way point
            # calculate the current twist
            update_value = pid.update()
            # publish the twist
            pid.publisher_.publish(genTwistMsg(coord(update_value, pid.current_state)))
            #print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            pid.current_state += update_value
    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

