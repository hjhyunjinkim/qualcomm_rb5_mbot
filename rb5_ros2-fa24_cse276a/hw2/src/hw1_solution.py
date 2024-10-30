#!/usr/bin/env python3
import sys
#import rospy

import rclpy 
from rclpy.node import Node

from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
import math 

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
    # print(f"yaw_cam: {yaw_cam}")
    
    # Step 2: Compute the camera's yaw in the world frame
    theta_cam_world = yaw_cam + theta_tag_world
    theta_cam_world += -np.pi/2 if yaw_cam < 0 else +np.pi/2
    theta_cam_world = (theta_cam_world + np.pi) % (2 * np.pi) - np.pi
    
    d = math.sqrt(x_cam**2 + z_cam ** 2)
    x_cam_world = x_tag_world - np.cos(theta_cam_world) * d
    y_cam_world = y_tag_world - np.sin(theta_cam_world) * d

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
        self.maximumValue = 0.07
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        self.count = 0
        self.tag_locations = {'0': np.array([1.05, 0.0, np.pi]),
                              '1': np.array([0.5, -0.95, np.pi/2]),
                              '2': np.array([-0.5, 1.0, 0])}
        
        self.state = np.zeros(3)
        
        self.subscription = self.create_subscription(
            PoseStamped,
            '/april_poses',  # Adjust to the actual topic name
            self.apriltag_callback,
            10  # Queue size
        )
        self.subscription
        
    def apriltag_callback(self, msg):
        tag_id = msg.header.frame_id
        
        if tag_id not in self.tag_locations:
            print(f"tag {tag_id} not in tag locations")
            return
        
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
        print(f"tag {tag_id} -> pose: [{updated_pose[0]:.3f}, {updated_pose[1]:.3f}, {updated_pose[2]:.3f}]")

        if self.count == 0:
            self.state = updated_pose
            self.count = 5
            self.I = np.array([0.0,0.0,0.0]) 
            # self.lastError = np.array([0.0,0.0,0.0])
        else:
            self.count -= 1


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

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)

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

def coord(twist, state):
    J = np.array([[np.cos(state[2]), np.sin(state[2]), 0.0],
                  [-np.sin(state[2]), np.cos(state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)
    


if __name__ == "__main__":
    rclpy.init()

    import time
    #rospy.init_node("hw1")
    #pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
    
    # init pid controller
    pid = PIDcontroller(0.02,0.005,0.005)

    # init current state
    waypoint = np.array([[0.0,0.0,0.0], 
                        [0.5,0.0,0.0],
                        [0.5,1.0,np.pi],
                        [0.0,0.0,0.0]]) 

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        print(">>>>> move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(pid.state)
        # publish the twist
        pid.publisher_.publish(genTwistMsg(coord(update_value, pid.state)))
        #print(coord(update_value, state))
        time.sleep(0.05)
        # update the current state
        pid.state += update_value
        rclpy.spin_once(pid, timeout_sec=0.05)
            
        while(np.linalg.norm(pid.getError(pid.state, wp)) > 0.07): # check the error between current state and current way point
            print(f"current state: [{pid.state[0]:.3f}, {pid.state[1]:.3f}, {pid.state[2]:.3f}]")
            # calculate the current twist
            update_value = pid.update(pid.state)
            # publish the twist
            pid.publisher_.publish(genTwistMsg(coord(update_value, pid.state)))
            #print(coord(update_value, state))
            time.sleep(0.05)
            # update the current state
            pid.state += update_value
            rclpy.spin_once(pid, timeout_sec=0.05)
        pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
        time.sleep(2.0)

    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
