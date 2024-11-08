#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseArray
import numpy as np
import math
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from rclpy.time import Duration
import math
from math import copysign, fabs, sqrt, pi, sin, cos, asin, acos, atan2, exp, log


EPSILON = 1e-0 # threshold for computing quaternion from rotation matrix


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


def quaternion_to_rotation_matrix(q):
        '''
        Convert a unit quaternion to a rotation matrix.

        This function is only for internal usage. Please use the following 
        instead when it is needed to do this conversion.
        
            R = q.to_rotation_matrix()

        '''
        x, y, z, w = q
        w2 = w * w
        x2 = x * x
        y2 = y * y
        z2 = z * z
        
        # Check the norm of the quaternion! It meanst be a unit quaternion!
        assert fabs(w2 + x2 + y2 + z2 - 1) < 1e-6
        
        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z
        
        R = np.asarray((
            ( w2 + x2 - y2 - z2,   xy - wz,             xz + wy           ),
            ( xy + wz,             w2 - x2 + y2 - z2,   yz - wx           ),
            ( xz - wy,             yz + wx,             w2 - x2 - y2 + z2 )
            ))
        
        return R


def rotation_matrix_to_quaternion(R):
        '''
        Convert a rotation matrix to a unit quaternion.
        
        This uses the Shepperd’s method for numerical stabilty.

        This function is only for internal usage. Please use the following 
        instead when it is needed to do this conversion.
        
            q = Quaternion.construct_from_rotation_matrix(R)

        '''
        
        # The rotation matrix meanst be orthonormal
        assert np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), 
                           atol=EPSILON)
    
        # Check the determinant of R! It meanst be 1.
        assert math.isclose(np.linalg.det(R), 1, abs_tol=EPSILON)
        
        w2 = (1 + R[0, 0] + R[1, 1] + R[2, 2])
        x2 = (1 + R[0, 0] - R[1, 1] - R[2, 2])
        y2 = (1 - R[0, 0] + R[1, 1] - R[2, 2])
        z2 = (1 - R[0, 0] - R[1, 1] + R[2, 2])
            
        yz = (R[1, 2] + R[2, 1])
        xz = (R[2, 0] + R[0, 2])
        xy = (R[0, 1] + R[1, 0])
    
        wx = (R[2, 1] - R[1, 2])
        wy = (R[0, 2] - R[2, 0])
        wz = (R[1, 0] - R[0, 1])
                    
            
        if R[2, 2] < 0:
          
            if R[0, 0] > R[1, 1]:
            
                x = sqrt(x2)
                w = wx / x
                y = xy / x
                z = xz / x
            
            else:
                 
                y = sqrt(y2)
                w = wy / y
                x = xy / y
                z = yz / y
    
        else:
              
            if R[0, 0] < -R[1, 1]:
                 
                z = sqrt(z2)
                w = wz / z
                x = xz / z
                y = yz / z
            
            else:
                 
                w = sqrt(w2)
                x = wx / w
                y = wy / w
                z = wz / w
        
        w = w * 0.5
        x = x * 0.5
        y = y * 0.5
        z = z * 0.5
        
        return x, y, z, w

def combine_transformations(R1, t1, R2, t2):
    """
    Combine two transformations given by rotation matrices and translation vectors.
    
    Parameters:
    R1 (numpy.ndarray): First 3x3 rotation matrix.
    t1 (numpy.ndarray): First translation vector, shape (3,).
    R2 (numpy.ndarray): Second 3x3 rotation matrix.
    t2 (numpy.ndarray): Second translation vector, shape (3,).
    
    Returns:
    R_combined (numpy.ndarray): Combined 3x3 rotation matrix.
    t_combined (numpy.ndarray): Combined translation vector, shape (3,).
    """
    # Ensure the input matrices and vectors have correct shapes
    assert R1.shape == (3, 3), "R1 meanst be a 3x3 matrix"
    assert t1.shape == (3,), "t1 meanst be a 3-element vector"
    assert R2.shape == (3, 3), "R2 meanst be a 3x3 matrix"
    assert t2.shape == (3,), "t2 meanst be a 3-element vector"
    
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = t1

    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2
    
    return np.dot(T1, T2)

def initialize_robot_landmarks_covariance(n_landmarks):
    # Dimensions for mean vector and covariance matrix
    dim_mean = 3 + 2 * n_landmarks
    mean_vector = np.zeros((dim_mean, 1))
    covariance_matrix = np.zeros((dim_mean, dim_mean))
    
    # Example initialization for robot's pose mean and covariance
    # Robot pose (x, y, theta) mean
    mean_vector[0:3, 0] = [0.0, 0.0, 0.0]  # Initial pose at origin (can be set differently)
    
    # Example values for robot's pose covariance
    covariance_matrix[0:3, 0:3] = np.diag([0.1, 0.1, 0.05])  # Variances for (x, y, theta)
    
    # Initialize means for landmarks
    for i in range(n_landmarks):
        mean_vector[3 + 2 * i, 0] = 0.0  # x-coordinate of the i-th landmark
        mean_vector[3 + 2 * i + 1, 0] = 0.0  # y-coordinate of the i-th landmark

    # Example initialization for landmark covariance
    for i in range(n_landmarks):
        idx = 3 + 2 * i
        covariance_matrix[idx:idx + 2, idx:idx + 2] = np.diag([0.2, 0.2])  # Variance for each landmark position (x, y)
    
    return mean_vector, covariance_matrix

def add_new_landmark(mean_vector, covariance_matrix):
    new_mean_vector = np.vstack((mean_vector, np.zeros((2, 1))))
    dim = covariance_matrix.shape[0]
    new_covariance_matrix = np.zeros((dim + 2, dim + 2))
    
    new_covariance_matrix[:dim, :dim] = covariance_matrix
    # Initialize new landmark's covariance (can be set to default values like [0.2, 0.2] for (x, y))
    new_covariance_matrix[dim:dim+2, dim:dim+2] = np.diag([0.2, 0.2])
    
    return new_mean_vector, new_covariance_matrix

# TODO: get Range-Bearing Observation
def ekf_slam_correction(mean, covariance, z, landmark_dict):
    """
    EKF SLAM Correction step for observed landmarks.

    Parameters:
    - mean: Current state estimate (robot pose and landmark positions).
    - covariance: Current covariance estimate.
    - z: List of observed features, each as [apriltag_id, r, phi] format.
    - landmark_dict: Indices of all known landmarks with their positions.

    Returns:
    - Updated mean and covariance.
    """
    # Measurement noise covariance
    sigma_r, sigma_r = np.std(np.array(list(zip(*z))), axis=1)
    Qt = np.diag([sigma_r**2, sigma_phi**2])
    
    for observation in enumerate(z):
        apriltag_id, r, phi = observation
        
        # Check if landmark is seen for the first time
        if apriltag_id not in landmark_dict.keys():
            # Initialize landmark position
            mean_jx = mean[0] + r * np.cos(phi + mean[2])
            mean_jy = mean[1] + r * np.sin(phi + mean[2])
            landmark_dict[apriltag_id] = np.array([mean_jx, mean_jy])
            
        delta = landmark_dict[apriltag_id] - mean[:2]
        q = np.dot(delta, delta)
        z_hat = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - mean[2]])

        # Compute Jacobians Fx_j and H_i_t
        Fx_j = np.zeros((5, len(mean)))
        Fx_j[:3, :3] = np.eye(3)
        Fx_j[3, 2 * j + 1] = 1
        Fx_j[4, 2 * j + 2] = 1

        H_i_t = (1 / q) * np.array([
            [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
            [delta[1], -delta[0], -q, -delta[1], delta[0]]
        ]).dot(Fx_j)
        
        # Compute Kalman gain
        S_i_t = H_i_t.dot(covariance).dot(H_i_t.T) + Qt
        K_i_t = covariance.dot(H_i_t.T).dot(np.linalg.inv(S_i_t))

        # Update mean and covariance
        mean = mean + K_i_t.dot(z[i] - z_hat)
        covariance = (np.eye(len(mean)) - K_i_t.dot(H_i_t)).dot(covariance)
    
    return mean, covariance

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
        self.maximeanmValue = 0.02
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

    def setTarget(self, target):
        """
        Set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(target)

    def getError(self, currentState, targetState):
        """
        Return the difference between two states.
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def setMaximeanmeanpdate(self, mv):
        """
        Set maximeanm velocity for stability.
        """
        self.maximeanmValue = mv

    def update(self, currentState):
        """
        Calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I += self.Ki * e * self.timestep
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D
        self.lastError = e

        # Scale down the twist if its norm is more than the maximeanm value
        resultNorm = np.linalg.norm(result)
        if resultNorm > self.maximeanmValue:
            result = (result / resultNorm) * self.maximeanmValue
            self.I = 0.0

        return result


class RobotStateEstimator(Node):
    def __init__(self):
        super().__init__('robot_state_estimator')
        self.subscription = self.create_subscription(
            PoseArray,
            '/april_poses',
            self.april_pose_callback,
            10)
        self.subscription
        self.br = TransformBroadcaster(self)
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.pose_updated = True

        # if you are not using tf2 service, define your aptiltag poses (wrt map frame) here, z value is omitted to ensure 2D transformation
        self.apriltag_world_poses = {
            'marker_0': (0.67, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5), # (x,y,z, qual_x, qual_y, qual_z, qual_w)
            'marker_1': (-0.5, 1.2, 0.0, -0.5, -0.5, 0.5, 0.5),
            # 'marker_2': (-0.5, 2.0, 0.0, -0.5, -0.5, 0.5, 0.5),
        }


    def april_pose_callback(self, msg):
        # Log the number of poses in the PoseArray
        self.pose_updated = False
        self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses")
        if len(msg.poses) < 1:
            return
        pose_ids = msg.header.frame_id.split(',')[:-1]
        
        # we will only use one landmark at a time in homework 2. in homework 3, all landmarks should be considered.
        tag_id = pose_ids[0]
        if tag_id not in self.apriltag_world_poses.keys():
            return
        pose_camera_apriltag = msg.poses[0]   # syntax: pose_ReferenceFrame_TargetFrame
        trans_camera_apriltag = np.array([
            pose_camera_apriltag.position.x,
            pose_camera_apriltag.position.y,
            pose_camera_apriltag.position.z,
        ])
        quat_camera_apriltag = np.array([
            pose_camera_apriltag.orientation.x,
            pose_camera_apriltag.orientation.y,
            pose_camera_apriltag.orientation.z,
            pose_camera_apriltag.orientation.w,
        ])
        rot_camera_apriltag = quaternion_to_rotation_matrix(quat_camera_apriltag)

        # project to 2d
        trans_camera_apriltag_2d = trans_camera_apriltag
        trans_camera_apriltag_2d[1] = 0.0

        rot_camera_apriltag_2d = np.array([
            [rot_camera_apriltag[0,0], 0.0, rot_camera_apriltag[2,0]],
            [0.0, 1.0, 0.0],
            [rot_camera_apriltag[2,0], 0.0, rot_camera_apriltag[0,0]],
        ])

        rot_apriltag_camera_2d = rot_camera_apriltag_2d.T
        trans_apriltag_camera_2d = -np.dot(rot_apriltag_camera_2d, trans_camera_apriltag_2d)
        
        # retrieve static transformation of apriltags wrt map
        rot_map_apriltag = quaternion_to_rotation_matrix(self.apriltag_world_poses[tag_id][3:])
        trans_map_apriltag = np.asarray(self.apriltag_world_poses[tag_id][:3])

        # apply transfomation
        T_map_camera = combine_transformations(
            rot_map_apriltag, trans_map_apriltag,
            rot_apriltag_camera_2d, trans_apriltag_camera_2d,
        )
        rot_map_camera, trans_map_camera = T_map_camera[:3, :3], T_map_camera[:3, 3]
        angle = math.atan2(rot_map_camera[1][2], rot_map_camera[0][2])

        self.current_state = np.array([trans_map_camera[0], trans_map_camera[1], angle])
        self.pose_updated = True


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

def ekf_slam_prediction(mean, covariance, u_t, R_t, delta_t):
    """
    EKF SLAM Prediction step for robot state and landmarks.

    Parameters:
    - mean: Current state estimate (robot pose and landmark positions).
    - covariance: Current covariance estimate.
    - u_t: Control input [v_t, omega_t] (velocity and angular velocity).
    - R_t: Process noise covariance matrix.
    - delta_t: Time step duration.

    Returns:
    - Predicted mean and covariance.
    """
    # Extract robot pose and control inputs
    v_t = u_t[0]
    omega_t = u_t[1]
    theta = mean[2, 0]  # Robot's orientation in the mean vector

    # Fx matrix to map control input into the larger state space
    Fx = np.zeros((3, mean.shape[0]))
    Fx[:3, :3] = np.eye(3)

    # Calculate predicted mean (mu_bar_t)
    if omega_t != 0:
        delta_mean = np.array([
            [-v_t / omega_t * np.sin(theta) + v_t / omega_t * np.sin(theta + omega_t * delta_t)],
            [v_t / omega_t * np.cos(theta) - v_t / omega_t * np.cos(theta + omega_t * delta_t)],
            [omega_t * delta_t]
        ])
    else:
        # Linear approximation when omega_t is zero
        delta_mean = np.array([
            [v_t * np.cos(theta) * delta_t],
            [v_t * np.sin(theta) * delta_t],
            [0]
        ])

    mean_bar = mean + Fx.T @ delta_mean

    # Calculate G_t matrix (Jacobian of motion model with respect to the state)
    G_t = np.eye(mean.shape[0])
    if omega_t != 0:
        G_t[:3, :3] = np.eye(3) + Fx.T @ np.array([
            [0, 0, -v_t / omega_t * np.cos(theta) + v_t / omega_t * np.cos(theta + omega_t * delta_t)],
            [0, 0, -v_t / omega_t * np.sin(theta) + v_t / omega_t * np.sin(theta + omega_t * delta_t)],
            [0, 0, 0]
        ]) @ Fx
    else:
        # Linearized model for G_t when omega_t is zero
        G_t[:3, :3] = np.eye(3) + Fx.T @ np.array([
            [0, 0, -v_t * np.sin(theta) * delta_t],
            [0, 0, v_t * np.cos(theta) * delta_t],
            [0, 0, 0]
        ]) @ Fx

    # Calculate predicted covariance (Sigma_bar_t)
    covariance_bar = G_t @ covariance @ G_t.T + Fx.T @ R_t @ Fx

    return mean_bar, covariance_bar


def main(args=None):
    rclpy.init(args=args)
    robot_state_estimator = RobotStateEstimator()
    waypoint = np.array([[0.0,0.0,0.0], 
                         [0.5,0.0,0.0],
                         [0.5,1.0,np.pi],
                         [0.0,0.0,0.0]])

    # init pid controller
    pid = PIDcontroller(0.1,0.005,0.005)
    current_state = robot_state_estimator.current_state
    
    # init vectors for KF - the number of lms are unknown 
    mean, covariance = initialize_robot_landmarks_covariance(n_landmarks=0)
    # init landmark list
    landmark_ids = []

    for wp in waypoint:
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(current_state)
        # publish the twist
        pid.publisher_.publish(genTwistMsg(coord(update_value, current_state)))
        #print(coord(update_value, current_state))
        time.sleep(0.05)
        # update the current state
        
        # 1: STATE PREDICTION & MEASUREMENT PREDICTION
        v_x, v_y, omega = coord(update_value, current_state)
        # current_state += update_value # TODO: check if needed to update the current state here
        # mean[:3] = current_state
        
        u_t = np.array([np.sqrt(v_x**2 + v_y**2), omega])
        R_t = np.diag([0.1, 0.1, 0.1])  # Process noise covariance matrix
        delta_t = pid.timestep
        
        # 2: MEASUREMENT PREDICTION
        mean, covariance = ekf_slam_prediction(mean, covariance, u_t, R_t, delta_t)
        
        # TODO: get z from observations as Range-Bearing observation (z: [apriltag_id, r, phi])
        rclpy.spin_once(robot_state_estimator)
        found_state, estimated_state = robot_state_estimator.pose_updated, robot_state_estimator.current_state
        if found_state: # if the tag is detected, we can use it to update current state.
            current_state = estimated_state #
        
        # 3,4 OBTAIN MEASUREMENT / ASSOCIATE
        mean, covariance = ekf_slam_correction(mean, covariance, z, landmark_dict)
        
        # 5: STATE UPDATE
        current_state = mean[:3]
        
        
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # check the error between current state and current way point
            # calculate the current twist
            update_value = pid.update(current_state)
            # publish the twist
            pid.publisher_.publish(genTwistMsg(coord(update_value, current_state)))
            #print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            current_state += update_value
            rclpy.spin_once(robot_state_estimator)
            found_state, estimated_state = robot_state_estimator.pose_updated, robot_state_estimator.current_state
            if found_state: # if the tag is detected, we can use it to update current state.
                current_state = estimated_state
    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    robot_state_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

