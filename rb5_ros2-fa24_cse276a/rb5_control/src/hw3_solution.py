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
    # Measurement noise covariance TODO: tune these values
    sigma_r = 0.64
    sigma_phi = 0.65
    Qt = np.diag([sigma_r**2, sigma_phi**2])
    mean_tx = mean[:3].squeeze()
    
    for observation in z:
        apriltag_id, r, phi = observation
        z_i = np.array([r, phi])
        
        if r > 2.0:
            continue
        
        # Check if landmark is seen for the first time
        if apriltag_id not in landmark_dict.keys():
            mean_jx = mean_tx[0] + r * np.cos(phi + mean_tx[2])
            mean_jy = mean_tx[1] + r * np.sin(phi + mean_tx[2])
            landmark_dict[apriltag_id] = {'idx': len(landmark_dict.keys())}
            # extend mean and covariance matrix
            mean = np.vstack((mean, [[mean_jx], [mean_jy]]))
            new_size = mean.shape[0]
            new_cov = np.zeros((new_size, new_size))
            new_cov[:new_size - 2, :new_size - 2] = covariance

            new_cov[-2, -2] = 0.2  
            new_cov[-1, -1] = 0.2 
            covariance = new_cov 
        

        j = landmark_dict[apriltag_id]['idx']
        mean_jx = mean[3 + 2 *j, 0]
        mean_jy = mean[3 + 2 *j + 1, 0]
        
        # print(f"Landmark ID: {apriltag_id} ({landmark_dict[apriltag_id]['idx']}), r: {r:.2f}, phi: {phi:.2f}, pos: ({mean_jx:.2f}, {mean_jy:.2f})")
        
        delta = ([mean_jx, mean_jy] - mean_tx[:2])
        q = np.dot(delta.T, delta)
        z_hat = np.array([np.sqrt(q), normalize_angle(np.arctan2(delta[1], delta[0]) - mean_tx[2])])

        # Compute Jacobians Fx_j and H_i_t
        Fx_j = np.zeros((5, len(mean)))
        Fx_j[:3, :3] = np.eye(3)
        Fx_j[3, 2 * (j+1) + 1] = 1
        Fx_j[4, 2 * (j+1) + 2] = 1  
        
        H_i_t = (1 / q) * np.array([
            [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
            [delta[1], -delta[0], -q, -delta[1], delta[0]]
        ]).dot(Fx_j)
                
        # Compute Kalman gain
        S_i_t = H_i_t.dot(covariance).dot(H_i_t.T) + Qt
        K_i_t = covariance.dot(H_i_t.T).dot(np.linalg.inv(S_i_t))
        
        # Update mean and covariance
        mean = mean + K_i_t.dot(z_i - z_hat).reshape(-1, 1)
        mean[2] = normalize_angle(mean[2])
        covariance = (np.eye(len(mean)) - K_i_t.dot(H_i_t)).dot(covariance)
        
    return mean, covariance

def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

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
        self.z = np.array([], dtype=np.float64)
        self.landmark_detected = False
        self.theta_r = 0.0

    def april_pose_callback(self, msg):
        # Log the number of poses in the PoseArray
        self.landmark_detected = False
        self.z = np.array([], dtype=np.float64)
        # self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses")
        if len(msg.poses) < 1:
            return
        pose_ids = msg.header.frame_id.split(',')[:-1]    
        z_list = []

        for i, pose_camera_apriltag in enumerate(msg.poses):
            tag_id = int(pose_ids[i].replace('marker_', ''))

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

            # Project to 2D (ignore y-axis)
            trans_camera_apriltag_2d = trans_camera_apriltag
            trans_camera_apriltag_2d[1] = 0.0  
            
            rot_camera_apriltag_2d = np.array([
            [rot_camera_apriltag[0,0], 0.0, rot_camera_apriltag[2,0]],
            [0.0, 1.0, 0.0],
            [rot_camera_apriltag[2,0], 0.0, rot_camera_apriltag[0,0]],
            ])

            # yaw = np.arctan2(rot_camera_apriltag[2, 0], rot_camera_apriltag[0, 0])

            theta_r = self.theta_r
            dx = trans_camera_apriltag_2d[2] * np.cos(theta_r) + trans_camera_apriltag_2d[0] * np.sin(theta_r)
            dy = trans_camera_apriltag_2d[2] * np.sin(theta_r) - trans_camera_apriltag_2d[0] * np.cos(theta_r)

            # Calculate range (r) and bearing (phi) in the world frame
            r = np.hypot(dx, dy)  # Equivalent to np.sqrt(dx**2 + dy**2)
            phi = normalize_angle(normalize_angle(math.atan2(dy, dx)) - theta_r)

            # Append the result as [tag_id, r, phi] to the list
            z_list.append([tag_id, r, phi])

        # Convert the list to a NumPy array of shape (n_landmarks, 3)
        self.z = np.array(z_list)
        
        if len(self.z.shape) == 1:
            self.z = np.expand_dims(self.z, axis=0)
        self.landmark_detected = True


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
    theta = mean[2, 0]  # Robot's orientation in the mean vector
    v_x, v_y, omega_t = u_t
    v_t = v_x * np.cos(theta) + v_y * np.sin(theta)

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
        G_t += Fx.T @ np.array([
            [0, 0, -v_t / omega_t * np.cos(theta) + v_t / omega_t * np.cos(theta + omega_t * delta_t)],
            [0, 0, -v_t / omega_t * np.sin(theta) + v_t / omega_t * np.sin(theta + omega_t * delta_t)],
            [0, 0, 0]
        ]) @ Fx
    else:
        # Linearized model for G_t when omega_t is zero
        G_t += Fx.T @ np.array([
            [0, 0, -v_t * np.sin(theta) * delta_t],
            [0, 0, v_t * np.cos(theta) * delta_t],
            [0, 0, 0]
        ]) @ Fx

    # Calculate predicted covariance (Sigma_bar_t)
    covariance_bar = G_t @ covariance @ G_t.T + Fx.T @ R_t @ Fx

    return mean_bar, covariance_bar

def log_state(idx, timestamp, state, mean, covariance, landamrk_dict):
    lm_str = ''
    for key, value in landamrk_dict.items():
        idx = value['idx']
        mean_per_lm = mean[3 + 2 * idx:3 + 2 * idx + 2]
        cov_per_lm = covariance[3 + 2 * idx:3 + 2 * idx + 2, 3 + 2 * idx:3 + 2 * idx + 2]
        lm_str += f"\t|[{int(key)}] [{mean_per_lm[0]}, {mean_per_lm[1]}] / [{cov_per_lm[0, 0]}, {cov_per_lm[0, 1]}, {cov_per_lm[1, 0]} {cov_per_lm[1, 1]}|\n"
    log_entry = f"{idx}, {timestamp:.3f}, {state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}" + lm_str + "\n"
    return log_entry

def generate_octagon_trajectory(e):
    trajectory = [
        (0, 0, 0),     
        (e, 0, 0), 
        (e + e / np.sqrt(2), e / np.sqrt(2), np.pi / 4),
        (e + e / np.sqrt(2), e + e / np.sqrt(2), np.pi / 2),
        (e, e + 2 * e / np.sqrt(2), 3 * np.pi / 4),
        (0, e + 2 * e / np.sqrt(2), np.pi),
        (- e / np.sqrt(2), e + e / np.sqrt(2), -3 * np.pi / 4),
        (- e / np.sqrt(2), e / np.sqrt(2), -np.pi / 2),
        (0, 0, 0)
    ]
    
    return trajectory

def get_scale_factor(vel):
    r = 0.025 # radius of the wheel
    lx = 0.055 # half of the distance between front wheel and back wheel
    ly = 0.07 # half of the distance between left wheel and right wheel
    calibration = 120.0
    angular_calibration = 120.0
        
    desired_twist = np.array([
        [calibration * vel.linear.x],
        [calibration * -vel.linear.y],
        [angular_calibration * vel.angular.z]
    ])

    # calculate the jacobian matrix
    jacobian_matrix = np.array([[1, -1, -(lx + ly)],
                                    [1, 1,  (lx + ly)],
                                    [1, 1, -(lx + ly)],
                                    [1, -1, (lx + ly)]]) / r
    # calculate the desired wheel velocity
    result = np.dot(jacobian_matrix, desired_twist)
    scale_factor = 50 / np.min(np.abs(result))
    
    if scale_factor < 1:
        scale_factor = 1.2
        
    return scale_factor


def main(args=None):
    rclpy.init(args=args)
    robot_state_estimator = RobotStateEstimator()
    log_file = open("./robot_state_log.txt", "w")
    start_time = time.time()
    
    
    n = 0.7
    square_waypoints = np.array([
        [0.0, 0.0, 0.0],
        [n, 0.0, 0.0],
        [n, n, np.pi/2],
        [0.0, n, np.pi],
        [0.0, 0.0, -np.pi/2]
    ])


    octagon_waypoints = generate_octagon_trajectory(e=0.5)
    # octagon_waypoints = octagon_waypoints + octagon_waypoints + octagon_waypoints # repeat the trajectory 3 times
    # waypoint = square_waypoints
    waypoint = octagon_waypoints
    # init pid controller
    pid = PIDcontroller(0.062,0,0.0005)
    # pid = PIDcontroller(0.1,0.005,0.005)
    current_state = np.array([0.0,0.0,0.0])
    
    # init vectors for KF - the number of lms are unknown 
    mean, covariance = initialize_robot_landmarks_covariance(n_landmarks=0)
    R_t = np.diag([0.2, 0.2453, 0.1699])  # Process noise covariance matrix
    # R_t = np.diag([0.3, 0.3, 0.5])  # Process noise covariance matrix
    landmark_dict = {}
    delta_t = pid.timestep
    
    for i, wp in enumerate(waypoint):
        patience = 15  
        stuck_counter = 0 
        last_state = np.array(current_state)
        last_velocity = np.array([0.0, 0.0, 0.0])  
        
        print(">>> move to way point", wp, "\n")
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(current_state)
        # publish the twist
        velocity = coord(update_value, current_state)
        pid.publisher_.publish(genTwistMsg(velocity))
        time.sleep(delta_t / 2)
        
        last_velocity = np.array(velocity)
        last_state = np.array(current_state)

        # 1: STATE PREDICTION & MEASUREMENT PREDICTION
        u_t = np.array(velocity)
        
        # 2: MEASUREMENT PREDICTION
        mean, covariance = ekf_slam_prediction(mean, covariance, u_t, R_t, delta_t)
        robot_state_estimator.theta_r = mean[2].item()
        rclpy.spin_once(robot_state_estimator)
        z = robot_state_estimator.z
        # 3,4 OBTAIN MEASUREMENT / ASSOCIATE
        mean, covariance = ekf_slam_correction(mean, covariance, z, landmark_dict)
            
        # 5: STATE UPDATE
        current_state = mean[:3].squeeze()
        log_file.write(log_state(i, time.time() - start_time, current_state, mean, covariance, landmark_dict))
        # print(f'current state {i}: {current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}')
        
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # check the error between current state and current way point)
            # calculate the current twist
            update_value = pid.update(current_state)
            # publish the twist
            velocity = coord(update_value, current_state)
            state_diff = np.linalg.norm(pid.getError(current_state, last_state))
            vel_diff = np.linalg.norm(velocity - last_velocity)
            print(f'{state_diff:.6f} {vel_diff:.6f}')

            # if state_diff < 0.007 and vel_diff < 0.001 and len(z) > 0:
            #     stuck_counter += 1
            #     if stuck_counter >= patience:
            #         velocity *= get_scale_factor(genTwistMsg(velocity))
            #         stuck_counter = -20  # Apply larger velocity for 5 iterations
            #         print("!!! Applying larger velocity to escape stuck state.")
            # elif state_diff < 0.007 and stuck_counter < 0:
            #     velocity *= get_scale_factor(genTwistMsg(velocity))
            #     stuck_counter += 1
            # else:
            #     stuck_counter = 0 

            last_state = np.array(current_state)
            last_velocity = np.array(velocity)
                            
            pid.publisher_.publish(genTwistMsg(velocity))
            #print(coord(update_value, current_state))
            time.sleep(delta_t / 2)
            
            # 1: STATE PREDICTION
            u_t = np.array(velocity)
                      
            # 2: MEASUREMENT PREDICTION
            mean, covariance = ekf_slam_prediction(mean, covariance, u_t, R_t, delta_t)
            robot_state_estimator.theta_r = mean[2].item()

            rclpy.spin_once(robot_state_estimator)
            z = robot_state_estimator.z
            # 3,4 OBTAIN MEASUREMENT / ASSOCIATE
            mean, covariance = ekf_slam_correction(mean, covariance, z, landmark_dict)

            # 5: STATE UPDATE
            current_state = mean[:3].squeeze()
            log_file.write(log_state(i, time.time() - start_time, current_state, mean, covariance, landmark_dict))
            print(f'current state {i}: {current_state[0]:.4f}, {current_state[1]:.4f}, {current_state[2]:.4f}')    # stop the car and exit
        
        pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
        time.sleep(3.0)

    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    robot_state_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

