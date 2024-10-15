#!/usr/bin/env python3
import numpy as np
from math import atan2, sqrt
from megapi import MegaPi  # Ensure the MegaPi library is installed
import time
import math

# Constants for motor ports
MFR = 2  # Motor front right
MBL = 3  # Motor back left
MBR = 10  # Motor back right
MFL = 11  # Motor front left

def read_waypoints(file_path):
    waypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, theta = map(float, line.strip().split(','))
            waypoints.append([x, y, theta])
    return waypoints

class MegaPiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR
        self.mbl = MBL
        self.mbr = MBR
        self.mfl = MFL

    def printConfiguration(self):
        print(f'MegaPiController: Communication Port: {self.port}')
        print(f'Motor Ports: MFR: {MFR}, MBL: {MBL}, MBR: {MBR}, MFL: {MFL}')

    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0):
        if self.verbose:
            print(f'Set Motors: vfl={int(vfl)}, vfr={int(vfr)}, vbl={int(vbl)}, vbr={int(vbr)}')
        self.bot.motorRun(self.mfl, -vfl)
        self.bot.motorRun(self.mfr, vfr)
        self.bot.motorRun(self.mbl, -vbl)
        self.bot.motorRun(self.mbr, vbr)

    def carStop(self):
        if self.verbose:
            print("CAR STOP")
        self.setFourMotors(0, 0, 0, 0)

class OpenLoopController:
    def __init__(self, mpi_ctrl, wheel_radius=0.025):
        """
        Initialize the OpenLoopController with the MegaPi controller and kinematic parameters.
        """
        self.mpi_ctrl = mpi_ctrl  # MegaPiController instance
        self.wheel_radius = wheel_radius
        self.lx = wheel_radius * 2
        self.ly = wheel_radius * 2
        self.K = self._calculate_kinematic_matrix()
        
        self.x = 0
        self.y = 0
        self.theta = 0
        self.dt = 1.0  # Time step for control

    #def _calculate_kinematic_matrix(self):
    #    """
    #    Calculate the kinematic model matrix based on the robot's dimensions.
    #    """
    #    return (self.wheel_radius / 4) * np.array([
    #        [1, 1, 1, 1],
    #        [-1, 1, 1, -1],
    #        [-(1 / (self.lx + self.ly)), 1 / (self.lx + self.ly),
    #         -(1 / (self.lx + self.ly)), 1 / (self.lx + self.ly)]
    #    ])

    def _calculate_kinematic_matrix(self):
        """
        Calculate the kinematic model matrix to convert [vx, vy, wz] to [ω1, ω2, ω3, ω4].
        """
        r = self.wheel_radius  # Wheel radius
        lx = self.lx
        ly = self.ly

        # Adjusting the matrix to align with direct kinematics
        return (4 / r) * np.array([
            [1, -1, -(lx + ly)],  # ω1
            [1, 1, (lx + ly)],    # ω2
            [1, 1, -(lx + ly)],   # ω3
            [1, -1, (lx + ly)]    # ω4
        ])

    def calculate_wheel_velocities(self, vx, vy, wz):
        """
        Compute individual wheel velocities using the kinematic model.
        """
        V = np.array([vx, vy, wz]).reshape(3, 1)  # Ensure V is a 3x1 column vector
        wheel_vels = np.dot(self.K, V).flatten()  # Flatten to get a 1D array of wheel velocities

        return wheel_vels

    def compute_velocity(self, next_wp, min_velocity=0.0, max_velocity=0.5):
        """
        Compute the linear and angular velocities needed to move to the next waypoint.
        """
        dx = next_wp[0] - self.x
        dy = next_wp[1] - self.y
        dtheta = next_wp[2] - self.theta

        distance = sqrt(dx**2 + dy**2)
        angle_to_goal = atan2(dy, dx)
        
        print("angle:", angle_to_goal)

        # Clamp velocity between min and max limits
        vx = max(min_velocity, min(distance, max_velocity)) * np.cos(angle_to_goal)
        vy = max(min_velocity, min(distance, max_velocity)) * np.sin(angle_to_goal)
        # wz = dtheta / max(distance, 0.01)
        wz = dtheta
        
        print("compute_velocity: ", vx, vy, wz)

        return vx, vy, wz
    
    def update_position(self, vx, vy, wz):
        self.x += vx * self.dt
        self.y += vy * self.dt
        self.theta += wz * self.dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi

    def navigate_to_waypoints(self, waypoints):
        """
        Navigate through a series of waypoints using open-loop control.
        """
        try:
            idx = 0
            distance_threshold = 0.1
            angle_threshold = math.pi / 5.0  # 35 degrees
            
            count = 0

            while idx < len(waypoints):
                print(f"\n({idx}) Current Position:", self.x, self.y, self.theta)
                current_wp = waypoints[idx]
                print("Target: ", current_wp)
                target_x, target_y, target_theta = current_wp

                vx, vy, wz = self.compute_velocity(current_wp)
                self.update_position(vx, vy, wz)
                
                wheel_vels = self.calculate_wheel_velocities(vx, vy, wz)
                wheel_vels = [int(vel) for vel in wheel_vels]

                # Set motor speeds using MegaPiController
                self.mpi_ctrl.setFourMotors(
                    vfl=wheel_vels[0], vfr=wheel_vels[1],
                    vbl=wheel_vels[2], vbr=wheel_vels[3]
                )
                
                distance_to_waypoint = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
                remaining_angle = (target_theta - self.theta + np.pi) % (2 * np.pi) - np.pi    

                if distance_to_waypoint < distance_threshold and abs(remaining_angle) < angle_threshold: 
                    idx += 1  

                time.sleep(self.dt)  # Adjust based on robot's behavior
                count += 1
                if count == 30: # Just in case the loop never stops
                    break
        except KeyboardInterrupt:
            print("\nCtrl+C detected! Stopping the car...")
            
        finally:
            time.sleep(1)
            self.mpi_ctrl.carStop()  # Stop the robot
            print("Car stopped and closed properly.")            


if __name__ == '__main__':
    waypoints = read_waypoints('/root/hw1/waypoints_ds.txt')

    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    controller = OpenLoopController(mpi_ctrl)

    controller.navigate_to_waypoints(waypoints)