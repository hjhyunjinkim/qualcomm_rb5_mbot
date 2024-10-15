#!/usr/bin/env python3
import numpy as np
from math import atan2, sqrt
from megapi import MegaPi  # Ensure the MegaPi library is installed
import time

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
        self.bot.motorRun(self.mfl, vfl)
        self.bot.motorRun(self.mfr, vfr)
        self.bot.motorRun(self.mbl, vbl)
        self.bot.motorRun(self.mbr, vbr)

    def carStop(self):
        if self.verbose:
            print("CAR STOP")
        self.setFourMotors(0, 0, 0, 0)

class OpenLoopController:
    def __init__(self, mpi_ctrl, wheel_radius=0.05, lx=0.2, ly=0.2):
        """
        Initialize the OpenLoopController with the MegaPi controller and kinematic parameters.
        """
        self.mpi_ctrl = mpi_ctrl  # MegaPiController instance
        self.wheel_radius = wheel_radius
        self.lx = lx
        self.ly = ly
        self.K = self._calculate_kinematic_matrix()

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

    def compute_velocity(self, current_wp, next_wp, min_velocity=0.1, max_velocity=0.5):
        """
        Compute the linear and angular velocities needed to move to the next waypoint.
        """
        dx = next_wp[0] - current_wp[0]
        dy = next_wp[1] - current_wp[1]
        dtheta = next_wp[2] - current_wp[2]

        distance = sqrt(dx**2 + dy**2)
        angle_to_goal = atan2(dy, dx)

        # Clamp velocity between min and max limits
        vx = max(min_velocity, min(distance, max_velocity)) * np.cos(angle_to_goal)
        vy = max(min_velocity, min(distance, max_velocity)) * np.sin(angle_to_goal)
        wz = dtheta / max(distance, 0.01)

        return vx, vy, wz

    def navigate_to_waypoints(self, waypoints):
        """
        Navigate through a series of waypoints using open-loop control.
        """

        for i in range(len(waypoints) - 1):
            current_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            vx, vy, wz = self.compute_velocity(current_wp, next_wp)
            wheel_vels = self.calculate_wheel_velocities(vx, vy, wz)
            wheel_vels = [int(vel) for vel in wheel_vels]

            # Set motor speeds using MegaPiController
            self.mpi_ctrl.setFourMotors(
                vfl=wheel_vels[0], vfr=wheel_vels[1],
                vbl=wheel_vels[2], vbr=wheel_vels[3]
            )

            time.sleep(2)  # Adjust based on robot's behavior

        self.mpi_ctrl.carStop()  # Stop the robot

if __name__ == '__main__':
    waypoints = read_waypoints('/root/hw1/waypoints_ds.txt')

    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    controller = OpenLoopController(mpi_ctrl)

    controller.navigate_to_waypoints(waypoints)