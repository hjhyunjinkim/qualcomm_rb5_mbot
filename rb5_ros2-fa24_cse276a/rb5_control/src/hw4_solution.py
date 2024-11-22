#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt


# Grid dimensions
GRID_WIDTH, GRID_HEIGHT = 20, 20

# Define landmarks and obstacles
landmarks = [
    (1, 0, 3, 1), (0, 1, 1, 3),
    (1, GRID_HEIGHT - 1, 3, GRID_HEIGHT),
    (0, GRID_HEIGHT - 3, 1, GRID_HEIGHT - 1),
    (GRID_WIDTH - 3, 0, GRID_WIDTH - 1, 1),
    (GRID_WIDTH - 1, 1, GRID_WIDTH, 3),
    (GRID_WIDTH - 3, GRID_HEIGHT - 1, GRID_WIDTH - 1, GRID_HEIGHT),
    (GRID_WIDTH - 1, GRID_HEIGHT - 3, GRID_WIDTH, GRID_HEIGHT - 1),
]
obstacle = [(GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2 - 2, GRID_WIDTH // 2 + 2, GRID_HEIGHT // 2 + 2)]
start = (GRID_WIDTH - 10, 5)
goal = (10, GRID_HEIGHT - 10)


def create_grid(width, height, landmarks, obstacles):
    grid = np.zeros((width, height))
    for x1, y1, x2, y2 in landmarks:
        grid[x1:x2, y1:y2] = 1  # Landmarks marked with 1
    for x1, y1, x2, y2 in obstacles:
        grid[x1:x2, y1:y2] = -1  # Obstacles marked with -1
    return grid


def plot_grid(grid, path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, origin="lower", cmap="Greys", alpha=0.7)
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color="blue", linewidth=2, label="Path")
    plt.scatter(start[0], start[1], color="green", label="Start")
    plt.scatter(goal[0], goal[1], color="red", label="Goal")
    plt.legend()
    plt.grid()
    plt.show()


def a_star(grid, start, goal, risk_map=None, risk_weight=0):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while pq:
        current_cost, current = heapq.heappop(pq)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]

        for dx, dy in neighbors:
            next_cell = (current[0] + dx, current[1] + dy)
            if 0 <= next_cell[0] < grid.shape[0] and 0 <= next_cell[1] < grid.shape[1]:
                if grid[next_cell] == -1:
                    continue
                new_cost = cost_so_far[current] + 1
                if risk_map is not None:
                    new_cost += risk_weight * risk_map[next_cell]
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + np.linalg.norm(np.array(next_cell) - np.array(goal))
                    heapq.heappush(pq, (priority, next_cell))
                    came_from[next_cell] = current
    return None


class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__('PID_Controller_NodePub')
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

    def setTarget(self, target):
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(target)

    def getError(self, currentState, targetState):
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def update(self, currentState):
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I += self.Ki * e * self.timestep
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D
        self.lastError = e

        resultNorm = np.linalg.norm(result)
        if resultNorm > self.maximumValue:
            result = (result / resultNorm) * self.maximumValue
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
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.pose_updated = True

    def april_pose_callback(self, msg):
        self.pose_updated = False
        if len(msg.poses) < 1:
            return
        pose_camera = msg.poses[0]
        self.current_state = np.array([pose_camera.position.x, pose_camera.position.y, 0.0])
        self.pose_updated = True


def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0, 0.0, 1.0]])
    return np.dot(J, twist)


def genTwistMsg(desired_twist):
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0]
    twist_msg.linear.y = desired_twist[1]
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg


def follow_path(robot_state_estimator, pid, path):
    current_state = robot_state_estimator.current_state
    for wp in path:
        pid.setTarget(np.array([wp[0], wp[1], 0.0]))

        while True:
            update_value = pid.update(current_state)
            twist = coord(update_value, current_state)
            pid.publisher_.publish(genTwistMsg(twist))

            rclpy.spin_once(robot_state_estimator)
            current_state = robot_state_estimator.current_state
            error = np.linalg.norm(pid.getError(current_state, np.array([wp[0], wp[1], 0.0])))

            if error < 0.05:
                break

            time.sleep(0.05)

    pid.publisher_.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))


def main(args=None):
    rclpy.init(args=args)
    robot_state_estimator = RobotStateEstimator()

    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, landmarks, obstacle)
    risk_map = np.zeros_like(grid)

    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if grid[x, y] == 1:
                risk_map[x, y] = 10
            elif grid[x, y] == -1:
                risk_map[x, y] = 100

    path_time = a_star(grid, start, goal, risk_map=None)
    path_safety = a_star(grid, start, goal, risk_map=risk_map, risk_weight=1)

    plot_grid(grid, path_time)
    plot_grid(grid, path_safety)

    pid = PIDcontroller(0.065, 0.0, 0.05)
    selected_path = path_time
    follow_path(robot_state_estimator, pid, selected_path)

    robot_state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()