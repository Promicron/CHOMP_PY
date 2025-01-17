import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import random

class CHOMPPlanner:
    def __init__(self, robot_id, joint_indices, start_conf, goal_conf, obstacle_cost_weight=0.5, smoothness_cost_weight=0.5, eta=0.5):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.start_conf = np.array(start_conf)
        self.goal_conf = np.array(goal_conf)
        self.num_joints = len(joint_indices)
        self.timesteps = 10  # Number of timesteps for trajectory discretization.
        self.trajectory = self.initialize_trajectory()
        self.dt = 1.0 / self.timesteps
        self.obstacle_cost_weight = obstacle_cost_weight
        self.smoothness_cost_weight = smoothness_cost_weight
        self.eta = eta  # Step size for the optimization
        self.threshold_distance = 0.1  # Threshold distance to consider an obstacle "close"

    def initialize_trajectory(self):
        return np.linspace(self.start_conf, self.goal_conf, self.timesteps)

    def calculate_smoothness_gradient(self):
        gradient = np.zeros_like(self.trajectory)
        for t in range(1, self.timesteps - 1):
            gradient[t] = 2 * self.trajectory[t] - self.trajectory[t - 1] - self.trajectory[t + 1]
        return self.smoothness_cost_weight * gradient

    def calculate_collision_gradient(self):
        gradient = np.zeros_like(self.trajectory)
        for t in range(1, self.timesteps - 1):
            grad = self.compute_collision_gradient(self.trajectory[t])
            gradient[t] = grad
        return gradient

    def compute_collision_gradient(self, q):
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, q[i])

        collision_gradient = np.zeros(self.num_joints)
        for obstacle_id in range(p.getNumBodies()):
            if obstacle_id == self.robot_id:
                continue
            for link_index in range(p.getNumJoints(self.robot_id)):
                closest_points = p.getClosestPoints(
                    bodyA=self.robot_id, bodyB=obstacle_id, distance=self.threshold_distance, linkIndexA=link_index
                )
                if closest_points:
                    closest_point = closest_points[0]
                    distance = closest_point[8]
                    contact_normal = np.array(closest_point[7])

                    local_position = [0.0, 0.0, 0.0]
                    joint_positions = list(q)
                    joint_velocities = [0.0] * (self.num_joints + 2)
                    joint_accelerations = [0.0] * (self.num_joints + 2)

                    linear_jacobian, angular_jacobian = p.calculateJacobian(
                        self.robot_id, closest_point[3], local_position, joint_positions + [0.0, 0.0], joint_velocities, joint_accelerations
                    )

                    jacobian = np.array(linear_jacobian)[:, :self.num_joints]
                    contact_velocity = -contact_normal * (self.threshold_distance - distance)
                    pseudo_inverse_jacobian = np.linalg.pinv(jacobian)
                    collision_gradient += self.obstacle_cost_weight * np.dot(pseudo_inverse_jacobian, contact_velocity)

        return collision_gradient

    def inverse_kinematics(self, target_position):
        joint_positions = p.calculateInverseKinematics(self.robot_id, self.joint_indices[-1], target_position)
        return np.array(joint_positions[:self.num_joints])

if __name__ == "__main__":
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    obstacle_positions = [
        [0.5, 0.0, 0.2],
        [0.7, -0.3, 0.2],
        [0.6, 0.3, 0.2]
    ]
    obstacles = [
        p.loadURDF("cube.urdf", pos, useFixedBase=True, globalScaling=0.2) for pos in obstacle_positions
    ]

    for obs_id in obstacles:
        p.resetVisualShapeData(obs_id, -1, rgbaColor=[1, 0, 0, 1])

    joint_indices = [0, 1, 2, 3, 4, 5, 6]
    start_conf = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]
    target_positions = [
        [0.9, -0.5, 0.2],
        [0.8, 0.0, 0.2],
        [1.0, -0.284, 0.1],
    ]

    for _ in range(3):
        target_positions.append([random.uniform(0.1, 0.9), random.uniform(-0.5, 0.5), random.uniform(0.2, 0.4)])

    for target_position in target_positions:
        planner = CHOMPPlanner(robot_id, joint_indices, start_conf, goal_conf=np.zeros(len(joint_indices)))
        goal_conf = planner.inverse_kinematics(target_position)
        planner.goal_conf = goal_conf
        planner.trajectory = planner.initialize_trajectory()

        start_time = time.time()
        for iteration in range(100):
            smoothness_gradient = planner.calculate_smoothness_gradient()
            collision_gradient = planner.calculate_collision_gradient()
            total_gradient = smoothness_gradient + collision_gradient
            planner.trajectory -= planner.eta * total_gradient
        end_time = time.time()

        collision_detected = False
        for waypoint in planner.trajectory:
            for i, joint_index in enumerate(joint_indices):
                p.resetJointState(robot_id, joint_index, waypoint[i])
            if p.getContactPoints(bodyA=robot_id):
                collision_detected = True
                break

        print(f"Target Position: {target_position}")
        print(f"Execution Time: {end_time - start_time:.4f} seconds")
        print(f"Collision Detected: {'Yes' if collision_detected else 'No'}")
        print()

    p.disconnect()
