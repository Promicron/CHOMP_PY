import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p

class CHOMPVisualizer:
    def __init__(self, robot_id, joint_indices):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        
    def get_end_effector_position(self, joint_config):
        """Get end-effector position for a given joint configuration."""
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_config[i])
        link_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
        return link_state[0]  # Return position
    
    def visualize_trajectory_3d(self, trajectory, obstacles=None, save_path=None):
        """
        Visualize the end-effector trajectory in 3D space with obstacles.
        
        Args:
            trajectory: numpy array of shape (timesteps, num_joints)
            obstacles: list of obstacle positions [(x, y, z), ...]
            save_path: path to save the figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get end-effector positions for each configuration
        ee_positions = []
        for config in trajectory:
            pos = self.get_end_effector_position(config)
            ee_positions.append(pos)
        ee_positions = np.array(ee_positions)
        
        # Plot trajectory
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                'b-', linewidth=2, label='End-effector trajectory')
        
        # Plot start and end points
        ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
                  color='red', s=100, label='Start')
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], 
                  color='green', s=100, label='End')
        
        # Plot obstacles if provided
        if obstacles:
            for obs_pos in obstacles:
                ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], 
                          color='purple', s=100, alpha=0.5, marker='s')
        
        # Customize plot
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-effector Trajectory in 3D Space')
        ax.legend()
        ax.grid(True)
        
        plt.show()
    
    def visualize_joint_trajectories(self, trajectory, save_path=None):
        """
        Visualize the evolution of joint angles over time.
        
        Args:
            trajectory: numpy array of shape (timesteps, num_joints)
            save_path: path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        timesteps = np.arange(len(trajectory))
        
        for joint_idx in range(len(self.joint_indices)):
            ax.plot(timesteps, trajectory[:, joint_idx], 
                   label=f'Joint {self.joint_indices[joint_idx]}',
                   marker='o', markersize=4)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Joint Angle (rad)')
        ax.set_title('Joint Angles Evolution')
        ax.legend()
        ax.grid(True)
        
        plt.show()
    
    def visualize_optimization_metrics(self, objective_values, smoothness_values, 
                                    collision_values, save_path=None):
        """
        Visualize the optimization metrics over iterations.
        
        Args:
            objective_values: list of objective function values
            smoothness_values: list of smoothness cost values
            collision_values: list of collision cost values
            save_path: path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        iterations = np.arange(len(objective_values))
        
        # Plot total objective value
        ax1.semilogy(iterations, objective_values, 'b-', label='Total Objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost (log scale)')
        ax1.set_title('Convergence of Total Objective')
        ax1.legend()
        ax1.grid(True)
        
        # Plot individual costs
        ax2.semilogy(iterations, smoothness_values, 'g-', label='Smoothness Cost')
        ax2.semilogy(iterations, collision_values, 'r-', label='Collision Cost')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cost (log scale)')
        ax2.set_title('Evolution of Individual Costs')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
    
        plt.show()
