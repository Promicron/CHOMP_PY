import numpy as np
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt 

class CHOMPPlanner:
    def __init__(self, robot_id, joint_indices, start_conf, goal_conf, obstacle_cost_weight=0.5, smoothness_cost_weight=0.5, eta=0.5):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.start_conf = np.array(start_conf)
        self.goal_conf = np.array(goal_conf)
        self.num_joints = len(joint_indices)
        self.timesteps = 10  # Number of timesteps for trajectory discretization.
        self.trajectory = self.initialize_trajectory()  # Initializes the trajectory with a linear interpolation.
        self.dt = 1.0 / self.timesteps # Time difference between timesteps.
        self.obstacle_cost_weight = obstacle_cost_weight
        self.smoothness_cost_weight = smoothness_cost_weight
        self.eta = eta # Step size for the optimization
        self.threshold_distance = 0.1  # Threshold distance to consider an obstacle "close"
        
    def initialize_trajectory(self):
        #  Initializes the trajectory using linear interpolation from start to goal configuration.
        return np.linspace(self.start_conf, self.goal_conf, self.timesteps)
    
    def calculate_smoothness_gradient(self):
        # Function to calculate the gradient of the smoothness cost for the entire trajectory.
        gradient = np.zeros_like(self.trajectory)
        # Iterate over the trajectory to calculate smoothness gradient for each point
        for t in range(1, self.timesteps - 1):
            gradient[t] = 2 * self.trajectory[t] - self.trajectory[t-1] - self.trajectory[t+1]
        # print(f"Smoothness Gradient: {gradient}")  # Debugging output
        return self.smoothness_cost_weight * gradient
    
    def calculate_collision_gradient(self):
        # Function to calculate the gradient of the collision cost for the entire trajectory.
        gradient = np.zeros_like(self.trajectory)
        for t in range(1, self.timesteps - 1):
            # Check the configuration at each waypoint for collision
            grad = self.compute_collision_gradient(self.trajectory[t])
            gradient[t] = grad
        # print(f"Collision Gradient: {gradient}") 
        return gradient
    
    def compute_collision_gradient(self, q):
        """
        Compute the collision gradient for a given configuration `q`.
        
        Parameters:
        - q: Joint configuration to evaluate
        
        Returns:
        - A numpy array representing the collision gradient for the given configuration.
        """

        # Move the robot to the configuration `q`
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_index, q[i])

        # Compute the collision gradient based on closest points
        collision_gradient = np.zeros(self.num_joints)
        
        # Find the closest obstacle to the robot
        closest_obstacle_id = None
        closest_distance = float('inf')
        for obstacle_id in range(p.getNumBodies()):
            if obstacle_id == self.robot_id:
                continue  # Skip the robot itself
            for link_index in range(p.getNumJoints(self.robot_id)):
                closest_points = p.getClosestPoints(bodyA=self.robot_id, bodyB=obstacle_id, distance=self.threshold_distance, linkIndexA=link_index)
                if closest_points:
                    if closest_points[0][8] < closest_distance:
                        closest_distance = closest_points[0][8]
                        closest_obstacle_id = obstacle_id
                        closest_point = closest_points[0]
        
        # If an obstacle is detected within the threshold, calculate gradient
        if closest_obstacle_id is not None:
            distance = closest_point[8]  # Distance to the obstacle
            contact_normal = np.array(closest_point[7])  # Normal pointing away from the obstacle
            
            # Calculate the Jacobian for this link at the current configuration
            local_position = [0.0, 0.0, 0.0] 
            joint_positions = list(q)
            joint_velocities = [0.0] * (self.num_joints + 2)  # Include finger joints
            joint_accelerations = [0.0] * (self.num_joints + 2)  # Include finger joints
            
            # Ensure that the number of joint positions, velocities, and accelerations match the number of DoF
            if len(joint_positions) == self.num_joints and len(joint_velocities) == self.num_joints + 2 and len(joint_accelerations) == self.num_joints + 2:
                linear_jacobian, angular_jacobian = p.calculateJacobian(self.robot_id, closest_point[3], local_position, joint_positions + [0.0, 0.0], joint_velocities, joint_accelerations)
                jacobian = np.array(linear_jacobian)[:, :self.num_joints]  # Remove columns corresponding to the finger joints
                
                # Compute the collision gradient contribution for this link
                contact_velocity = -contact_normal * (self.threshold_distance - distance)
                pseudo_inverse_jacobian = np.linalg.pinv(jacobian)
                collision_gradient += self.obstacle_cost_weight * np.dot(pseudo_inverse_jacobian, contact_velocity)
            else:
                print("Error: Mismatch in number of DoF and joint parameters.")
        
        return collision_gradient

    def inverse_kinematics(self, target_position):       
        joint_positions = p.calculateInverseKinematics(self.robot_id, self.joint_indices[-1], target_position)
        return np.array(joint_positions[:self.num_joints])

if __name__ == "__main__":
    # Connect to physics server
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load environment and robot
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    
    # Add obstacles to the environment
    obstacle1_id = p.loadURDF("cube.urdf", [0.5, 0.0, 0.2], useFixedBase=True, globalScaling=0.2)
    obstacle2_id = p.loadURDF("cube.urdf", [0.7, -0.3, 0.2], useFixedBase=True, globalScaling=0.2)
    obstacle3_id = p.loadURDF("cube.urdf", [0.6, 0.3, 0.2], useFixedBase=True, globalScaling=0.1)
    
    p.resetVisualShapeData(obstacle2_id, -1, rgbaColor=[1, 0, 0, 1])
    
    joint_indices = [0, 1, 2, 3, 4, 5, 6]  # Assuming 7 DOF robot arm
    start_conf = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]  # Initial joint configuration
    goal_conf = [1.0, 0.5, -0.5, -1.0, 0.5, -0.5, 1.0] # A dummy joint configuration that's reassigned later

    # Target positions to test
    target_position = [0.9, -0.5, 0.2]
    # target_position = [0.8, 0, 0.2] 
    # target_position = [0.55, 0.4, 0.25]
    sphere_id = p.loadURDF("sphere2.urdf", target_position, useFixedBase=True, globalScaling=0.1) # Sphere as the target object
    
    planner = CHOMPPlanner(robot_id, joint_indices, start_conf, goal_conf)
    
    # Find joint configuration to reach target position
    goal_conf = planner.inverse_kinematics(target_position)
    planner.goal_conf = goal_conf
    print('Goal Confg:')
    print(planner.goal_conf)
    planner.trajectory = planner.initialize_trajectory()

    # Lists to store the values for plotting
    objective_values = []
    smoothness_values = []
    collision_values = []
    
    # Print initial trajectory
    print("Initial trajectory:")
    print(planner.trajectory)
    
    # Optimizing the trajectory (simplified loop for demonstration)
    for iteration in range(100):  # Run optimization for 100 iterations
        smoothness_gradient = planner.calculate_smoothness_gradient()
        collision_gradient = planner.calculate_collision_gradient()
        total_gradient = smoothness_gradient + collision_gradient
        planner.trajectory -= planner.eta * total_gradient
        
        # Print objective function value and gradients for debugging
        objective_value = np.sum(np.square(total_gradient))
        smoothness_value = np.sum(np.square(smoothness_gradient))
        collision_value = np.sum(np.square(collision_gradient))
        print(f"Iteration {iteration}:")
        print(f"Objective Value = {objective_value}")
        print(f"Smoothness Value = {smoothness_value}")
        print(f"Collision Value = {collision_value}")
        # print(f"Total Gradient: {total_gradient}")
        
        # Store the values for plotting
        objective_values.append(objective_value)
        smoothness_values.append(smoothness_value)
        collision_values.append(collision_value)

        # Step the simulation for visualization
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_index, planner.trajectory[-1][i])
        # p.stepSimulation()
        time.sleep(0.5) 
    
    # Print optimized trajectory
    print("Optimized trajectory:")
    print(planner.trajectory)

    # Display the final trajectory
    for waypoint in planner.trajectory:
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_index, waypoint[i])
        p.stepSimulation()
        time.sleep(0.5)  # Optional delay to visualize each step


    # Plot the objective, smoothness, and collision values over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(objective_values, label='Objective Value')
    plt.plot(smoothness_values, label='Smoothness Value')
    plt.plot(collision_values, label='Collision Value')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Objective, Smoothness, and Collision Values over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Disconnect from physics server
    p.disconnect()
