import numpy as np
import pybullet as p
import pybullet_data
import time
from scipy.optimize import minimize

class CHOMPPlanner:
    def __init__(self, robot_id, joint_indices, start_conf, goal_conf, obstacle_cost_weight=15.0, smoothness_cost_weight=10.0, eta=0.5):
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
        
    def initialize_trajectory(self):
        #  Initializes the trajectory using linear interpolation from start to goal configuration.
        return np.linspace(self.start_conf, self.goal_conf, self.timesteps)
    
    def calculate_smoothness_gradient(self):
        # Function to calculate the gradient of the smoothness cost for the entire trajectory.
        gradient = np.zeros_like(self.trajectory)
        # Iterate over the trajectory to calculate smoothness gradient for each point
        for t in range(1, self.timesteps - 1):
            gradient[t] = 2 * self.trajectory[t] - self.trajectory[t-1] - self.trajectory[t+1]
        return self.smoothness_cost_weight * gradient
    
    def calculate_collision_gradient(self):
        # Function to calculate the gradient of the collision cost for the entire trajectory.
        gradient = np.zeros_like(self.trajectory)
        for t in range(1, self.timesteps - 1):
            # Check the configuration at each timestep for collision
            grad = self.compute_collision_gradient(self.trajectory[t])
            gradient[t] = grad
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
        threshold_distance = 1  # Threshold distance to consider an obstacle "close"
        
        # Iterate over each link in the robot
        for link_index in self.joint_indices:
            # Check against each obstacle in the environment
            for obstacle_id in range(p.getNumBodies()):
                if obstacle_id == self.robot_id:
                    continue  # Skip the robot itself
                # Get closest points between the robot's link and the obstacle
                closest_points = p.getClosestPoints(bodyA=self.robot_id, bodyB=obstacle_id, distance=threshold_distance, linkIndexA=link_index)
                
                # Process each detected close point
                for point in closest_points:
                    distance = point[8]  # Distance to the obstacle
                    #normal_on_link = np.array(point[7])  # Normal pointing away from the obstacle
                    
                    # If an obstacle is detected within the threshold, calculate gradient
                    if distance < threshold_distance:
                        # print(f"Obstacle detected near link {link_index} of robot at distance {distance:.4f} from obstacle {obstacle_id}")

                        # Calculate the Jacobian for this link at the current configuration
                        jacobian = np.zeros(self.num_joints)
                        for j in range(self.num_joints):
                            delta_q = np.zeros(self.num_joints)
                            delta_q[j] = 1e-5
                            perturbed_conf = q + delta_q
                            for i, joint_index in enumerate(self.joint_indices):
                                p.resetJointState(self.robot_id, joint_index, perturbed_conf[i])
                            perturbed_state = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[0]
                            jacobian[j] = (np.linalg.norm(np.array(perturbed_state) - np.array(point[5])) - distance) / 1e-5
                        
                        # Compute the collision gradient contribution for this link
                        collision_gradient -= self.obstacle_cost_weight * jacobian * (threshold_distance - distance)
        
        return collision_gradient


    def inverse_kinematics(self, target_position):       
        # Function to calculate the inverse kinematics to find a joint configuration that reaches the target position.
        def objective_function(q):
            # Function to minimize to find joint configuration
            for i, joint_index in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_index, q[i])
            link_state = p.getLinkState(self.robot_id, self.joint_indices[-1], computeForwardKinematics=True)[0]
            return np.linalg.norm(np.array(link_state) - np.array(target_position))
        
        # Initial guess is the start configuration
        result = minimize(objective_function, self.start_conf, method='BFGS')
        return result.x

if __name__ == "__main__":
    # Connect to physics server
    p.connect(p.GUI)
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
    sphere_id = p.loadURDF("sphere2.urdf", target_position,useFixedBase=True, globalScaling=0.1) # Sphere as the target object
    
    planner = CHOMPPlanner(robot_id, joint_indices, start_conf, goal_conf)
    
    # Find joint configuration to reach target position
    goal_conf = planner.inverse_kinematics(target_position)
    planner.goal_conf = goal_conf
    print('Goal Confg:')
    print(planner.goal_conf)
    planner.trajectory = planner.initialize_trajectory()
    
    # Print initial trajectory
    print("Initial trajectory:")
    print(planner.trajectory)
    
    # Optimizing the trajectory (simplified loop for demonstration)
    for _ in range(100):  # Run optimization for 100 iterations
        smoothness_gradient = planner.calculate_smoothness_gradient()
        collision_gradient = planner.calculate_collision_gradient()
        total_gradient = smoothness_gradient + collision_gradient
        planner.trajectory -= planner.eta * total_gradient
        
        # Step the simulation for visualization
        for i, joint_index in enumerate(joint_indices):
            p.resetJointState(robot_id, joint_index, planner.trajectory[-1][i])
        p.stepSimulation()
        time.sleep(0.5) 
    
    # Print optimized trajectory
    print("Optimized trajectory:")
    print(planner.trajectory)
    
    # Disconnect from physics server
    p.disconnect()
