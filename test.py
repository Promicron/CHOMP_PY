import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Function to calculate Jacobian
def calculate_jacobian(robot_id, link_index, joint_indices, local_position=[0, 0, 0]):
    return p.calculateJacobian(robot_id, link_index, local_position,
                               [p.getJointState(robot_id, idx)[0] for idx in joint_indices],
                               [0.0] * len(joint_indices),  
                               [0.0] * len(joint_indices))


link_index = 5  # End-effector link index
joint_indices = [0, 1, 2, 3, 4, 5, 6]
print(p.getNumJoints(robot_id))
print("Joint positions:", [p.getJointState(robot_id, idx)[0] for idx in joint_indices])
print("Velocities:", [0.0] * len(joint_indices))
print("Accelerations:", [0.0] * len(joint_indices))

jacobian_linear, jacobian_angular = calculate_jacobian(robot_id, link_index, joint_indices)
print("Linear Jacobian:\n", jacobian_linear)
print("Angular Jacobian:\n", jacobian_angular)

p.disconnect()
