from fivebar import FiveBar
import gpath as gp
import numpy as np
import math
from zmqRemoteApi import RemoteAPIClient
import time
import matplotlib.pyplot as plt

l1 = 0.25
l2 = 0.38
b = 0.0
five = FiveBar(np.array([-b, l1, l2]), np.array([b, l1, l2]))
path = gp.Path(five)

client = RemoteAPIClient()
sim = client.getObject('sim')
# Get objects
base = sim.getObjectHandle('base')
end = sim.getObjectHandle('end')
joint11 = sim.getObjectHandle('q11')
joint21 = sim.getObjectHandle('q21')

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in asynchronous mode:
sim.startSimulation()
j1 = sim.getJointPosition(joint11) + math.pi
j2 = sim.getJointPosition(joint21)
# end_pose = sim.getObjectPosition(end, base)
# start = path.robot.ikine(np.array([end_pose[0], end_pose[1], 0]))
start = path.robot.fkine(np.array([j1, j2, 0]))
p1 = np.array([-0.4, 0.2, 0.])
p2 = np.array([-0.2, 0.4, 0.])
p3 = np.array([0.2, 0.4, 0.])
p4 = np.array([0.4, 0.2, 0.])
p5 = np.array([0.4, 0.4, 0.])
p6 = np.array([0.0, 0.6, 0.])
p7 = np.array([-0.3, 0.3, 0.])
pose = np.block([[start], [p1], [p2], [p3], [p4], [p5], [p6], [p7]])
max_v = np.array([2, 2, 2, 2, 2, 2, 2, 2])
max_a = np.array([4, 4, 4, 4, 4, 4, 4, 4])
# q, qd, qdd, p, pd, pdd = path.go_to(goals=pose, max_v=max_v, max_a=max_a)
q, qd, qdd, p, pd, pdd = path.line(pose=pose, max_v=max_v, max_a=max_a)
# q, qd, qdd, p, pd, pdd = path.move_y_from_end(.3)

n = max(q.shape)
q1_cop = np.zeros(n)
q2_cop = np.zeros(n)
p1_cop = np.zeros(n)
p2_cop = np.zeros(n)

idx = 0
while idx < n:
    sim.setJointPosition(joint11, float(q[idx, 0] - math.pi))
    sim.setJointPosition(joint21, float(q[idx, 1]))
    p1_cop[idx] = sim.getObjectPosition(end, base)[0]
    p2_cop[idx] = sim.getObjectPosition(end, base)[1]
    q1_cop[idx] = sim.getJointPosition(joint11) + math.pi
    q2_cop[idx] = sim.getJointPosition(joint21)
    idx += 1
sim.stopSimulation()
#  We need make sure we really stopped:
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)
#  Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

# Show results
plt.figure(1)
plt.plot(np.rad2deg(q[:, 0]), 'r', label='$q_1$')
plt.plot(np.rad2deg(q[:, 1]), 'g', label='$q_2$')
plt.plot(np.rad2deg(q1_cop), 'k--', label='$q_{1}c$')
plt.plot(np.rad2deg(q2_cop), 'k--', label='$q_{2}c$')
plt.ylabel('Joints [°] ')
plt.xlabel('Iterations')
plt.title("Joints Position")
plt.legend()
plt.figure(2)
plt.plot(p[:, 0], 'r', label='x')
plt.plot(p[:, 1], 'g', label='y')
plt.plot(p1_cop, 'k--', label='xc')
plt.plot(p2_cop, 'k--', label='yc')
plt.ylabel('Position [m] ')
plt.xlabel('Iterations')
plt.title("Cartesian Position")
plt.legend()
plt.figure(3)
plt.plot(p[:, 0], p[:, 1], 'r', label='desired')
plt.plot(p1_cop, p2_cop, 'k--', label='coppeliasim')
plt.xlabel('End position X [m] ')
plt.ylabel('End position Y [m]')
plt.title("End effector position")
plt.legend()
plt.show()

# CoppeliaSim kinematic configuration
# Add IK pluggin
# simIK = client.getObject('simIK')
# Set IK
# ik_env = simIK.createEnvironment()
# # Prepare ik group 1 (q11)
# ik_group1 = simIK.createIkGroup(ik_env)
# simIK.setIkGroupCalculation(ik_env, ik_group1,
#                             simIK.method_damped_least_squares, 0.1, 100)
# ik_element, sim_to_ik_object_map = simIK.addIkElementFromScene(
#     ik_env, ik_group1, base, tip1, target1, simIK.constraint_position)
# simIK.setJointMode(ik_env, sim_to_ik_object_map[joint11],
#                    simIK.jointmode_passive)
# simIK.setJointMode(ik_env, sim_to_ik_object_map[joint12], simIK.jointmode_ik)
# # Second group (q22)
# ik_group2 = simIK.createIkGroup(ik_env)
# simIK.setIkGroupCalculation(ik_env, ik_group2,
#                             simIK.method_damped_least_squares, 0.1, 100)
# ik_element, sim_to_ik_object_map = simIK.addIkElementFromScene(
#     ik_env, ik_group2, base, tip2, target2, simIK.constraint_position)
# simIK.setJointMode(ik_env, sim_to_ik_object_map[joint21],
#                    simIK.jointmode_passive)
# simIK.setJointMode(ik_env, sim_to_ik_object_map[joint22], simIK.jointmode_ik)
