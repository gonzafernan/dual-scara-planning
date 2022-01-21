import gpath as gp
import numpy as np
import math
from zmqRemoteApi import RemoteAPIClient
import time
import matplotlib.pyplot as plt

path = gp.Path()
# st = np.array([-0.1, 0.3, 0.])
# gl = np.array([0.0, 0.35, 0.3])
# robot.ikine(pose[0, :])
# robot.ikine(pose[1, :])
# robot.ikine(pose[2, :])
# q, qd, qdd, p, pd, pdd = path.line_poly(start=st, goal=gl, mean_v=5)
# q, qd, qdd, p, pd, pdd = path.line(pose=pose, max_v=max_v, max_a=max_a)
# q, qd, qdd, p, pd, pdd = path.go_to_poly(start=st, goal=gl, mean_v=0.5)

# no resuelve la cinematica inversa
# plt.plot(q[:, 0], q[:, 1])
client = RemoteAPIClient()
sim = client.getObject('sim')
simIK = client.getObject('simIK')
# Get objects
base = sim.getObjectHandle('base')
tip1 = sim.getObjectHandle('tip1')
target1 = sim.getObjectHandle('target1')
joint11 = sim.getObjectHandle('q11')
joint12 = sim.getObjectHandle('q12')
tip2 = sim.getObjectHandle('tip2')
target2 = sim.getObjectHandle('target2')
joint21 = sim.getObjectHandle('q21')
joint22 = sim.getObjectHandle('q22')

# Set IK
ik_env = simIK.createEnvironment()
# Prepare ik group 1 (q11)
ik_group1 = simIK.createIkGroup(ik_env)
simIK.setIkGroupCalculation(ik_env, ik_group1,
                            simIK.method_damped_least_squares, 0.1, 100)
ik_element, sim_to_ik_object_map = simIK.addIkElementFromScene(
    ik_env, ik_group1, base, tip1, target1, simIK.constraint_position)
simIK.setJointMode(ik_env, sim_to_ik_object_map[joint11],
                   simIK.jointmode_passive)
simIK.setJointMode(ik_env, sim_to_ik_object_map[joint12], simIK.jointmode_ik)
# Second group (q22)
ik_group2 = simIK.createIkGroup(ik_env)
simIK.setIkGroupCalculation(ik_env, ik_group2,
                            simIK.method_damped_least_squares, 0.1, 100)
ik_element, sim_to_ik_object_map = simIK.addIkElementFromScene(
    ik_env, ik_group2, base, tip2, target2, simIK.constraint_position)
simIK.setJointMode(ik_env, sim_to_ik_object_map[joint21],
                   simIK.jointmode_passive)
simIK.setJointMode(ik_env, sim_to_ik_object_map[joint22], simIK.jointmode_ik)

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in asynchronous mode:
sim.startSimulation()
count = 0

sim.setJointPosition(joint11, 0)
sim.setJointPosition(joint21, 0)
j1 = sim.getJointPosition(joint11) + math.pi - 0.001
j2 = sim.getJointPosition(joint21)
start = path.robot.fkine(np.array([j1, j2, 0]))
# p1 = np.array([-0.4, 0.2, 0.])
# p2 = np.array([-0.2, 0.4, 0.])
# p3 = np.array([0.2, 0.4, 0.])
# p4 = np.array([0.4, 0.2, 0.])
# pose = np.block([[start], [p1], [p2], [p3], [p4]])
# max_v = np.array([1, 1, 1, 1])
# max_a = np.array([4, 4, 4, 4])
# q, qd, qdd, p, pd, pdd = path.go_to(goals=pose, max_v=max_v, max_a=max_a)

# q, qd, qdd, p, pd, pdd = path.line(pose=pose, max_v=1, max_a=4)
q, qd, qdd, p, pd, pdd = path.move_y_from_end(.3)
q[:, 0] = q[:, 0] % (np.pi * 2)
q1_cop = []
q2_cop = []
while count < max(q.shape):
    sim.setJointPosition(joint11, float(q[count, 0] - math.pi))
    sim.setJointPosition(joint21, float(q[count, 1]))
    simIK.applyIkEnvironmentToScene(ik_env, ik_group1)
    simIK.applyIkEnvironmentToScene(ik_env, ik_group2)
    q1_cop.append(sim.getJointPosition(joint11) + math.pi)
    q2_cop.append(sim.getJointPosition(joint21))
    count += 1
sim.stopSimulation()
# If you need to make sure we really stopped:
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

plt.figure()
plt.plot(q[:, 0], 'r')
plt.plot(q[:, 1], 'g')
plt.plot(q1_cop, 'k--')
plt.plot(q2_cop, 'k--')
plt.show()
