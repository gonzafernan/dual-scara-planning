from FiveBar import FiveBar
import Path as gp
import numpy as np
import math
from zmqRemoteApi import RemoteAPIClient
import time
import matplotlib.pyplot as plt

l1 = 0.25
l2 = 0.38
l22 = 0.3878
b = 0.0
five = FiveBar(np.array([-b, l1, l2]), np.array([b, l1, l22]))
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
# LINE 1
start = path.robot.fkine(np.array([j1, j2, 0]))
point1 = np.array([0., 0.3, 0.])
point2 = np.array([-0.3, 0.3, 0.])
line1 = np.block([[start], [point1], [point2]])
q1, qd1, qdd1, p1, pd1, pdd1 = path.line(pose=line1,
                                         max_v=[0.1, 0.1],
                                         max_a=[0.2, 0.2])
# ARC 1
point3 = np.array([-0.319, 0.304, 0.])
point4 = np.array([-0.35, 0.35, 0.])
q2, qd2, qdd2, p2, pd2, pdd2 = path.arc(start=point2,
                                        way_point=point3,
                                        goal=point4,
                                        max_vel=0.5,
                                        max_acc=1)

# LINE 2
point5 = np.array([-0.35, 0.4, 0.])
line2 = np.block([[point4], [point5]])
q3, qd3, qdd3, p3, pd3, pdd3 = path.line(pose=line2, max_v=0.1, max_a=0.2)

# ARC 2
point6 = np.array([-0.346, 0.419, 0.])
point7 = np.array([-0.3, 0.45, 0.])
q4, qd4, qdd4, p4, pd4, pdd4 = path.arc(start=point5,
                                        way_point=point6,
                                        goal=point7,
                                        max_vel=0.5,
                                        max_acc=1)
# LINE 3
point8 = np.array([0., 0.45, 0.])
line3 = np.block([[point7], [point8]])
q5, qd5, qdd5, p5, pd5, pdd5 = path.line(pose=line3, max_v=0.1, max_a=0.2)

# CIRCLE
c = np.array([0, 0.375, 0])
q6, qd6, qdd6, p6, pd6, pdd6 = path.circle(start=point8,
                                           center=c,
                                           max_vel=0.1,
                                           max_acc=1)

# LINE 4
point9 = np.array([0.3, 0.45, 0.])
line4 = np.block([[point8], [point9]])
q7, qd7, qdd7, p7, pd7, pdd7 = path.line(pose=line4, max_v=0.1, max_a=0.2)

# ARC 3
point10 = np.array([0.319, 0.446, 0.])
point11 = np.array([0.35, 0.4, 0.])
q8, qd8, qdd8, p8, pd8, pdd8 = path.arc(start=point9,
                                        way_point=point10,
                                        goal=point11,
                                        max_vel=0.5,
                                        max_acc=1)

# LINE 5
point12 = np.array([0.35, 0.35, 0.])
line5 = np.block([[point11], [point12]])
q9, qd9, qdd9, p9, pd9, pdd9 = path.line(pose=line5, max_v=0.1, max_a=0.2)

# ARC 4
point13 = np.array([0.346, 0.331, 0.])
point14 = np.array([0.3, 0.3, 0.])
q10, qd10, qdd10, p10, pd10, pdd10 = path.arc(start=point12,
                                              way_point=point13,
                                              goal=point14,
                                              max_vel=0.5,
                                              max_acc=1)

# line 6
line6 = np.block([[point14], [point1]])
q11, qd11, qdd11, p11, pd11, pdd11 = path.line(pose=line6, max_v=0.1, max_a=0.2)
p = np.block([[p1], [p2], [p3], [p4], [p5], [p6], [p7], [p8], [p9], [p10],
              [p11]])
q = np.block([[q1], [q2], [q3], [q4], [q5], [q6], [q7], [q8], [q9], [q10],
              [q11]])
n = max(q.shape)
q1_cop = np.zeros(n)
q2_cop = np.zeros(n)
p1_cop = np.zeros(n)
p2_cop = np.zeros(n)

idx = 0
while idx < n:
    sim.setJointPosition(joint11, float(q[idx, 0] - math.pi))
    sim.setJointPosition(joint21, float(q[idx, 1]))
    end_position = sim.getObjectPosition(end, base)
    p1_cop[idx] = end_position[0]
    p2_cop[idx] = end_position[1]
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
plt.grid(True)
plt.figure(2)
plt.plot(p[:, 0], 'r', label='x')
plt.plot(p[:, 1], 'g', label='y')
plt.plot(p1_cop, 'k--', label='xc')
plt.plot(p2_cop, 'k--', label='yc')
plt.ylabel('Position [m] ')
plt.xlabel('Iterations')
plt.title("Cartesian Position")
plt.legend()
plt.grid(True)
plt.figure(3)
plt.plot(p[:, 0], p[:, 1], 'r', label='desired')
plt.plot(p1_cop, p2_cop, 'k--', label='coppeliasim')
plt.xlabel('End position X [m] ')
plt.ylabel('End position Y [m]')
plt.title("End effector position")
plt.legend()
plt.grid(True)
plt.show()

print("The averrage error")
pose_x_diff = (p[:, 0] - p1_cop)**2
pose_y_diff = (p[:, 1] - p2_cop)**2
pose_error = np.sqrt(pose_x_diff + pose_y_diff)
q1_diff = q[:, 0] - q1_cop
q2_diff = q[:, 1] - q2_cop

plt.figure(4)
plt.plot(pose_error)
plt.grid(True)
plt.show()

print("Pose mean error:", pose_error.mean())
print("Pose max error:", max(pose_error))
print("Pose min error:", min(pose_error))
print("Q1 mean error:", q1_diff.mean())
print("Q1 max error:", max(q1_diff))
print("Q1 min error:", min(q1_diff))
print("Q2 mean error:", q2_diff.mean())
print("Q2 max error:", max(q2_diff))
print("Q2 min error:", min(q2_diff))

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
