import gpath as gp
import fivebar as fb
import numpy as np
import math
from zmqRemoteApi import RemoteAPIClient
import time
import matplotlib.pyplot as plt

d1 = np.array([-0.125, 0.2, 0.2])
d2 = np.array([0.125, 0.2, 0.2])
robot = fb.FiveBar(d1, d2)
path = gp.Path(robot)
st = np.array([-0.1, 0.3, 0.])
gl = np.array([0.0, 0.35, 0.3])
pose = np.array([[-.1, 0.3, 0.], [0.1, 0.3, 0.], [.1, 0.3, 0.]])
robot.ikine(pose[0, :])
robot.ikine(pose[1, :])
robot.ikine(pose[2, :])
max_v = np.array([1, 1])
max_a = np.array([4, 4])
# q, qd, qdd, p, pd, pdd = path.line_poly(start=st, goal=gl, mean_v=5)
q, qd, qdd, p, pd, pdd = path.line(pose=pose, max_v=max_v, max_a=max_a)
# q, qd, qdd, p, pd, pdd = path.go_to_poly(start=st, goal=gl, mean_v=0.5)
# q, qd, qdd, p, pd, pdd = path.go_to(goals=pose, max_v=max_v, max_a=max_a)

# no resuelve la cinematica inversa
# plt.plot(q[:, 0], q[:, 1])
plt.plot(p[:, 0], p[:, 1])
plt.show()
client = RemoteAPIClient()
sim = client.getObject('sim')

joint1 = sim.getObjectHandle('joint_11')
joint2 = sim.getObjectHandle('joint_21')
# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in asynchronous mode:
sim.startSimulation()
count = 0
# while (t := sim.getSimulationTime()) < 100:
while count < max(q.shape):
    # if count % 2 == 0:
    sim.setJointTargetPosition(joint1, float(q[count, 0] - math.pi / 2))
    sim.setJointTargetPosition(joint2, float(q[count, 1] - math.pi / 2))
    # else:
    #     sim.setJointTargetPosition(joint1, 0 * math.pi / 4)
    #     sim.setJointTargetPosition(joint2, 0 * math.pi / 4)
    count += 10
    time.sleep(0.01)
sim.stopSimulation()
# If you need to make sure we really stopped:
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)
