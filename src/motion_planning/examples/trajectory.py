import gpath
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath

# Create a Path object
path = gpath.Path()

# Define some interesting points
st = np.array([-0.4, 0.4, 0.])
wp = np.array([0.0, 0.55, 0.])
gl = np.array([0.2, 0.4, 0.3])
pose = np.block([[st], [wp], [gl]])
max_v = [0.1, 0.1]
max_a = [0.2, 0.2]

# ARC
q_ar, qd_ar, qdd_ar, p_ar, pd_ar, pdd_ar = path.arc(np.array([0.35, 0.35, 0]),
                                                    np.array([0.346, 0.331, 0]),
                                                    np.array([0.3, 0.3, 0]),
                                                    0.5, 1)
# CIRCLE
st = np.array([0.1, 0.45, 0])
c = np.array([0., 0.45, 0])
q_cir, qd_cir, qdd_cir, p_cir, pd_cir, pdd_cir = path.circle(start=st,
                                                             center=c,
                                                             max_vel=.1,
                                                             max_acc=.5)

# LINE
q_lin1, qd_lin1, qdd_lin1, p_lin1, pd_lin1, pdd_lin1 = path.line_poly(
    start=pose[0, :], goal=pose[1, :], mean_v=5)
# LINE
pose = np.array([[0.35, 0.4, 0], [0.35, 0.35, 0]])
max_v = 0.1
max_a = 0.2
q_lin2, qd_lin2, qdd_lin2, p_lin2, pd_lin2, pdd_lin2 = path.line(
    pose=pose, max_v=max_v, max_a=max_a, enable_way_point=True)
# JOINT
q_j, qd_j, qdd_j, p_j, pd_j, pdd_j = path.go_to_poly(start=pose[0, :],
                                                     goal=pose[1, :],
                                                     mean_v=0.5)
# JOINT
q_j2, qd_j2, qdd_j2, p_j2, pd_j2, pdd_j2 = path.go_to(goals=pose,
                                                      max_v=max_v,
                                                      max_a=max_a,
                                                      way_point=False)
# MOVE FROM END FRAME
q_f1, qd_f1, qdd_f1, p_f1, pd_f1, pdd_f1 = path.move_z_from_end(z=0.05,
                                                                max_v=1,
                                                                max_a=0.5)
q_f2, qd_f2, qdd_f2, p_f2, pd_f2, pdd_f2 = path.move_y_from_end(y=0.05,
                                                                max_v=1,
                                                                max_a=0.5)
q_f3, qd_f3, qdd_f3, p_f3, pd_f3, pdd_f3 = path.move_x_from_end(x=0.05,
                                                                max_v=1,
                                                                max_a=0.5)
q_f4, qd_f4, qdd_f4, p_f4, pd_f4, pdd_f4 = path.move_from_end(
    np.array([0.05, -0.05, 0.05]))

# Joint diferents trajectories
q = np.block([[q_lin2], [q_ar]])
qd = np.block([[qd_lin2], [qd_ar]])
qdd = np.block([[qdd_lin2], [qdd_ar]])
p = np.block([[p_lin2], [p_ar]])
pd = np.block([[pd_lin2], [pd_ar]])
pdd = np.block([[pdd_lin2], [pdd_ar]])

# Show a trajectory
path.plot_joint(q, qdd, qdd)
path.plot_task(p, pd, pdd)
plt.show()

# Normalize desire trajectory
D = 3  # Diametro mayor
d = 1  # diametro menor
i = D / d  # relacion de transmision
pasos_rev = 200 / (2 * np.pi)
factor = i * pasos_rev
q = gpath.normalize_trajectory(q, factor)
# Save trajectory for send to robot

trj_path = dirname(dirname(abspath(__file__)))
trj_path += '/trajectories'
file_name = 'trajectory.csv'
path.export_trajectory(q, trj_path, file_name)
