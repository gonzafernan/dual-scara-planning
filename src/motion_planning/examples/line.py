from Path import Path
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from utilities import normalize_trajectory, export_trajectory

# Create a Path object
path = Path()

# Define some interesting points
st = path.robot.fkine(np.deg2rad([180, 0, 0]))
wp_1 = np.array([0.2, 0.5, 0.])
wp_2 = np.array([-0.2, 0.5, 0.])
gl = st.copy()
pose = np.block([[st], [wp_1], [wp_2], [gl]])
max_v = [0.3, 0.3, 0.3]
delta_p = np.diff(pose, axis=0)
# max_a = [0.5 + max_v[i]**2 / max(abs(dp)) for i, dp in enumerate(delta_p)]
max_a = [1, 1, 1]
# LINE
q_lin, qd_lin, qdd_lin, p_lin, pd_lin, pdd_lin = path.line(
    pose=pose, max_v=max_v, max_a=max_a, enable_way_point=False)
# Show a trajectory
path.plot_joint(q_lin, qd_lin, qdd_lin)
path.plot_task(p_lin, pd_lin, pdd_lin)
plt.show()

# Normalize desire trajectory
# D = 3  # Diametro mayor
# d = 1  # diametro menor
# i = D / d  # relacion de transmision
# pasos_rev = 200 / (2 * np.pi)
# factor = i * pasos_rev
factor = 4 * 300 / np.pi
q = normalize_trajectory(q_lin, factor)
# Save trajectory for send to robot

trj_path = dirname(dirname(abspath(__file__)))
trj_path += '/trajectories'
file_name = 'line.csv'
export_trajectory(q, trj_path, path.total_time, file_name)
