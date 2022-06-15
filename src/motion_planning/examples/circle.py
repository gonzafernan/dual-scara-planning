from Path import Path
import numpy as np
from os.path import dirname, abspath
from utilities import export_trajectory, normalize_trajectory

# Create a Path object
path = Path()

# CIRCLE
st = path.robot.fkine(np.deg2rad([180, 0, 0]))
c = st + np.array([0., 0.05, 0])

q_cir, qd_cir, qdd_cir, p_cir, pd_cir, pdd_cir = path.circle(start=st,
                                                             center=c,
                                                             max_vel=0.2,
                                                             max_acc=1.)

# Show a trajectory
path.plot_joint(q_cir, qdd_cir, qdd_cir)
path.plot_task(p_cir, pd_cir, pdd_cir)

# Normalize desire trajectory
factor = 8 * 300 / np.pi
q = normalize_trajectory(q_cir, factor)

# Save trajectory for send to robot
trj_path = dirname(dirname(abspath(__file__)))
trj_path += '/trajectories'
file_name = 'circle.csv'

export_trajectory(q, trj_path, path.total_time, file_name)
