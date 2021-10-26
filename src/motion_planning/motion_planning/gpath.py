from timelaw import TimeLaw
import numpy as np
import matplotlib.pyplot as plt


class Path(object):
    """Docstrings for Path"""

    def __init__(self) -> None:
        self.tl = TimeLaw()

    def to_goal(self, start: np.matrix, goal: np.matrix,
                med_v: float) -> tuple[np.matrix, np.matrix, np.matrix]:
        """ Lineal interpolation in task space

        Args:
            start(np.matrix): start point 1X3
            goal(np.matrix): goal point 1X3
            med_v(float): mean velocity

        Returns:
            q(np.matrix): position n x 3
            qd(np.matrix): velocity n x 3
            qdd(np.matrix): acceleration n x 3
        """
        dt = 0.001
        tf = np.max((goal - start) / med_v)
        a = self.tl.poly_coeff(0., 1., tf)
        n = int(tf / dt + 1)
        q = np.zeros((n, 3))
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))
        for i, t in enumerate(np.arange(start=0, stop=tf, step=dt)):
            s, sd, sdd = self.tl.poly(t, a)
            q[i, :] = start + (goal - start) * s
            qd[i, :] = sd * (goal - start)
            qdd[i, :] = sdd * (goal - start)
        return q, qd, qdd

    def to_goal_lspb(self, start: np.matrix, goal: np.matrix, max_v: float,
                     max_a: float) -> tuple[np.matrix, np.matrix, np.matrix]:
        """ Lineal interpolation in task space
            Linear segment with parabolic blend

        Args:
            start(np.matrix): start point 1X3
            goal(np.matrix): goal point 1X3
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.matrix): position n x 3
            qd(np.matrix): velocity n x 3
            qdd(np.matrix): acceleration n x 3
        """
        dx = goal - start
        tau, T = self.tl.lspb_param(np.max(dx), max_v, max_a)
        dt = 0.001
        tf = tau + T + dt
        n = int(tf / dt)
        q = np.zeros((n, 3))
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))
        for i, t in enumerate(np.arange(start=0, stop=tf, step=dt)):
            s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
            q[i, :] = start + s * dx
            qd[i, :] = sd * dx
            qdd[i, :] = sdd * dx
        return q, qd, qdd


if __name__ == '__main__':
    path = Path()
    st = np.matrix([0., 0., 0.])
    gl = np.matrix([15., 10., 3.])
    # q, qd, qdd = path.to_goal(start=st, goal=gl, med_v=5)
    q, qd, qdd = path.to_goal_lspb(start=st, goal=gl, max_v=5, max_a=10)

    fig = plt.figure(figsize=plt.figaspect(2.))
    fig.suptitle('A tale of 2 subplots')
    ax = fig.add_subplot(3, 1, 1, projection='3d')
    ax.plot(q[:, 0], q[:, 1], zs=q[:, 2])
    ax = fig.add_subplot(3, 1, 2)
    ax.grid(True)
    ax.plot(qd[:, 0], label='$\dot{x}$')  # noqa
    ax.plot(qd[:, 1], "r--", label='$\dot{y}$')  # noqa
    ax.plot(qd[:, 2], "g", label='$\dot{z}$')  # noqa
    plt.legend()
    ax = fig.add_subplot(3, 1, 3)
    ax.grid(True)
    ax.plot(qdd[:, 0], label='$\ddot{x}$')  # noqa
    ax.plot(qdd[:, 1], "r--", label='$\ddot{y}$')  # noqa
    ax.plot(qdd[:, 2], "g", label='$\ddot{z}$')  # noqa
    plt.legend()
    fig.suptitle("Polynomic time law")
    plt.show()
