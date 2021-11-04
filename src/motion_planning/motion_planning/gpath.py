from timelaw import TimeLaw
from fivebar import FiveBar
import numpy as np
import matplotlib.pyplot as plt


class Path(object):
    """Docstrings for Path"""

    def __init__(self) -> None:
        self.tl = TimeLaw()
        self.robot = FiveBar()  # Have to be initialized
        self.robot.arms[0].working = -1
        self.dt = 0.001
        self.tf = 0.

    def go_to(self, start: np.ndarray, goal: np.ndarray, max_v: float,
              max_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Go to goal with linear segment with parabolic blend in joint
            space

        Args:
            start(np.array): start point 3X
            goal(np.array): goal point 3X
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.array): joint position 3 x n
            qd(np.array): joint velocity 3 x n
            qdd(np.array): joint acceleration 3 x n
            p(np.array): task position 3 x n
            pd(np.array): task velocity 3 x n
            pdd(np.array): task acceleration 3 x n
        """
        jstart = self.robot.ikine(start)
        jgoal = self.robot.ikine(goal)
        dq = jgoal - jstart
        tau, T = self.tl.lspb_param(np.max(dq), max_v, max_a)
        self.tf = tau + T + self.dt
        n = int(self.tf / self.dt)
        p = np.zeros((3, n))
        q = np.zeros((3, n))
        qd = np.zeros((3, n))
        qdd = np.zeros((3, n))
        for i, t in enumerate(np.arange(start=0, stop=self.tf, step=self.dt)):
            s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
            q[:, i] = jstart + (jgoal - jstart) * s
            qd[:, i] = sd * (jgoal - jstart)
            qdd[:, i] = sdd * (jgoal - jstart)
            p[:, i] = self.robot.fkine(q[:, i])
        pd = np.gradient(p, self.dt, axis=1)
        pdd = np.gradient(pd, self.dt, axis=1)
        return q, qd, qdd, p, pd, pdd

    def go_to_poly(self, start: np.ndarray, goal: np.ndarray,
                   mean_v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Quintic polynomial interpolation in joint space

        Args:
            start(np.array): start point task space 3x
            goal(np.array): goal point task space 3x
            mean_v(float): mean velocity in joint space

        Returns:
            q(np.array): joints position n x 3
            qd(np.array): joints velocity n x 3
            qdd(np.array): joints acceleration n x 3
            p(np.array): task position 3 x n
            pd(np.array): task velocity 3 x n
            pdd(np.array): task acceleration 3 x n
        """
        jstart = self.robot.ikine(start)
        jgoal = self.robot.ikine(goal)
        self.tf = np.max((jgoal - jstart) / mean_v)
        a = self.tl.poly_coeff(0., 1., self.tf)
        n = int(self.tf / self.dt)
        q = np.zeros((3, n))
        qd = np.zeros((3, n))
        qdd = np.zeros((3, n))
        p = np.zeros((3, n))
        for i, t in enumerate(np.arange(start=0, stop=self.tf, step=self.dt)):
            s, sd, sdd = self.tl.poly(t, a)
            q[:, i] = jstart + (jgoal - jstart) * s
            qd[:, i] = sd * (jgoal - jstart)
            qdd[:, i] = sdd * (jgoal - jstart)
            p[:, i] = self.robot.fkine(q[:, i])
        pd = np.gradient(p, self.dt, axis=1)
        pdd = np.gradient(pd, self.dt, axis=1)
        return q, qd, qdd, p, pd, pdd

    def line_poly(self, start: np.ndarray, goal: np.ndarray,
                  mean_v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Lineal interpolation in task space using a quintic polynomic

        Args:
            start(np.array): start point 3X1
            goal(np.array): goal point 3X1
            mean_v(float): mean velocity

        Returns:
            q(np.array): joint position 3 x n
            qd(np.array): joint velocity 3 x n
            qdd(np.array): joint acceleration 3 x n
            p(np.array): task position 3 x n
            pd(np.array): task velocity 3 x n
            pdd(np.array): task acceleration 3 x n
        """
        self.tf = np.max((goal - start) / mean_v)
        a = self.tl.poly_coeff(0., 1., self.tf)
        n = int(self.tf / self.dt)
        q = np.zeros((3, n))
        p = np.zeros((3, n))
        pd = np.zeros((3, n))
        pdd = np.zeros((3, n))
        for i, t in enumerate(np.arange(start=0, stop=self.tf, step=self.dt)):
            s, sd, sdd = self.tl.poly(t, a)
            p[:, i] = (start + (goal - start) * s)
            pd[:, i] = sd * (goal - start)
            pdd[:, i] = sdd * (goal - start)
            q[:, i] = self.robot.ikine(p[:, i])
        qd = np.gradient(q, self.dt, axis=1)
        qdd = np.gradient(qd, self.dt, axis=1)
        return q, qd, qdd, p, pd, pdd

    def line_lspb(self, start: np.ndarray, goal: np.ndarray, max_v: float,
                  max_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Lineal interpolation in task space
            Linear segment with parabolic blend

        Args:
            start(np.array): start point 3X
            goal(np.array): goal point 3X
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.array): joint position 3 x n
            qd(np.array): joint velocity 3 x n
            qdd(np.array): joint acceleration 3 x n
            p(np.array): task position 3 x n
            pd(np.array): task velocity 3 x n
            pdd(np.array): task acceleration 3 x n
        """
        dx = goal - start
        tau, T = self.tl.lspb_param(np.max(dx), max_v, max_a)
        self.tf = tau + T + self.dt
        n = int(self.tf / self.dt)
        q = np.zeros((3, n))
        p = np.zeros((3, n))
        pd = np.zeros((3, n))
        pdd = np.zeros((3, n))
        for i, t in enumerate(np.arange(start=0, stop=self.tf, step=self.dt)):
            s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
            p[:, i] = (start + s * dx)
            pd[:, i] = sd * (goal - start)
            pdd[:, i] = sdd * (goal - start)
            q[:, i] = self.robot.ikine(p[:, i])
        qd = np.gradient(q, self.dt, axis=1)
        qdd = np.gradient(qd, self.dt, axis=1)
        return q, qd, qdd, p, pd, pdd

    def plot_joint(self, q: np.ndarray, qd: np.ndarray,
                   qdd: np.ndarray) -> None:
        """ Plot joints values

        Args:
            q(np.array): joints position
            qd(np.array): joints velocity
            qdd(np.array): joints acceleration

        Returns:
        """
        t = np.arange(start=0, stop=self.tf, step=self.dt)

        # fig = plt.figure(figsize=plt.figaspect(2.))
        fig = plt.figure()
        fig.suptitle("Joints")
        ax = fig.add_subplot(3, 1, 1)
        ax.grid(True)
        ax.plot(t, q[0, :], label='$q_{1}$')  # noqa
        ax.plot(t, q[1, :], "r--", label='$q_{2}$')  # noqa
        ax.plot(t, q[2, :], "g-.", label='$q_{3}$')  # noqa
        ax.set_ylabel('rad')
        plt.legend()
        ax = fig.add_subplot(3, 1, 2)
        ax.grid(True)
        ax.plot(t, qd[0, :], label='$\dot{q_1}$')  # noqa
        ax.plot(t, qd[1, :], "r--", label='$\dot{q_2}$')  # noqa
        ax.plot(t, qd[2, :], "g-.", label='$\dot{q_3}$')  # noqa
        ax.set_ylabel('rad/s')
        plt.legend()
        ax = fig.add_subplot(3, 1, 3)
        ax.grid(True)
        ax.plot(t, qdd[0, :], label='$\ddot{q_1}$')  # noqa
        ax.plot(t, qdd[1, :], "r--", label='$\ddot{q_2}$')  # noqa
        ax.plot(t, qdd[2, :], "g-.", label='$\ddot{q_3}$')  # noqa
        ax.set_ylabel('rad/s^2')
        ax.set_xlabel('time (s)')
        plt.legend()

    def plot_task(self, p: np.ndarray, pd: np.ndarray, pdd: np.ndarray) -> None:
        """ Plot cartesian values

        Args:
            p(np.array): cartesian position
            pd(np.array): cartesian velocity
            pdd(np.array): cartesian acceleration

        Returns:

        Note:
            tf has to be a sum of all trajectories times
        """
        t = np.arange(start=0, stop=self.tf, step=self.dt)
        fig = plt.figure(figsize=plt.figaspect(2.))
        fig.suptitle("Task space")
        ax = fig.add_subplot(4, 1, 1, projection='3d')
        ax.plot(p[0, :], p[1, :], zs=p[2, :])
        ax = fig.add_subplot(4, 1, 2)
        ax.grid(True)
        ax.plot(t, p[0, :], label='$x$')  # noqa
        ax.plot(t, p[1, :], "r--", label='y')  # noqa
        ax.plot(t, p[2, :], "g-.", label='z')  # noqa
        ax.set_ylabel('m')
        plt.legend()
        ax = fig.add_subplot(4, 1, 3)
        ax.grid(True)
        ax.plot(t, pd[0, :], label='$\dot{x}$')  # noqa
        ax.plot(t, pd[1, :], "r--", label='$\dot{y}$')  # noqa
        ax.plot(t, pd[2, :], "g-.", label='$\dot{z}$')  # noqa
        ax.set_ylabel('m/s')
        plt.legend()
        ax = fig.add_subplot(4, 1, 4)
        ax.grid(True)
        ax.plot(t, pdd[0, :], label='$\ddot{x}$')  # noqa
        ax.plot(t, pdd[1, :], "r--", label='$\ddot{y}$')  # noqa
        ax.plot(t, pdd[2, :], "g-.", label='$\ddot{z}$')  # noqa
        ax.set_ylabel('m/s^2')
        ax.set_xlabel('time (s)')
        plt.legend()


if __name__ == '__main__':
    path = Path()
    st = np.array([-0.1, 0.3, 0.])
    gl = np.array([0.0, 0.35, 0.3])
    # q, qd, qdd, p, pd, pdd = path.line_poly(start=st, goal=gl, mean_v=5)
    # q, qd, qdd, p, pd, pdd = path.line_lspb(start=st,
    #                                         goal=gl,
    #                                         max_v=5,
    #                                         max_a=10)
    # q, qd, qdd, p, pd, pdd = path.go_to_poly(start=st, goal=gl, mean_v=0.5)
    q, qd, qdd, p, pd, pdd = path.go_to(start=st, goal=gl, max_v=5, max_a=10)
    path.plot_joint(q, qd, qdd)
    path.plot_task(p, pd, pdd)
    plt.show()
