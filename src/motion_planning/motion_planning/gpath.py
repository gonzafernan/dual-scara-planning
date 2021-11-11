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
        tau, T = self.tl.lspb_param(np.max(abs(dq)), max_v, max_a)
        self.tf = tau + T
        n = int(self.tf / self.dt) + 1
        p = np.zeros((n, 3))
        q = np.zeros((n, 3))
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))
        for i, t in enumerate(
                np.arange(start=0, stop=self.tf + self.dt, step=self.dt)):
            s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
            q[i, :] = jstart + (jgoal - jstart) * s
            qd[i, :] = sd * (jgoal - jstart)
            qdd[i, :] = sdd * (jgoal - jstart)
            p[i, :] = self.robot.fkine(q[i, :])
        pd = np.gradient(p, self.dt, axis=0)
        pdd = np.gradient(pd, self.dt, axis=0)
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
        self.tf = self.dt * np.ceil(
            np.max(abs(jgoal - jstart) / mean_v) / self.dt)
        a = self.tl.poly_coeff(0., 1., self.tf)
        n = int(self.tf / self.dt) + 1
        q = np.zeros((n, 3))
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))
        p = np.zeros((n, 3))
        for i, t in enumerate(
                np.arange(start=0, stop=self.tf + self.dt, step=self.dt)):
            s, sd, sdd = self.tl.poly(t, a)
            q[i, :] = jstart + (jgoal - jstart) * s
            qd[i, :] = sd * (jgoal - jstart)
            qdd[i, :] = sdd * (jgoal - jstart)
            p[i, :] = self.robot.fkine(q[i, :])
        pd = np.gradient(p, self.dt, axis=0)
        pdd = np.gradient(pd, self.dt, axis=0)
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
        self.tf = self.dt * np.ceil(
            np.max(abs(goal - start) / mean_v) / self.dt)
        a = self.tl.poly_coeff(0., 1., self.tf)
        n = int(self.tf / self.dt) + 1
        q = np.zeros((n, 3))
        p = np.zeros((n, 3))
        pd = np.zeros((n, 3))
        pdd = np.zeros((n, 3))
        for i, t in enumerate(
                np.arange(start=0, stop=self.tf + self.dt, step=self.dt)):
            s, sd, sdd = self.tl.poly(t, a)
            p[i, :] = (start + (goal - start) * s)
            pd[i, :] = sd * (goal - start)
            pdd[i, :] = sdd * (goal - start)
            q[i, :] = self.robot.ikine(p[i, :])
        qd = np.gradient(q, self.dt, axis=0)
        qdd = np.gradient(qd, self.dt, axis=0)
        return q, qd, qdd, p, pd, pdd

    def line(self, pose: np.ndarray, max_v: np.ndarray,
             max_a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Lineal interpolation in task space
            Linear segment with parabolic blend

        Args:
            pose(np.ndarray): goals positions point nX3
            max_v(np.ndarray): max velocity nX
            max_a(np.ndarray): max acceleration nX

        Returns:
            q(np.array): joint position 3 x n
            qd(np.array): joint velocity 3 x n
            qdd(np.array): joint acceleration 3 x n
            p(np.array): task position 3 x n
            pd(np.array): task velocity 3 x n
            pdd(np.array): task acceleration 3 x n
        """
        if pose.ndim != 2:
            raise ValueError(
                "Error: goal and start have to have the same shape")
        if pose.shape[0] != 2:
            last_segment = pose.shape[0] - 1  # Number of segments
            segment = 0
            pre_T = 0
            dx = np.zeros((3, last_segment))
            tau = np.zeros(last_segment)
            T = np.zeros(last_segment)
            for i in range(0, last_segment):
                dx[:, i] = pose[i + 1, :] - pose[i, :]
                tau_, T_ = self.tl.lspb_param(np.max(abs(dx[:, i])), max_v[i],
                                              max_a[i])
                tau[i] = tau_
                T[i] = T_
            self.tf = tau[-1] + T.sum()
            T2 = T[0]
        else:
            dx = pose[1, :] - pose[0, :]
            tau, T = self.tl.lspb_param(np.max(dx), max_v, max_a)
            self.tf = tau + T
        n = int(self.tf / self.dt) + 1
        q = np.zeros((n, 3))
        p = np.zeros((n, 3))
        pd = np.zeros((n, 3))
        pdd = np.zeros((n, 3))
        for i, t in enumerate(
                np.arange(start=0, stop=self.tf + self.dt, step=self.dt)):
            if pose.shape[0] == 2:
                s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
                p[i, :] = pose[0, :] + s * dx
                pd[i, :] = sd * dx
                pdd[i, :] = sdd * dx
            else:
                s, sd, sdd = self.tl.lspb(t=t - pre_T,
                                          tau=tau[segment],
                                          T=T[segment])
                s2, sd2, sdd2 = self.tl.lspb(t=t - T2,
                                             tau=tau[segment + 1],
                                             T=T[segment + 1])
                p[i, :] = pose[
                    segment, :] + s * dx[:, segment] + s2 * dx[:, segment + 1]
                pd[i, :] = sd * dx[:, segment] + sd2 * dx[:, segment + 1]
                pdd[i, :] = sdd * dx[:, segment] + sdd2 * dx[:, segment + 1]
                if s == 1 and segment != last_segment - 2:
                    pre_T = T2
                    segment += 1
                    T2 += T[segment]

            q[i, :] = self.robot.ikine(p[i, :])
        qd = np.gradient(q, self.dt, axis=0)
        qdd = np.gradient(qd, self.dt, axis=0)
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
        t = np.arange(start=0, stop=self.tf + self.dt, step=self.dt)

        # fig = plt.figure(figsize=plt.figaspect(2.))
        fig = plt.figure()
        fig.suptitle("Joints")
        ax = fig.add_subplot(3, 1, 1)
        ax.grid(True)
        ax.plot(t, q[:, 0], label='$q_{1}$')  # noqa
        ax.plot(t, q[:, 1], "r--", label='$q_{2}$')  # noqa
        ax.plot(t, q[:, 2], "g-.", label='$q_{3}$')  # noqa
        ax.set_ylabel('rad')
        plt.legend()
        ax = fig.add_subplot(3, 1, 2)
        ax.grid(True)
        ax.plot(t, qd[:, 0], label='$\dot{q_1}$')  # noqa
        ax.plot(t, qd[:, 1], "r--", label='$\dot{q_2}$')  # noqa
        ax.plot(t, qd[:, 2], "g-.", label='$\dot{q_3}$')  # noqa
        ax.set_ylabel('rad/s')
        plt.legend()
        ax = fig.add_subplot(3, 1, 3)
        ax.grid(True)
        ax.plot(t, qdd[:, 0], label='$\ddot{q_1}$')  # noqa
        ax.plot(t, qdd[:, 1], "r--", label='$\ddot{q_2}$')  # noqa
        ax.plot(t, qdd[:, 2], "g-.", label='$\ddot{q_3}$')  # noqa
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
        t = np.arange(start=0, stop=self.tf + self.dt, step=self.dt)
        fig = plt.figure(figsize=plt.figaspect(2.))
        fig.suptitle("Task space")
        ax = fig.add_subplot(4, 1, 1, projection='3d')
        ax.plot(p[:, 0], p[:, 1], zs=p[:, 2])
        ax = fig.add_subplot(4, 1, 2)
        ax.grid(True)
        ax.plot(t, p[:, 0], label='$x$')  # noqa
        ax.plot(t, p[:, 1], "r--", label='y')  # noqa
        ax.plot(t, p[:, 2], "g-.", label='z')  # noqa
        ax.set_ylabel('m')
        plt.legend()
        ax = fig.add_subplot(4, 1, 3)
        ax.grid(True)
        ax.plot(t, pd[:, 0], label='$\dot{x}$')  # noqa
        ax.plot(t, pd[:, 1], "r--", label='$\dot{y}$')  # noqa
        ax.plot(t, pd[:, 2], "g-.", label='$\dot{z}$')  # noqa
        ax.set_ylabel('m/s')
        plt.legend()
        ax = fig.add_subplot(4, 1, 4)
        ax.grid(True)
        ax.plot(t, pdd[:, 0], label='$\ddot{x}$')  # noqa
        ax.plot(t, pdd[:, 1], "r--", label='$\ddot{y}$')  # noqa
        ax.plot(t, pdd[:, 2], "g-.", label='$\ddot{z}$')  # noqa
        ax.set_ylabel('m/s^2')
        ax.set_xlabel('time (s)')
        plt.legend()


if __name__ == '__main__':
    path = Path()
    st = np.array([-0.1, 0.3, 0.])
    gl = np.array([0.0, 0.35, 0.3])
    pose = np.array([[0., 0.3, 0.], [0.0, 0.3, 0.5], [0., 0.3, 0.]])
    max_v = np.array([0.5, 0.5])
    max_a = np.array([1, 1])
    # q, qd, qdd, p, pd, pdd = path.line_poly(start=st, goal=gl, mean_v=5)
    # q, qd, qdd, p, pd, pdd = path.line(pose=pose, max_v=max_v, max_a=max_a)
    # q, qd, qdd, p, pd, pdd = path.go_to_poly(start=st, goal=gl, mean_v=0.5)
    q, qd, qdd, p, pd, pdd = path.go_to(start=st, goal=gl, max_v=5, max_a=10)
    path.plot_joint(q, qd, qdd)
    path.plot_task(p, pd, pdd)
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    plt.show()
