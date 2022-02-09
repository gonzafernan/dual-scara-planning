from timelaw import TimeLaw
from fivebar import FiveBar
import numpy as np
import matplotlib.pyplot as plt


class Path(object):
    """Path class"""

    def __init__(self, robot: FiveBar = FiveBar()) -> None:
        self.tl = TimeLaw()
        self.robot = robot
        self.dt = 0.01
        self.tf = 0.

    def move_from_end(
        self,
        goal: np.ndarray = np.array([0.0, 0., 0.]),
        max_v: float = 1,
        max_a: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Move relative to end effector frame in task space.
            Linear segment with parabolic blend

        Args:
            goal(np.ndarray): goal point 3X
            max_v(np.ndarray): max velocity
            max_a(np.ndarray): max acceleration

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """

        start = self.robot.endPose
        goal += self.robot.endPose
        pose = np.block([[start], [goal]])
        return self.line(pose, max_v, max_a)

    def move_x_from_end(
        self,
        x: float = 0.05,
        max_v: float = 1,
        max_a: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Move relative to end effector frame, x direcction in task space.
            Linear segment with parabolic blend

        Args:
            pose(float):  relative positions to move
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """

        start = self.robot.endPose
        goal = np.copy(self.robot.endPose)
        goal[0] = self.robot.endPose[0] + x
        pose = np.block([[start], [goal]])
        return self.line(pose, max_v, max_a)

    def move_y_from_end(
        self,
        y: float = 0.05,
        max_v: float = 1,
        max_a: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Move relative to end effector frame, y direcction in task space.
            Linear segment with parabolic blend

        Args:
            pose(float):  relative positions to move
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """

        start = self.robot.endPose
        goal = np.copy(self.robot.endPose)
        goal[1] = self.robot.endPose[1] + y
        pose = np.block([[start], [goal]])
        return self.line(pose, max_v, max_a)

    def move_z_from_end(
        self,
        z: float = 0.05,
        max_v: float = 1,
        max_a: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Move relative to end effector frame, z direcction in task space.
            Linear segment with parabolic blend

        Args:
            pose(float):  relative positions to move
            max_v(float): max velocity
            max_a(float): max acceleration

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """

        start = self.robot.endPose
        goal = np.copy(self.robot.endPose)
        goal[2] = self.robot.endPose[2] + z
        pose = np.block([[start], [goal]])
        return self.line(pose, max_v, max_a)

    def initialize(
            self,
            n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(
            (n, 3))

    def update_state(self, start, delta_x, s, sd, sdd):
        x = start + delta_x * s
        xd = sd * delta_x
        xdd = sdd * delta_x
        return x, xd, xdd

    def multi_point_interpolation(self, start, delta_p, t, tau, T, shift,
                                  enable_way_point):
        s, sd, sdd = self.tl.lspb(t=t - shift[0], tau=tau[0], T=T[0])
        next_point = True if s == 1 else False
        if enable_way_point:
            s2, sd2, sdd2 = self.tl.lspb(t=t - shift[1], tau=tau[1], T=T[1])
            return next_point, np.add(
                self.update_state(start, delta_p[0, :], s, sd, sdd),
                self.update_state(0, delta_p[1, :], s2, sd2, sdd2))
        else:
            return next_point, self.update_state(start, delta_p[0, :], s, sd,
                                                 sdd)

    def go_to(
        self,
        goals: np.ndarray,
        max_v: float,
        max_a: float,
        way_point: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Go to goal with linear segment with parabolic blend in joint
            space

        Args:
            goal(np.ndarray): goal point 3X
            max_v(np.ndarray): max velocity
            max_a(np.ndarray): max acceleration
            way_point(bool): enable use internal point like way points.
                Default = True

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """
        if goals.ndim != 2:
            raise ValueError("Error: goals has to be n X 3 (2 dimensions)")
        if goals.shape[0] != 2:
            goal_number = goals.shape[0] - 1  # Number of segments
            g_idx = 0
            T1 = 0
            dq = np.zeros((goal_number, 3))
            tau = np.zeros(goal_number)
            T = np.zeros(goal_number)
            for i in range(0, goal_number):
                # + counter clockwise
                dq[i, :] = self.robot.ikine(goals[i + 1, :]) - self.robot.ikine(
                    goals[i, :])
                tau[i], T[i] = self.tl.lspb_param(np.max(abs(dq[i, :])),
                                                  max_v[i], max_a[i])
            if way_point:
                last_goal = goal_number - 2
                T2 = T[0]
                self.tf = tau[-1] + T.sum()
            else:
                last_goal = goal_number - 1
                T2 = T[0] + tau[0]
                self.tf = tau.sum() + T.sum()
        else:
            dq = self.robot.ikine(goals[1, :]) - self.robot.ikine(goals[0, :])
            tau, T = self.tl.lspb_param(np.max(dq), max_v, max_a)
            self.tf = tau + T

        n = round(self.tf / self.dt) + 1
        p, q, qd, qdd = self.initialize(n)

        jstart = self.robot.ikine(goals[0, :])

        for i, t in enumerate(np.linspace(start=0, stop=self.tf, num=n)):
            if goals.shape[0] == 2:
                s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
                q[i, :], qd[i, :], qdd[i, :] = self.update_state(
                    jstart, dq, s, sd, sdd)
            else:
                next_point, [q[i, :], qd[i, :],
                             qdd[i, :]] = self.multi_point_interpolation(
                                 jstart, dq[g_idx:g_idx + 2, :], t,
                                 tau[g_idx:g_idx + 2], T[g_idx:g_idx + 2],
                                 [T1, T2], way_point)

                if next_point and g_idx != last_goal:
                    T1 = T2
                    g_idx += 1
                    T2 += T[g_idx]
                    jstart = self.robot.ikine(goals[g_idx, :])
                    if not way_point:
                        T2 += tau[g_idx]

            p[i, :] = self.robot.fkine(q[i, :])
        pd = np.gradient(p, self.dt, axis=0)
        pdd = np.gradient(pd, self.dt, axis=0)
        return q, qd, qdd, p, pd, pdd

    def go_to_poly(
        self, start: np.ndarray, goal: np.ndarray, mean_v: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Quintic polynomial interpolation in joint space

        Args:
            start(np.array): start point task space 3x
            goal(np.array): goal point task space 3x
            mean_v(float): mean velocity in joint space

        Returns:
            q(np.array): joints position n x 3
            qd(np.array): joints velocity n x 3
            qdd(np.array): joints acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """
        jstart = self.robot.ikine(start)
        jgoal = self.robot.ikine(goal)
        dq = jgoal - jstart
        self.tf = self.dt * np.ceil(
            np.max(abs(jgoal - jstart) / mean_v) / self.dt)
        a = self.tl.poly_coeff(0., 1., self.tf)
        n = round(self.tf / self.dt) + 1
        p, q, qd, qdd = self.initialize(n)

        for i, t in enumerate(np.linspace(start=0, stop=self.tf, num=n)):
            s, sd, sdd = self.tl.poly(t, a)

            q[i, :], qd[i, :], qdd[i, :] = self.update_state(
                jstart, dq, s, sd, sdd)
            p[i, :] = self.robot.fkine(q[i, :])
        pd = np.gradient(p, self.dt, axis=0)
        pdd = np.gradient(pd, self.dt, axis=0)
        return q, qd, qdd, p, pd, pdd

    def line_poly(
        self, start: np.ndarray, goal: np.ndarray, mean_v: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]:
        """ Lineal interpolation in task space
            Interpolation method: quintic polynomic

        Args:
            start(np.array): start point 3X1
            goal(np.array): goal point 3X1
            mean_v(float): mean velocity

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """
        self.tf = self.dt * np.ceil(
            np.max(abs(goal - start) / mean_v) / self.dt)
        a = self.tl.poly_coeff(0., 1., self.tf)

        n = round(self.tf / self.dt) + 1
        q, p, pd, pdd = self.initialize(n)

        for i, t in enumerate(np.linspace(start=0, stop=self.tf, num=n)):
            if i == n:
                break
            s, sd, sdd = self.tl.poly(t, a)
            p[i, :], pd[i, :], pdd[i, :] = self.update_state(
                start, goal - start, s, sd, sdd)
            q[i, :] = self.robot.ikine(p[i, :])
        qd = np.gradient(q, self.dt, axis=0)
        qdd = np.gradient(qd, self.dt, axis=0)
        return q, qd, qdd, p, pd, pdd

    def line(
        self,
        pose: np.ndarray,
        max_v: np.ndarray,
        max_a: np.ndarray,
        enable_way_point: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Lineal interpolation in task space
            Interpolation method: linear segment with parabolic blend

        Args:
            pose(np.ndarray): goals positions point nX3
            max_v(np.ndarray): max velocity nX
            max_a(np.ndarray): max acceleration nX
            enable_way_point(bool): Use each intermediate points like a way
                point. Default = True

        Returns:
            q(np.array): joint position n x 3
            qd(np.array): joint velocity n x 3
            qdd(np.array): joint acceleration n x 3
            p(np.array): task position n x 3
            pd(np.array): task velocity n x 3
            pdd(np.array): task acceleration n x 3
        """
        if pose.ndim != 2:
            raise ValueError("Error: pose has to be n X 3 (2 dimensions)")
        if pose.shape[0] != 2:
            segment_number = pose.shape[0] - 1
            s_idx = 0
            T1 = 0
            dx = np.zeros((segment_number, 3))
            tau = np.zeros(segment_number)
            T = np.zeros(segment_number)
            for i in range(0, segment_number):
                dx[i, :] = pose[i + 1, :] - pose[i, :]
                tau[i], T[i] = self.tl.lspb_param(np.max(abs(dx[i, :])),
                                                  max_v[i], max_a[i])
            if enable_way_point:
                last_segment = segment_number - 2
                T2 = T[0]
                self.tf = tau[-1] + T.sum()
            else:
                last_segment = segment_number - 1
                T2 = T[0] + tau[0]
                self.tf = tau.sum() + T.sum()
        else:
            dx = pose[1, :] - pose[0, :]
            tau, T = self.tl.lspb_param(np.max(dx), max_v, max_a)
            self.tf = tau + T

        n = round(self.tf / self.dt) + 1
        q, p, pd, pdd = self.initialize(n)

        for i, t in enumerate(np.linspace(start=0, stop=self.tf, num=n)):
            if pose.shape[0] == 2:
                s, sd, sdd = self.tl.lspb(t=t, tau=tau, T=T)
                p[i, :], pd[i, :], pdd[i, :] = self.update_state(
                    pose[0, :], dx, s, sd, sdd)
            else:
                next_point, [p[i, :], pd[i, :],
                             pdd[i, :]] = self.multi_point_interpolation(
                                 pose[s_idx, :], dx[s_idx:s_idx + 2, :], t,
                                 tau[s_idx:s_idx + 2], T[s_idx:s_idx + 2],
                                 [T1, T2], enable_way_point)
                if next_point and s_idx != last_segment:
                    T1 = T2
                    s_idx += 1
                    T2 += T[s_idx]
                    if not enable_way_point:
                        T2 += tau[s_idx]

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

        Note:
            tf has to be a sum of all trajectories times
        """
        t = np.linspace(start=0, stop=self.tf, num=max(q.shape))

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
        t = np.linspace(start=0, stop=self.tf, num=max(p.shape))

        fig = plt.figure(figsize=plt.figaspect(2.))
        fig.suptitle("Task space")
        ax = fig.add_subplot(4, 1, 1, projection='3d')
        ax.plot(p[:, 0], p[:, 1], zs=p[:, 2])
        ax.plot(p[0, 0], p[0, 1], 'r*')
        ax.plot(p[-1, 0], p[-1, 1], 'r*')
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
    # path.robot.endPose = np.ndarray([0.0, 0.3, 0.0])
    # path.robot.endPose = np.array([0., 0.3, 0.])
    # q, qd, qdd, p, pd, pdd = path.move_from_end(np.array([0.05, -0.05, 0.05]))
    # st = np.array([-0.5, 0.5, 0.])
    # gl = np.array([0.0, 0.35, 0.3])
    pose = np.array([[-0.6, 0.1, 0.], [0.0, 0.6, 0.], [0.6, 0.1, 0.],
                     [0.0, 0.5, 0.0], [-0.5, 0.2, 0.0], [0.0, 0.4, 0.0],
                     [0.5, 0.2, 0.0], [0., 0.3, 0.0], [-0.4, 0.2, 0.0],
                     [0., 0.25, 0.]])
    max_v = np.array([0.1 for i in range(0, 9)])
    max_a = np.array([0.1 for i in range(0, 9)])
    # q, qd, qdd, p, pd, pdd = path.line_poly(start=pose[0, :],
    #                                         goal=pose[1, :],
    #                                         mean_v=5)
    # q, qd, qdd, p, pd, pdd = path.line(pose=pose,
    #                                    max_v=max_v,
    #                                    max_a=max_a,
    #                                    enable_way_point=False)
    # q, qd, qdd, p, pd, pdd = path.go_to_poly(start=pose[0, :],
    #                                          goal=pose[1, :],
    #                                          mean_v=0.5)
    q, qd, qdd, p, pd, pdd = path.go_to(goals=pose,
                                        max_v=max_v,
                                        max_a=max_a,
                                        way_point=False)

    path.plot_joint(q, qd, qdd)
    path.plot_task(p, pd, pdd)
    plt.figure(3)
    plt.plot(p[:, 0], p[:, 1], 'r')

    plt.show()
