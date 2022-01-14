import numpy as np
import matplotlib.pyplot as plt
# import os

# Table 7.1 The MDH parameters of the five-bar legs
# ij a(ij) μ_ij σ_ij γ_ij b_ij α_ij d_ij θ_ij r_ij
# 11 0     1    0    0    0    0    d11  q11  0
# 12 11    0    0    0    0    0    d12  q12  0
# 13 12    0    0    0    0    π    d13  q13  0
# 14 13    0    1    0    0    0    q14  0    0
# 21 0     1    0    0    0    0    d21  q21  0
# 22 21    0    0    0    0    0    d22  q22  0
# 23 22    0    2    0    0    0    d23  0    0


class Arm(object):
    """ This class contain the representation o a 2 dof plannar robotic arm

    Attributes:
        links (np.ndarray): 2X1 proximal and distal link length.
        working (int): Current working mode of the arm.
        base (double): distance form the origin to the base of the arm.
        rOA (np.ndarray): 2x1 point in task space that represent the union
        between the proximal and distal links.

    """

    def __init__(self, d: np.ndarray, wkMode: int) -> None:
        """__init_ method:

        Arg:
            d (np.ndarray): 3X1 d column of MDH array
            wkMode (int): Current working mode of the arm.

        """
        self.links = d[1:]  # proximal and distal links
        self.working = wkMode  # +1 -1
        self.base = np.array([d[0], 0.0], dtype=float)
        self.rOA = np.array([0.0, 0.0], dtype=float)


class FiveBar(object):
    """ This class implement a model of a five bars robot also know as
    scara parallel robot. Units are in meters [m] and radians [rad].

    Atributtes:
        arms (list(Arm)): 2X1 list that cointain the 2 dof arms.
        endEff (np.ndarray): 3X1 vector that represent the
            end effector pose [x,y,z].
        assembly (int): Current assembly mode.
        joints (np.ndarray): joints position in radians

    """

    def __init__(
        self,
        d1: np.ndarray = np.array([0., .23, .35]),
        d2: np.ndarray = np.array([0., .23, .35])
    ) -> None:
        """__init_ method:

        Arg:
            d1 (np.ndarray): 3X1 d column of MDH array
            d2 (np.ndarray): 3X1 d column of MDH array

        """
        self.arms = [Arm(d1, -1), Arm(d2, 1)]
        # End Effector co0rdinates
        self.endPose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.assembly = 1  # +1 o -1
        self.joints = np.array([0.0, 0.0, 0.0], dtype=float)
        self.jlimit = np.array([[np.deg2rad(-40),
                                 np.deg2rad(220)],
                                [np.deg2rad(-40),
                                 np.deg2rad(220)], [0, 0.5]])
        self.minDistalAngle = np.deg2rad(5)
        self.minProximalAngle = np.deg2rad(5)

    def ikine(self, pose: np.ndarray = 0) -> np.ndarray:
        """ Inverse kinematics. If non parameter is passed the current
        position of the robot

        Arg:
            pose (np.ndarray): 3x1 vector. Dessire end effector pose [x, y, z]


        Retrun:
            q (np.ndarray): 3X1 vector. Joint position for achive dessire pose
                [q1, q2, d3]

        """
        joints = np.zeros(3)
        if isinstance(pose, int):
            p = self.endPose[:2]
            # z axis is equal to q[2]
            joints[-1] = self.endPose[-1].item()
        else:
            p = pose[:2]
            joints[-1] = pose[-1].item()

        for i, arm in enumerate(self.arms):
            phi = np.linalg.norm(p - arm.base)
            if phi <= arm.links.sum(
            ):  # if it is equal it is in serie singularity
                OCu = (p - arm.base) / phi
                foot = (arm.links[0]**2 - arm.links[1]**2 + phi**2) / (2 * phi)
                height = np.sqrt(arm.links[0]**2 - foot**2)

                A = arm.base + foot * OCu + arm.working * height * np.array(
                    [[0, 1], [-1, 0]]) @ OCu

                # Update robot
                joints[i] = np.arctan2(A[1], A[0])
            else:
                # TODO:Implementar alguna forma para que no se bloquee.
                # TODO:Throw exception
                print('Point outside of workspace')
                joints[:2] = self.joints[:2]
        self.arms[0].rOA = self.arms[0].base + self.arms[0].links[0] * \
            np.array([np.cos(joints[0]), np.sin(joints[0])])
        self.arms[1].rOA = self.arms[1].base + self.arms[1].links[0] * \
            np.array([np.cos(joints[1]), np.sin(joints[1])])
        self.joints = joints
        if self.is_inside_bounds(joints[0], joints[1]):
            return joints
        else:
            print("Out of bounds")
            return joints

    def fkine(self, joint: np.ndarray = 0) -> np.ndarray:
        """ Forward kinematics

        Arg:
            q (np.ndarray): 3X1 vector. Joint position for achive dessire pose
                [q1, q2, d3]

        Retrun:
            p (np.ndarray): 3x1 vector. Dessire end effector pose [x, y, z]

        """
        if isinstance(joint, int):
            q = self.joints
        else:
            q = joint
        rOA1 = self.arms[0].base + self.arms[0].links[0] * \
            np.array([np.cos(q[0]), np.sin(q[0])])
        rOA2 = self.arms[1].base + self.arms[1].links[0] * \
            np.array([np.cos(q[1]), np.sin(q[1])])

        phi = np.linalg.norm(rOA2 - rOA1)

        if phi <= self.arms[0].links[1] + self.arms[1].links[1]:
            f = (rOA2 - rOA1) / 2.0
            h = np.sqrt(4 * self.arms[0].links[1]**2 - phi**2) * \
                np.array([-f[1], f[0]]) / phi

            # In working mode -+ we have to rotate -90 deg
            if (self.arms[0].working == -1 and self.arms[1].working == 1):
                p = rOA1 + f + self.assembly * h
            else:
                p = rOA1 + f - self.assembly * h

            p = np.append(p, q[-1])
            # Update robot
            self.endPose = p
            self.arms[0].rOA = rOA1
            self.arms[1].rOA = rOA2
            if self.is_inside_bounds(q[0], q[1]):
                return p
            else:
                print("Out of bounds")
                return np.zeros((1, 3))
        else:
            print('Imposible to solve forward kinematic')
            return np.zeros((1, 3))

    def showRobot(self):
        """ Show robot with numpy

        """
        # Link 11
        l11_x = [self.arms[0].base[0], self.arms[0].rOA[0]]
        l11_y = [self.arms[0].base[1], self.arms[0].rOA[1]]

        # Link 12
        l12_x = [self.arms[0].rOA[0], self.endPose[0]]
        l12_y = [self.arms[0].rOA[1], self.endPose[1]]

        # Link 21
        l21_x = [self.arms[1].base[0], self.arms[1].rOA[0]]
        l21_y = [self.arms[1].base[1], self.arms[1].rOA[1]]

        # Link 22
        l22_x = [self.arms[1].rOA[0], self.endPose[0]]
        l22_y = [self.arms[1].rOA[1], self.endPose[1]]

        # Arm 1
        plt.plot(l11_x, l11_y, 'r', linewidth=3.0)
        plt.plot(l12_x, l12_y, 'r', linewidth=3.0)
        plt.plot(self.arms[0].rOA[0], self.arms[0].rOA[1], 'k *')
        # Arm 2
        plt.plot(l21_x, l21_y, 'b', linewidth=3.0)
        plt.plot(l22_x, l22_y, 'b', linewidth=3.0)
        plt.plot(self.arms[1].rOA[0], self.arms[1].rOA[1], 'k *')

        # End
        plt.plot(self.endPose[0], self.endPose[1], 'k *')

        plt.axis(0.6 * np.array([-1, 1, -1, 1]))
        plt.grid(True)

    def distals_angle(self):
        end2r1 = self.arms[0].rOA - self.endPose[:2]
        end2r1 /= np.linalg.norm(end2r1)
        end2r2 = self.arms[1].rOA - self.endPose[:2]
        end2r2 /= np.linalg.norm(end2r2)
        dot_product = np.dot(end2r1, end2r2)
        return abs(np.arccos(dot_product))

    def proximals_angle(self):
        base2r1 = self.arms[0].rOA - self.arms[0].base[:2]
        base2r1 /= np.linalg.norm(base2r1)
        base2r2 = self.arms[1].rOA - self.arms[1].base[:2]
        base2r2 /= np.linalg.norm(base2r2)
        dot_product = np.dot(base2r2, base2r1)
        return abs(np.arccos(dot_product))

    def is_inside_bounds(self, q1, q2):
        return self.distals_angle(
        ) > self.minDistalAngle and self.proximals_angle(
        ) > self.minProximalAngle and q2 < q1

    def work_space(self):
        q1 = np.arange(start=self.jlimit[0, 0],
                       stop=self.jlimit[0, 1] + 0.1,
                       step=0.1)
        q2 = np.copy(q1)
        max_points = 1035
        p = np.zeros((max_points, 3))
        q = np.copy(p)
        count = 0
        for q_2 in q2:
            for q_1 in q1:
                self.fkine(np.array([q_1, q_2, 0.]))
                if self.is_inside_bounds(q_1, q_2):
                    p[count, :] = np.copy(
                        self.endPose)  # self.fkine(np.array([q_1, q_2, 0.]))
                    q[count, :] = np.array([q_1, q_2, 0.])
                    # self.showRobot()
                    # filenumber = count
                    # filenumber = format(filenumber, "05")
                    # filename = "image{}.png".format(filenumber)
                    # plt.savefig(filename)
                    # plt.close()
                    count += 1
        # os.system("ffmpeg -f image2 -r 25 -i image%05d.png -vcodec mpeg4 -y \
        #         workspace.avi")
        # os.system("rm *.png")
        plt.figure(1)
        plt.plot(p[:count, 0], p[:count, 1], 'r .')
        # plot((proximal + distal)*cos(dtheta-pi/2:0.01:pi/2-dtheta)-base, ...
        rl = self.arms[0].links.sum()
        plt.plot(rl * np.cos(q1), rl * np.sin(q1), 'k')
        # rs = self.arms[0].links[1] - self.arms[0].links[0]
        # plt.plot(rs * np.cos(q1), rs * np.sin(q1), 'k')
        q3 = np.arange(start=self.jlimit[0, 0],
                       stop=np.deg2rad(120) + 0.1,
                       step=0.1)
        plt.plot(0.35 * np.cos(q3) + 0.23 * np.cos(self.jlimit[0, 0]),
                 0.35 * np.sin(q3) + 0.23 * np.sin(self.jlimit[0, 0]), 'k')
        q4 = np.arange(start=np.deg2rad(65),
                       stop=np.deg2rad(210) + 0.1,
                       step=0.1)
        # Find what kind of curve is.
        plt.plot(0.355 * np.cos(q4) + 0.23 * np.sin(self.jlimit[0, 1]),
                 0.355 * np.sin(q4) + 0.23 * np.cos(self.jlimit[0, 1]), 'k')
        plt.grid(True)
        # plt.figure(2)
        # plt.plot(q[:count, 0], q[:count, 1], 'b .')
        # plt.grid(True)


def main():
    l1 = 0.23
    l2 = 0.35
    b = 0.0
    five = FiveBar(np.array([-b, l1, l2]), np.array([b, l1, l2]))
    five.work_space()
    # [-0.1, 0.2, 0.] [0.0, 0.3, 0.]
    # five.endPose = np.array([0.1, 0.2, 0.])
    # print("-+ mode 1")
    # five.arms[0].working = -1
    # five.arms[1].working = 1
    # five.assembly = 1

    # five.ikine()
    # print("inverse")
    # print(five.joints)
    # plt.subplot(2, 2, 1)
    # five.showRobot()
    # # five.fkine(np.array([-0.1, 0.2]))
    # print("direct")
    # five.fkine()
    # print(five.endPose)
    # plt.subplot(2, 2, 2)
    # five.showRobot()
    plt.show()
    # print(np.array([[0, 1], [-1, 0]]) @ np.array([1, 2]))


if __name__ == "__main__":
    main()
