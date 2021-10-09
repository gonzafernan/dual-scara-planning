import numpy as np
import matplotlib.pyplot as plt

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
            d (np.ndarray): 3X1 d column of MDH matrix
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

    def __init__(self, d1: np.ndarray, d2: np.ndarray) -> None:
        """__init_ method:

        Arg:
            d1 (np.ndarray): 3X1 d column of MDH matrix
            d2 (np.ndarray): 3X1 d column of MDH matrix

        """
        self.arms = [Arm(d1, 1), Arm(d2, 1)]
        # End Effector co0rdinates
        self.endPose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.assembly = 1  # +1 o -1
        self.joints = np.array([0.0, 0.0, 0.0], dtype=float)

    def ikine(self, *args) -> np.ndarray:
        """ Inverse kinematics

        Arg:
            p (np.ndarray): 3x1 vector. Dessire end effector pose [x, y, z]

        Retrun:
            q (np.ndarray): 3X1 vector. Joint position for achive dessire pose
                [q1, q2, d3]

        """
        if len(args) == 1:
            p = args[0][:2]  # EndEff x-y
            # z axis is equal to q[2]
            self.joints[2] = args[0][-1]
        else:
            p = self.endPose[:2]
            # z axis is equal to q[2]
            self.joints[2] = self.endPose[-1]

        for i, arm in enumerate(self.arms):
            phi = np.linalg.norm(-p+arm.base)
            if phi <= arm.links.sum():  # if equal is in serie singularity
                f = (p - arm.base) / 2
                h = np.sqrt(4 * arm.links[0]**2 - phi**2) * \
                    np.array([f[1], -f[0]]) / phi

                r = f + arm.working * h

                # Update robot
                self.joints[i] = np.arctan2(r[1], r[0])
            else:
                # TODO:Implementar alguna forma para que no se bloquee.
                print('Point outside of workspace')
        return self.joints

    def fkine(self, *args) -> np.ndarray:
        """ Forward kinematics

        Arg:
            q (np.ndarray): 3X1 vector. Joint position for achive dessire pose
                [q1, q2, d3]

        Retrun:
            p (np.ndarray): 3x1 vector. Dessire end effector pose [x, y, z]

        """
        if len(args) == 1:
            q = args[0]
        else:
            q = self.joints

        rOA1 = self.arms[0].base + self.arms[0].links[0] * \
            np.array([np.cos(q[0]), np.sin(q[0])])
        rOA2 = self.arms[1].base + self.arms[1].links[0] * \
            np.array([np.cos(q[1]), np.sin(q[1])])

        phi = np.linalg.norm(rOA1 - rOA2)

        if phi <= self.arms[0].links[1] + self.arms[1].links[1]:
            f = (rOA2 - rOA1) / 2.0
            h = np.sqrt(4 * self.arms[0].links[0]**2 - phi**2) * \
                np.array([-f[1], f[0]]) / phi

            # In working mode +- we have to change the assembly mode
            if (self.arms[0].working == 1 and
                    self.arms[1].working == -1):
                p = rOA1 + f - self.assembly * h
            else:
                p = rOA1 + f + self.assembly * h

            p = np.append(p, q[-1])
            # Update robot
            self.endPose = p
            self.arms[0].rOA = rOA1
            self.arms[1].rOA = rOA2
            return p
        else:
            print('Imposible to solve inverse kinematic')
            return np.NaN

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
        plt.plot(l11_x, l11_y, color='r', linewidth=3.0)
        plt.plot(l12_x, l12_y, color='r', linewidth=3.0)
        # Arm 2
        plt.plot(l21_x, l21_y, color='b', linewidth=3.0)
        plt.plot(l22_x, l22_y, color='b', linewidth=3.0)

        plt.axis(0.4*np.array([-1, 1, -1, 1]))
        plt.grid(True)
        plt.show()


def main():
    l = 0.205  # noqa
    b = 0.125
    five = FiveBar(np.array([-b, l, l]), np.array([b, l, l]))
    five.endPose = np.array([0.0, 0.3])
    print("-+ mode 1")
    five.arms[0].wkode_ = -1
    five.ikine()
    print(five.joints)
    five.assembly = -1
    five.fkine()
    print(five.endPose)
    five.showRobot()


if __name__ == "__main__":
    main()
