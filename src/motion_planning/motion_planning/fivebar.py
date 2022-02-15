import numpy as np
import matplotlib.pyplot as plt
import os

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
    """ This class contain the representation o a 2 dof planar robotic arm

    Attributes:
        links (np.ndarray): 2X1 proximal and distal link length.
        working (int): Current working mode of the arm.
        base (double): distance form the origin to the base of the arm.
        rOA (np.ndarray): 2x1 point in task space that represent the union \
        between the proximal and distal links.

    """

    def __init__(self, d: np.ndarray, wkMode: int) -> None:
        """__init_ method:

        Args:
            d (np.ndarray): 3X1 d column of MDH array
            wkMode (int): Current working mode of the arm.

        """
        self.links = d[1:]  # proximal and distal links
        self.working = wkMode  # +1 -1
        self.base = np.array([d[0], 0.0], dtype=float)
        self.rOA = np.array([0.0, 0.0], dtype=float)


class FiveBar(object):
    """ This class implement a model of a five bars robot also know as \
        scara parallel robot. Units are in meters [m] and radians [rad].

    Attributes:
        arms (list(Arm)): 2X1 list that contain the 2 dof arms.
        endEff (np.ndarray): 3X1 vector that represent the \
        end effector pose [x,y,z].
        assembly (int): Current assembly mode.
        joints (np.ndarray): joints position in radians

    """

    def __init__(
        self,
        d1: np.ndarray = np.array([0., .25, .38]),
        d2: np.ndarray = np.array([0., .25, .38])
    ) -> None:
        """ __init_ method:

        Args:
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
        self.minDistalAngle = np.deg2rad(1)
        self.minProximalAngle = np.deg2rad(1)
        self.make_video = False

    def transform_angle(self, q1, q2):
        """ Normalize angles q1 and q2 """
        if q1 <= -np.pi / 2:
            q1 = q1 % (2 * np.pi)
        if q2 <= -np.pi / 2:
            q2 = q2 % (2 * np.pi)
        return q1, q2

    def are_distals_ok(self):
        end2r1 = self.arms[0].rOA - self.endPose[:2]
        end2r1 /= np.linalg.norm(end2r1)
        end2r2 = self.arms[1].rOA - self.endPose[:2]
        end2r2 /= np.linalg.norm(end2r2)
        dot_product = np.dot(end2r1, end2r2)
        return abs(np.arccos(dot_product)) >= self.minDistalAngle

    def are_proximals_ok(self, q1, q2):
        if q1 > q2:
            angle = q1 - q2
            if angle >= self.minProximalAngle:
                return True
            else:
                return False
        else:
            return False

    def are_inside_limits(self, q1, q2):
        return (q1 >= self.jlimit[0, 0] and
                q1 <= self.jlimit[0, 1]) and (q2 >= self.jlimit[1, 0] and
                                              q2 <= self.jlimit[1, 1])

    def is_inside_bounds(self, q1, q2):
        q1, q2 = self.transform_angle(q1, q2)
        if not self.are_distals_ok():
            print("distal")
        if not self.are_proximals_ok(q1, q2):
            print("proximal")
        if not self.are_inside_limits(q1, q2):
            print("Limits")
        return self.are_distals_ok() and self.are_proximals_ok(
            q1, q2) and self.are_inside_limits(q1, q2)

    def ikine(self, pose: np.ndarray = 0) -> np.ndarray:
        """ Five Bar inverse kinematics. If non parameter is passed the current \
            position of the robot

        Args:
            pose (np.ndarray): 3x1 vector. Desire end effector pose [x, y, z]

        Returns:
            np.ndarray: q 3X1 vector. Joint position for achieve desire pose \
            [q1, q2, d3]

        """
        joints = np.zeros(3)
        if isinstance(pose, int):
            p = np.copy(self.endPose[:2])
            # z axis is equal to q[2]
            joints[-1] = np.copy(self.endPose[-1].item())
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
                print(f'Point outside of workspace {pose}')
                RuntimeError(f"Point outside of workspace {pose}")
                # exit()
        self.arms[0].rOA = self.arms[0].base + self.arms[0].links[0] * \
            np.array([np.cos(joints[0]), np.sin(joints[0])])
        self.arms[1].rOA = self.arms[1].base + self.arms[1].links[0] * \
            np.array([np.cos(joints[1]), np.sin(joints[1])])
        if self.is_inside_bounds(joints[0], joints[1]):
            joints[0], joints[1] = self.transform_angle(joints[0], joints[1])
            self.joints = joints
            self.endPose = np.block([p, joints[-1]])
            return joints
        else:
            print(f"Pose out of bounds {pose}")
            RuntimeError(f"Pose out of bounds {pose}")
            # exit()

    def fkine(self, joint: np.ndarray = 0) -> np.ndarray:
        """ Five Bar forward kinematics. If non parameter is passed the current \
            position of the robot

        Args:
            q (np.ndarray): 3X1 vector. Joint position for achieve desire pose \
            [q1, q2, d3]

        Returns:
            np.ndarray: p 3x1 vector. Desire end effector pose [x, y, z]

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
            self.arms[0].rOA = rOA1
            self.arms[1].rOA = rOA2
            if self.is_inside_bounds(q[0], q[1]):
                self.endPose = p
                return p
            else:
                print(f"Joints out of bounds: {np.rad2deg(q)}")
                RuntimeError(f"Joints out of bounds: {np.rad2deg(q)}")
                # exit()
        else:
            print(f'Imposible to solve forward kinematic for joints: {q}')
            exit()

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

        plt.axis(0.7 * np.array([-1, 1, -1, 1]))
        plt.grid(True)

    def work_space(self):
        q1 = np.arange(start=self.jlimit[0, 0] - 0.1,
                       stop=self.jlimit[0, 1] + 0.2,
                       step=0.1)
        q2 = np.copy(q1)
        max_points = q1.size * q1.size
        p = np.zeros((max_points, 3))
        q = np.zeros((max_points, 3))
        count = 0
        for q_1 in q1:
            for q_2 in q2:
                try:
                    self.fkine(np.array([q_1, q_2, 0.]))

                    if self.is_inside_bounds(q_1, q_2):
                        q[count, :] = np.array([q_1, q_2, 0])
                        p[count, :] = self.endPose
                        if self.make_video:
                            self.showRobot()
                            filenumber = count
                            filenumber = format(filenumber, "05")
                            filename = "image{}.png".format(filenumber)
                            plt.savefig(filename)
                            plt.close()
                        count += 1
                except RuntimeError:
                    continue
        if self.make_video:
            os.system(
                "ffmpeg -f image2 -r 25 -i image%05d.png -vcodec mpeg4 -y \
                    workspace.avi")
            os.system("rm *.png")

        plt.figure(1)
        print(count)
        plt.plot(p[:count, 0], p[:count, 1], 'r .')
        rl = self.arms[0].links.sum()
        plt.plot(rl * np.cos(q1), rl * np.sin(q1), 'k--')
        q3 = np.arange(start=np.deg2rad(-50), stop=np.deg2rad(125), step=0.1)
        plt.plot(0.38 * np.cos(q3) + 0.25 * np.cos(self.jlimit[0, 0]),
                 0.38 * np.sin(q3) + 0.25 * np.sin(self.jlimit[0, 0]), 'k--')
        q4 = np.arange(start=np.deg2rad(57), stop=np.deg2rad(220), step=0.1)
        plt.plot(0.38 * np.cos(q4) + 0.25 * np.cos(self.jlimit[0, 1]),
                 0.38 * np.sin(q4) + 0.25 * np.sin(self.jlimit[0, 1]), 'k--')
        plt.grid(True)
        plt.figure(2)
        plt.plot(np.rad2deg(q[:count, 0]), np.rad2deg(q[:count, 1]), 'b .')
        plt.grid(True)
        plt.show()
        return q


def main():
    l1 = 0.25
    l2 = 0.38
    b = 0.0
    five = FiveBar(np.array([-b, l1, l2]), np.array([b, l1, l2]))
    # print(five.fkine(np.array([np.pi, 0, 0])))
    five.work_space()


if __name__ == "__main__":
    main()
