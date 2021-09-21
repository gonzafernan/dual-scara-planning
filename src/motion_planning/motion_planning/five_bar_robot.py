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

    """Docstring for Arm. """

    def __init__(self, d:np.ndarray, wkMode:int):
        """TODO: to be defined. """
        self.d_ = d[1:]
        self.wkMode_ = wkMode # +1 -1
        self.base_ = np.array([d[0],0.0], dtype=float)
        self.rOA = np.array([0,0])

class FiveBar(object):

    """Docstring for FiveBar. """

    def __init__(self, d1:np.ndarray, d2:np.ndarray):
        """TODO: to be defined. """

        self.arms = [Arm(d1,1), Arm(d2,1)]
        self.endEff = np.array([0.0,0.0], dtype=float) # End Effector co0rdinates
        self.assMode = 1 # +1 o -1
        self.q = np.array([0.0,0.0], dtype=float)

    def ikine(self):
        """TODO: Docstring for ikine.

        :arg1: TODO
        :returns: TODO

        """
        for i,arm in enumerate(self.arms):
            phi = np.linalg.norm(-self.endEff+arm.base_)
            if phi <= arm.d_.sum(): # si es = esta en una singularidad serie
                f = (self.endEff - arm.base_) / 2
                h = np.sqrt(4 * arm.d_[0]**2 - phi**2) * \
                            np.array([f[1],-f[0]]) / phi

                r = f + arm.wkMode_ * h
                self.q[i] = np.arctan2(r[1],r[0])
            else:
                # TODO:Implementar alguna forma para que no se bloquee.
                print('Point outside of workspace')
        return self.q

    def fkine(self):
        """TODO: Docstring for fkine.

        :q: current joint position
        :ass: Assembly mode
        :returns: TODO
        """

        self.arms[0].rOA = self.arms[0].base_ + self.arms[0].d_[0] * \
            np.array([np.cos(self.q[0]), np.sin(self.q[0])])
        rOA1 = self.arms[0].rOA
        self.arms[1].rOA = self.arms[1].base_ + self.arms[1].d_[0] * \
            np.array([np.cos(self.q[1]), np.sin(self.q[1])])
        rOA2 = self.arms[1].rOA

        phi = np.linalg.norm(rOA1 - rOA2)

        if phi <= self.arms[0].d_[1] + self.arms[1].d_[1]:
            f = (rOA2 - rOA1) / 2.0
            h = np.sqrt(4 * self.arms[0].d_[0]**2 - phi**2) * \
                        np.array([-f[1],f[0]]) / phi

            p = rOA1 + f  + self.assMode * h
            # Verifico que este en el ensamble correcto
            if (self.arms[0].wkMode_ == 1 and
                    self.arms[1].wkMode_ == -1):
                p = rOA1 + f  - self.assMode * h
            # elif (self.arms[0].wkMode_ == -1 and
            #         self.arms[1].wkMode_ == 1 and self.assMode == -1):
            #     p = rOA1 + f  + self.assMode * h

            self.endEff = p
            return p
        else:
            print('Imposible to solve inverse kinematic')
            return NaN

    def showRobot(self):
        """TODO: Docstring for showRobot.
        :returns: TODO

        """
        # Link 11
        l11_x = [self.arms[0].base_[0], self.arms[0].rOA[0]]
        l11_y = [self.arms[0].base_[1], self.arms[0].rOA[1]]

        # Link 12
        l12_x = [self.arms[0].rOA[0], self.endEff[0]]
        l12_y = [self.arms[0].rOA[1], self.endEff[1]]

        # Link 21
        l21_x = [self.arms[1].base_[0], self.arms[1].rOA[0]]
        l21_y = [self.arms[1].base_[1], self.arms[1].rOA[1]]

        # Link 22
        l22_x = [self.arms[1].rOA[0], self.endEff[0]]
        l22_y = [self.arms[1].rOA[1], self.endEff[1]]

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
    l = 0.205
    b = 0.125
    five = FiveBar(np.array([-b, l,l]),np.array([b, l, l]))
    five.endEff = np.array([0.0,0.3])
    print("-+ mode 1")
    five.arms[0].wkMode_ = -1
    five.ikine()
    print(five.q)
    five.assMode = -1
    five.fkine()
    print(five.endEff)
    five.showRobot()

if __name__ == "__main__":
    main()
