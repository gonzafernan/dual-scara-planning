import numpy as np

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
        self.wkMode_ = wkMode
        self.base_ = np.array([d[0],0.0], dtype=float)

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
            if phi <= arm.d_.sum():
                f = (self.endEff - arm.base_) / 2
                h = np.sqrt(4 * arm.d_[0]**2 - phi**2) * \
                            np.array([f[1],-f[0]]) / phi

                r = f + arm.wkMode_ * h
                self.q[i] = np.arctan2(r[1],r[0])

    def fkine(self):
        """TODO: Docstring for fkine.

        :returns: TODO
        """
        rOA1 = self.arms[0].base_ + self.arms[0].d_[0] * \
            np.array([np.cos(self.q[0]), np.sin(self.q[0])])
        rOA2 = self.arms[1].base_ + self.arms[1].d_[0] * \
            np.array([np.cos(self.q[1]), np.sin(self.q[1])])

        phi = np.linalg.norm(rOA1 - rOA2)

        if phi <= self.arms[0].d_[1] + self.arms[1].d_[1]:
            f = (rOA2 - rOA1) / 2.0
            h = np.sqrt(4 * self.arms[0].d_[0]**2 - phi**2) * \
                        np.array([-f[1],f[0]]) / phi
            self.endEff = rOA1 + f  + self.assMode * h


def main():
    l = 0.205
    b = 0.125
    five = FiveBar(np.array([-b, l,l]),np.array([b, l, l]))
    five.endEff = np.array([0.05,0.3])
    print("++")
    five.ikine()
    print(five.q)
    five.fkine()
    print(five.endEff)

    print("-+")
    five.endEff = np.array([0.05,0.3])
    five.arms[0].wkMode_ = -1
    five.ikine()
    print(five.q)
    five.fkine()
    print(five.endEff)

    print("--")
    five.endEff = np.array([0.05,0.3])
    five.arms[1].wkMode_ = -1
    five.ikine()
    print(five.q)
    five.fkine()
    print(five.endEff)
# Atencion con este modo, es inverso a los demás para la cinematica directa da en el otro ensamble
    print("+-")
    five.endEff = np.array([0.05,0.3])
    five.arms[0].wkMode_ = 1
    five.assMode = -1
    five.ikine()
    print(five.q)
    five.fkine()
    print(five.endEff)
if __name__ == "__main__":
    main()
