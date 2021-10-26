import numpy as np
import matplotlib.pyplot as plt


class TimeLaw(object):
    """" Class that implement time laws

    """

    def __init__(self) -> None:
        pass

    def lspb_param(self, *args, **kwargs) -> tuple:
        """ Trapezoidal timing law parameters or
            Linear segment with parabolic blend
        Note: If a trapezoidal timing law can not be reached,
            with the parameters, this function keep doing a
            trapezoidal timing law. This aproche is achive by
            choosing tau = sqrt(delta_q/max_a) tringular law
            and T = max_v / max_a. This keep a trapezoidal law

        Args:
        delta_q (float): total displacement.
        max_v (float): maximal velocity.
        max_a (float): maximal acceleration.
        dt (float, optional): step time for round values.
            Defaults to 0.001

        Returns:
        tau (float): acceleration time
        T (float): Time to start deceleration
        """
        if len(kwargs) != 0:
            delta_q = kwargs["delta_q"]
            max_v = kwargs["max_v"]
            max_a = kwargs["max_a"]
            dt = 0.001
            if "dt" in kwargs:
                dt = kwargs["dt"]

        if len(args) != 0:
            delta_q = args[0]
            max_v = args[1]
            max_a = args[2]
            dt = 0.001
            if len(args) > 3:
                dt = args[3]

        T = delta_q / max_v
        tau = max_v / max_a

        # Si tau es mayor a T --> elijo tau como una ley triangular y para
        # mantener el trapecio elijo T = qv/qa que es el valor que ya
        # obtuve anteriormente para mantener mi ley trapezoidal
        if T < tau:
            T = tau
            tau = np.sqrt(delta_q / max_a)
        # Redondeo de los parametros T y tau
        T = dt * np.ceil(T / dt)
        tau = dt * np.ceil(tau / dt)

        # Verifico T y tau no sean menores a Tmin y  taumin
        K = 4  # Factor que define Tmin y taumin admisible
        if T < K * dt:
            T = K * dt
        if tau < K * dt:
            tau = K * dt
        return tau, T

    def lspb_s(self, t: float, tau: float, T: float) -> float:
        """ Trapezoidal time law or
            Linear segment with parabolic blend.
            Range from [0, 1]

        Args:
        t (float): current time
        T (float): time to start deceleration
        tau (float): acceleration time

        Returns:
        float: s(t) value.
        """
        a = 1 / (T * tau)
        v = 1 / T
        if t <= 0:
            s = 0
        elif t > 0 and t <= tau:
            s = a * t**2 / 2
        elif t > tau and t <= T:
            s_tau = a * tau**2 / 2
            s = s_tau + v * (t - tau)
        elif t > T and t <= T + tau:
            s_tau = a * tau**2 / 2
            s_T = s_tau + v * (T - tau)
            s = s_T + v * (t - T) - a * (t - T)**2 / 2
        elif t > T + tau:
            s = 1
        return s

    def lspb_sd(self, t: float, tau: float, T: float) -> float:
        """ Derivate trapezoidal time law or
            Linear segment with parabolic blend

        Args:
        t (float): current time
        T (float): deceleration time
        tau (float): acceleration time

        Retruns:
        float: sd(t) value.
        """
        a = 1 / (T * tau)
        v = 1 / T
        if t <= 0:
            sd = 0
        elif t <= tau:
            sd = a * t
        elif t > tau and t <= T:
            sd = v
        elif t > T and t <= T + tau:
            sd = v - a * (t - T)
        elif t > T + tau:
            sd = 0
        return sd

    def lspb_sdd(self, t: float, tau: float, T: float) -> float:
        """ Second derivate trapezoidal time law or
            Linear segment with parabolic blend

        Args:
        t (float): current time
        T (float): deceleration time
        tau (float): acceleration time

        Returns:
        float: sdd(t) value.
        """
        a = 1 / (T * tau)
        if t <= 0:
            sdd = 0
        elif t <= tau:
            sdd = a
        elif t > tau and t <= T:
            sdd = 0
        elif t > T and t <= T + tau:
            sdd = -a
        elif t > T + tau:
            sdd = 0
        return sdd

    def lspb(self, t: float, tau: float,
             T: float) -> tuple[float, float, float]:
        """ Trapezoidal time law or
            Linear segment with parabolic blend

        Args:
        t (float): current time
        T (float): deceleration time
        tau (float): acceleration time

        Retruns:
        float: sd(t) value.
        float: sdd(t) value.
        float: sddd(t) value.
        """
        s = self.lspb_s(t, tau, T)
        sd = self.lspb_sd(t, tau, T)
        sdd = self.lspb_sdd(t, tau, T)
        return s, sd, sdd

    def poly_coeff(self,
                   qi: float,
                   qf: float,
                   tf: float = 1.,
                   vi: float = 0.,
                   vf: float = 0.,
                   ai: float = 0.,
                   af: float = 0.) -> np.ndarray:
        """ Quintic polynomial coefficient
        A quintic (5th order) polynomial is used with default zero boundary
        conditions for velocity and acceleration.

        s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0

        Args:
        qi (float): initial pose
        qf (float): final pose
        T (float): final time. Default to 1.
        vi (float, optional): initial velocity. Default to 0
        vf (float, optional): final velocity. Default to 0
        ai (float, optional): initial acceleration. Default to 0
        af (float, optional): final acceleration. Default to 0


        Returns:
        a (np.ndarray): list with coefficients [a0, a1, a2, a3, a4, a5]
        """

        M = np.matrix([[1.0, 0, 0, 0, 0, 0],
                       [0, 1.0, 0, 0, 0, 0],
                       [0, 0, 2.0, 0, 0, 0],
                       [1.0, tf, tf**2, tf**3, tf**4, tf**5],
                       [0, 1.0, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                       [0, 0, 2.0, 6 * tf, 12 * tf**2, 20 * tf**3]])

        b = np.matrix([[qi], [vi], [ai], [qf], [vf], [af]])

        a = np.linalg.inv(M) @ b

        return a

    def poly(self, t: float, a: np.matrix) -> tuple[float, np.matrix]:
        """ Quintic polynomial coefficient
        A quintic (5th order) polynomial is used with default zero boundary
        conditions for velocity and acceleration.

        s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0
        sd(t) = 5 * a5 * t^4 + 4 * a4 * t^3 + 3 * a3 * t^2 + 2 * a2 * t^1 + a1
        sdd(t) = 20 * a5 * t^3 + 12 * a4 * t^2 + 6 * a3 * t + 2 * a2

        Args:
        t (float): current time
        a (np.matrix): coefficients


        Returns:
        s (float): s(t) value
        sd (float): sd(t) value
        sdd (float): sdd(t) value
        """
        T = np.matrix([1.0, t, t**2, t**3, t**4, t**5])
        Td = np.matrix([0, 1.0, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
        Tdd = np.matrix([0, 0, 2.0, 6 * t, 12 * t**2, 20 * t**3])
        s = T @ a
        sd = Td @ a
        sdd = Tdd @ a
        return np.asscalar(s), np.asscalar(sd), np.asscalar(sdd)


if __name__ == '__main__':
    tl = TimeLaw()
    a = tl.poly_coeff(qi=0.0, qf=1.0)
    dt = 0.001
    n = int(1 / dt)
    s = np.zeros(n)
    sd = np.zeros(n)
    sdd = np.zeros(n)
    for i, t in enumerate(np.arange(start=0, stop=1, step=dt)):
        s[i], sd[i], sdd[i] = tl.poly(t, a)
    tau, T = tl.lspb_param(delta_q=1, max_v=2.5, max_a=10)
    n = int((tau + T) / dt) + 1
    st = np.zeros(n)
    std = np.zeros(n)
    stdd = np.zeros(n)
    for i, t in enumerate(np.arange(start=0, stop=dt + tau + T, step=dt)):
        st[i], std[i], stdd[i] = tl.lspb(t=t, tau=tau, T=T)
    plt.figure(1)
    plt.plot(s)
    plt.plot(sd)
    plt.plot(sdd)
    plt.grid(True)
    plt.title("Polynomic time law")
    plt.figure(2)
    plt.plot(st)
    plt.plot(std)
    plt.plot(stdd)
    plt.title("LSPB")
    plt.grid(True)
    plt.show()
