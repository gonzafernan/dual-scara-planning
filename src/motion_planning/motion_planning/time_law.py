import numpy as np


class TimeLaw(object):
    """" Class that implement time laws

    """

    def __init__(self) -> None:
        pass

    def trapezoidal_param(self, *args, **kwargs) -> tuple:
        """ Trapezoidal timing law parameters

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

    def trapezoidal_s(self, t: float, tau: float, T: float) -> float:
        """ Trapezoidal time law

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
        elif t <= tau:
            s = a * tau**2 / 2
        elif t > tau and t <= T:
            s_tau = a * tau**2 / 2
            s = s_tau + v * (t-tau)
        elif t > T and t <= T+tau:
            s_tau = a * tau**2 / 2
            s_T = s_tau + v * (T-tau)
            s = s_T + v * (t-T) - a * (t-T)**2 / 2
        elif t > T+tau:
            s = 1
        return s

    def trapezoidal_sd(self, t: float, tau: float, T: float) -> float:
        """ Derivate trapezoidal time law

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
        elif t > T and t <= T+tau:
            sd = v - a * (t-T)
        elif t > T+tau:
            sd = 0
        return sd

    def trapezoidal_sdd(self, t: float, tau: float, T: float) -> float:
        """ Second derivate trapezoidal time law

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
        elif t > T and t <= T+tau:
            sdd = -a
        elif t > T+tau:
            sdd = 0
        return sdd

    def quintic_poly_coeff(self, qi, qf, vi=0, vf=0, ai=0, af=0) -> np.ndarray:
        """ Quintic polynomial coefficient
        A quintic (5th order) polynomial is used with default zero boundary
        conditions for velocity and acceleration.
        Time is assumed to vary from 0 to 1.

        s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0

        Args:
        qi (float): initial pose
        qf (float): final pose
        vi (float, optional): initial velocity. Default to 0
        vf (float, optional): final velocity. Default to 0
        ai (float, optional): initial acceleration. Default to 0
        af (float, optional): final acceleration. Default to 0


        Returns:
        a (np.ndarray): list with coefficients [a0, a1, a2, a3, a4, a5]
        """
        T = 1.0

        M = np.matrix([[1.0, 0, 0, 0, 0, 0],
                       [0, 1.0, 0, 0, 0, 0],
                       [0, 0, 2.0, 0, 0, 0],
                       [1.0, T, T**2, T**3, T**4, T**5],
                       [0, 1.0, 2*T, 3*T**2, 4*T**3, 5*T**4],
                       [0, 0, 2.0, 6*T, 12*T**2, 20*T**3]])

        b = np.array([[qi], [vi], [ai], [qf], [vf], [af]])

        a = np.linalg.inv(M) @ b

        return a

    def quintic_poly_fun(self, t, a) -> tuple:
        """ Quintic polynomial coefficient
        A quintic (5th order) polynomial is used with default zero boundary
        conditions for velocity and acceleration.
        Time is assumed to vary from 0 to 1.

        s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0
        sd(t) = 5 * a5 * t^4 + 4 * a4 * t^3 + 3 * a3 * t^2 + 2 * a2 * t^1 + a1
        sdd(t) = 20 * a5 * t^3 + 12 * a4 * t^2 + 6 * a3 * t + 2 * a2

        Args:
        t (float): current time
        a (np.ndarray): coefficients


        Returns:
        s (float): s(t) value
        sd (float): sd(t) value
        sdd (float): sdd(t) value
        """
        T = np.array([1.0, t, t**2, t**3, t**4, t**5])
        Td = [0, 1.0, 2*t, 3*t**2, 4*t**3, 5*t**4]
        Tdd = [0, 0, 2.0, 6*t, 12*t**2, 20*t**3]
        s = T * a
        sd = Td * a
        sdd = Tdd * a
        return s, sd, sdd


if __name__ == '__main__':
    tl = TimeLaw()
    a = tl.quintic_poly_coeff(
        qi=0.0, qf=1.0, vi=0.0, vf=0.0, ai=0.0, af=0.0)
    print(a)
