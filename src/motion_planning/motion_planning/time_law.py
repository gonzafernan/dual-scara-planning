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
