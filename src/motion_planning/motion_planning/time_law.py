import numpy as np


class TimeLaw(object):
    """" Class that implement time laws

    """

    def __init__(self) -> None:
        pass

    def trapezoidal_param(self, *args, **kwargs):
        """ Timing law parameters
        Arg: TODO

        """
        if len(kwargs) != 0:
            delta_q = kwargs["dq"]
            v_max = kwargs["v_max"]
            a_max = kwargs["a_max"]
            if "dt" in kwargs:
                dt = kwargs["dt"]

        if len(args) != 0:
            delta_q = args[0]
            v_max = kwargs[1]
            a_max = kwargs[2]
            dt = 0.01
            if len(args) > 3:
                dt = kwargs[3]

            T = delta_q / v_max
            tau = v_max / a_max

            # Si tau es mayor a T --> elijo tau como una ley triangular y para
            # mantener el trapecio elijo T = qv/qa que es el valor que ya
            # obtuve anteriormente para mantener mi ley trapezoidal
            if T < tau:
                T = tau
                tau = np.sqrt(delta_q / a_max)
            # Redondeo de los parametros T y tau
            T = dt * np.ceil(T / dt)
            tau = dt * np.ceil(tau / dt)

            # Verifico T y tau no sean menores a Tmin y  taumin
            K = 4  # Factor que define Tmin y taumin admisible
            if T < K * dt:
                T = K * dt
            if tau < K * dt:
                tau = K * dt
            return (T, tau)

    def trapezoidal_s(self, t: float, T: float, tau: float) -> float:
        """ Trapezoidal time law

        Arg:
        t (float): current time
        T (float):
        tau (float): aceleration time
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

    def trapezoidal_ds(self, t: float, T: float, tau: float) -> float:
        """ Derivate trapezoidal time law

        Arg:
        t (float): current time
        T (float):
        tau (float): aceleration time
        """
        a = 1 / (T * tau)
        v = 1 / T
        if t <= 0:
            ds = 0
        elif t <= tau:
            ds = a * t
        elif t > tau and t <= T:
            ds = v
        elif t > T and t <= T+tau:
            ds = v - a * (t-T)
        elif t > T+tau:
            ds = 0
        return ds

    def trapezoidal_dds(self, t: float, T: float, tau: float) -> float:
        """ Second derivate trapezoidal time law

        Arg:
        t (float): current time
        T (float):
        tau (float): aceleration time
        """
        a = 1 / (T * tau)
        if t <= 0:
            dds = 0
        elif t <= tau:
            dds = a
        elif t > tau and t <= T:
            dds = 0
        elif t > T and t <= T+tau:
            dds = -a
        elif t > T+tau:
            dds = 0
        return dds
