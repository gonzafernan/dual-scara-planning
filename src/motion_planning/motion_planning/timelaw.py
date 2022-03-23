import numpy as np
import matplotlib.pyplot as plt


class TimeLaw(object):
    """ Class that implement time laws

    """

    def __init__(self) -> None:
        pass

    def lspb_param(self, *args, **kwargs) -> tuple:
        """ Trapezoidal timing law parameters or \
            linear segment with parabolic blend


        Args:
            delta_q (float): total displacement.
            max_v (float): maximal velocity.
            max_a (float): maximal acceleration.
            dt (float, optional): step time for round values. \
            Defaults to 0.001

        Returns:
            A tuple containing, respectively

            tau (float) acceleration time

            T (float) Time to start deceleration

        Note:
            If a trapezoidal timing law can not be reached, \
            with the parameters, this function keep doing a \
            trapezoidal timing law. This approach is achieve by \
            choosing :math:`\tau = \sqrt{\Delta q / a_{max}}` triangular law \ # noqa
            and :math:`T = v_{max} / a_{max}`. This keep a trapezoidal law
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
        """ Trapezoidal time law or Linear segment with parabolic blend. \
            Range from [0, 1]

        Args:
            t (float): current time
            T (float): time to start deceleration
            tau (float): acceleration time

        Returns:
            float s(t) value.
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
        elif t > T and t < T + tau:
            s_tau = a * tau**2 / 2
            s_T = s_tau + v * (T - tau)
            s = s_T + v * (t - T) - a * (t - T)**2 / 2
        elif t >= T + tau:
            s = 1
        return s

    def lspb_sd(self, t: float, tau: float, T: float) -> float:
        """ Derivative trapezoidal time law or \
            linear segment with parabolic blend

        Args:
            t (float): current time
            T (float): deceleration time
            tau (float): acceleration time

        Returns:
            float sd(t) value.
        """
        a = 1 / (T * tau)
        v = 1 / T
        if t <= 0:
            sd = 0
        elif t <= tau:
            sd = a * t
        elif t > tau and t <= T:
            sd = v
        elif t > T and t < T + tau:
            sd = v - a * (t - T)
        elif t >= T + tau:
            sd = 0
        return sd

    def lspb_sdd(self, t: float, tau: float, T: float) -> float:
        """ Second derivative trapezoidal time law or \
            Linear segment with parabolic blend

        Args:
            t (float): current time
            T (float): deceleration time
            tau (float): acceleration time

        Returns:
            float sdd(t) value.
        """
        a = 1 / (T * tau)
        if t <= 0:
            sdd = 0
        elif t <= tau:
            sdd = a
        elif t > tau and t <= T:
            sdd = 0
        elif t > T and t < T + tau:
            sdd = -a
        elif t >= T + tau:
            sdd = 0
        return sdd

    def lspb(self, t: float, tau: float,
             T: float) -> tuple[float, float, float]:
        """ Trapezoidal time law or \
            Linear segment with parabolic blend

        Args:
            t (float): current time
            T (float): deceleration time
            tau (float): acceleration time

        Returns:
            A tuple containing floats, respectively

            sd(t) value.

            sdd(t) value.

            sdd(t) value.
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
        """ Quintic polynomial coefficient. \
            A quintic (5th order) polynomial is used with default zero \
            boundary conditions for velocity and acceleration.

        .. math::
            s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0

        Args:
            qi (float): initial pose
            qf (float): final pose
            tf (float): final time. Default to 1.
            vi (float, optional): initial velocity. Default to 0
            vf (float, optional): final velocity. Default to 0
            ai (float, optional): initial acceleration. Default to 0
            af (float, optional): final acceleration. Default to 0

        Returns:
            np.ndarray coefficients [a0, a1, a2, a3, a4, a5]
        """

        M = np.array([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
                      [0, 0, 2.0, 0, 0, 0],
                      [1.0, tf, tf**2, tf**3, tf**4, tf**5],
                      [0, 1.0, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                      [0, 0, 2.0, 6 * tf, 12 * tf**2, 20 * tf**3]])

        b = np.array([[qi], [vi], [ai], [qf], [vf], [af]])

        a = np.linalg.inv(M) @ b

        return a

    def poly(self, t: float, a: np.array) -> tuple[float, np.array]:
        """ Quintic polynomial coefficient \
            A quintic (5th order) polynomial is used with default \
            zero boundary conditions for velocity and acceleration.

        .. math::
            s(t) = a5 * t^5 + a4 * t^4 + a3 * t^4 + a2 * t^2 + a1 * t + a0

            sd(t) = 5 * a5 * t^4 + 4 * a4 * t^3 + 3 * a3 * t^2 + 2 * a2 * t^1 \
            + a1

            sdd(t) = 20 * a5 * t^3 + 12 * a4 * t^2 + 6 * a3 * t + 2 * a2

        Args:
            t (float): current time
            a (np.array): coefficients

        Returns:
            A tuple containing floats, respectively

            s(t) value

            sd(t) value

            sdd(t) value
        """
        T = np.array([1.0, t, t**2, t**3, t**4, t**5])
        Td = np.array([0, 1.0, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
        Tdd = np.array([0, 0, 2.0, 6 * t, 12 * t**2, 20 * t**3])
        s = T @ a
        sd = Td @ a
        sdd = Tdd @ a
        return s.item(), sd.item(), sdd.item()


if __name__ == '__main__':
    tl = TimeLaw()
    a = tl.poly_coeff(qi=0.0, qf=1.0)
    dt = 0.01
    n = int(1 / dt) + 1
    time_poly = np.linspace(start=0, stop=1, num=n)
    s = np.zeros(n)
    sd = np.zeros(n)
    sdd = np.zeros(n)
    for i, t in enumerate(time_poly):
        s[i], sd[i], sdd[i] = tl.poly(t, a)
    tau, T = tl.lspb_param(delta_q=1, max_v=1, max_a=2)
    n = int((tau + T) / dt) + 1
    time_lspb = np.linspace(start=0, stop=tau + T, num=n)
    st = np.zeros(n)
    std = np.zeros(n)
    stdd = np.zeros(n)
    for i, t in enumerate(time_lspb):
        st[i], std[i], stdd[i] = tl.lspb(t=t, tau=tau, T=T)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(time_poly, s, label='$s$')
    ax1.plot(time_poly, sd, label='$\dot{s}$')  # noqa
    ax1.plot(time_poly, sdd, label='$\ddot{s}$')  # noqa
    ax1.grid(True)
    plt.title("Polynomial time law")
    ax1.set_xlabel('Time [s]')
    plt.legend()

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(time_lspb, st, label='$s$')
    ax2.plot(time_lspb, std, label='$\dot{s}$')  # noqa
    ax2.plot(time_lspb, stdd, label='$\ddot{s}$')  # noqa
    plt.title("LSPB")
    ax2.grid(True)
    ax2.set_xlabel('Time [s]')
    plt.legend()
    plt.show()
