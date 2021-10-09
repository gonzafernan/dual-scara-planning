import numpy as np
import pytest
import sys
sys.path.append('/home/jere/Documentos/five-bar-robot/src/'
        + 'motion_planning/motion_planning')  # noqa
import time_law as tl  # noqa


@pytest.fixture
def data():
    # delta_q = 5
    # max_v = 1
    # max_a = 1.5
    # dt = 0.001
    data = {"delta_q": 5, "max_v": 1, "max_a": 1.5, "dt": 0.001}
    return data


def test_parameters(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    tau, T = law.trapezoidal_param(
        delta_q=data["delta_q"], max_v=data["max_v"],
        max_a=data["max_a"])
    assert tau == tau_ and T == T_


def test_s(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    s = law.trapezoidal_s(t=tau_, tau=tau_, T=T_)
    s_ = (data["max_a"] * tau_**2 * 0.5) / data["delta_q"]
    assert np.round(s, decimals=4) == np.round(s_, decimals=4)


def test_sd(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    sd = 0.0
    # Integrating
    for t in np.arange(start=0, stop=T_+tau_+data["dt"], step=data["dt"]):
        sd = sd + law.trapezoidal_sd(t=t, tau=tau_, T=T_)
    sd = sd * data["dt"]
    assert np.round(sd, decimals=4) == 1.0
    maxv = law.trapezoidal_sd(t=tau_+data["dt"], tau=tau_, T=T_)
    assert np.round(maxv, decimals=4) == np.round(
        1.0 / T_, decimals=4)


def test_sdd(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    sdd = 0.0
    # Integrating
    for t in np.arange(start=0, stop=tau_+data["dt"], step=data["dt"]):
        sdd = sdd + law.trapezoidal_sdd(t=t, tau=tau_, T=T_)
    sdd = sdd * data["dt"]
    assert np.round(sdd, decimals=4) == np.round(1.0/T_, decimals=4)
    maxa = law.trapezoidal_sdd(t=tau_-data["dt"], tau=tau_, T=T_)
    assert np.round(maxa, decimals=4) == np.round(
        1.0 / (tau_ * T_), decimals=4)
