import numpy as np
import pytest
import sys

sys.path.append('/home/jere/Documentos/five-bar-robot/src/' +
                'motion_planning/motion_planning')  # noqa
import timelaw as tl  # noqa


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
    tau, T = law.lspb_param(delta_q=data["delta_q"],
                            max_v=data["max_v"],
                            max_a=data["max_a"])
    assert tau == tau_ and T == T_


def test_s(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    s = law.lspb_s(t=tau_, tau=tau_, T=T_)
    s_ = (data["max_a"] * tau_**2 * 0.5) / data["delta_q"]
    assert np.round(s, decimals=4) == np.round(s_, decimals=4)
    s = law.lspb_s(t=T_ + tau_, tau=tau_, T=T_)
    s_ = 1.0
    assert s == s_
    s = law.lspb_s(t=0.0, tau=tau_, T=T_)
    s_ = 0.0
    assert s == s_


def test_sd(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    sd = 0.0
    # Integrating
    for t in np.arange(start=0, stop=T_ + tau_ + data["dt"], step=data["dt"]):
        sd = sd + law.lspb_sd(t=t, tau=tau_, T=T_)
    sd = sd * data["dt"]
    assert np.round(sd, decimals=4) == 1.0
    maxv = law.lspb_sd(t=tau_ + data["dt"], tau=tau_, T=T_)
    assert np.round(maxv, decimals=4) == np.round(1.0 / T_, decimals=4)


def test_sdd(data):
    tau_ = data["max_v"] / data["max_a"]
    T_ = data["delta_q"] / data["max_v"]
    # rounded
    tau_ = data["dt"] * np.ceil(tau_ / data["dt"])
    T_ = data["dt"] * np.ceil(T_ / data["dt"])
    law = tl.TimeLaw()
    sdd = 0.0
    # Integrating
    for t in np.arange(start=0, stop=tau_ + data["dt"], step=data["dt"]):
        sdd = sdd + law.lspb_sdd(t=t, tau=tau_, T=T_)
    sdd = sdd * data["dt"]
    assert np.round(sdd, decimals=4) == np.round(1.0 / T_, decimals=4)
    maxa = law.lspb_sdd(t=tau_ - data["dt"], tau=tau_, T=T_)
    assert np.round(maxa, decimals=4) == np.round(1.0 / (tau_ * T_), decimals=4)


def test_polynomic_law(data):
    law = tl.TimeLaw()
    a = law.poly_coeff(qi=0.0, qf=1.0, vi=0.0, vf=0.0, ai=0.0, af=0.0)
    coeff = np.array([[0.], [0.], [0.], [10.], [-15.], [6.]])
    for i, v in enumerate(a):
        assert np.round(v) == np.round(coeff[i])

    s, sd_current, sdd = law.poly(0, a)
    assert s == 0
    s, sd_current, sdd = law.poly(1, a)
    assert s <= 1.
    # Test for velocity
    sd = 0
    for t in np.arange(start=0, stop=1, step=data["dt"]):
        s, sd_current, sdd = law.poly(t, a)
        sd = sd + sd_current
    sd = sd * data["dt"]
    assert np.round(sd, decimals=4) == 1.
