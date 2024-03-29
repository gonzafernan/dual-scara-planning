import numpy as np
import pytest
import Path as gp


@pytest.fixture
def gpath():

    class Gpath():
        path = gp.Path()
        st = np.array([-0.1, 0.3, 0.])
        gl = np.array([0.0, 0.35, 0.3])

    gpath = Gpath()
    return gpath


def test_lspb(gpath):
    pose = np.concatenate((gpath.st, gpath.gl), axis=0).reshape(2, 3)
    q, qd, qdd, p, pd, pdd = gpath.path.line(pose=pose, max_v=5, max_a=10)
    for i in range(0, 3):
        assert p[0, i] == gpath.st[i]
        assert np.round(p[-1, i], decimals=5) == np.round(gpath.gl[i],
                                                          decimals=5)


def test_poly(gpath):
    q, qd, qdd, p, pd, pdd = gpath.path.line_poly(start=gpath.st,
                                                  goal=gpath.gl,
                                                  mean_v=5)
    for i in range(0, 3):
        assert p[0, i] == gpath.st[i]
        assert np.round(p[-1, i], decimals=2) == np.round(gpath.gl[i],
                                                          decimals=2)


def test_go_to(gpath):
    gls = np.concatenate((gpath.st, gpath.gl), axis=0).reshape(2, 3)
    q, qd, qdd, p, pd, pdd = gpath.path.go_to(goals=gls, max_v=5, max_a=10)
    for i in range(0, 3):
        assert np.round(p[0, i]) == np.round(gpath.st[i])
        assert np.round(p[-1, i], decimals=3) == np.round(gpath.gl[i],
                                                          decimals=3)


def test_go_to_poly(gpath):
    q, qd, qdd, p, pd, pdd = gpath.path.go_to_poly(start=gpath.st,
                                                   goal=gpath.gl,
                                                   mean_v=0.5)
    for i in range(0, 3):
        assert np.round(p[0, i]) == np.round(gpath.st[i])
        assert np.round(p[-1, i], decimals=5) == np.round(gpath.gl[i],
                                                          decimals=5)
