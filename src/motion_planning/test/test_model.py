import numpy as np
import pytest
import sys
sys.path.append('/home/jere/Documentos/five-bar-robot/src/'
        + 'motion_planning/motion_planning')  # noqa
import fivebar as fb  # noqa


@pytest.fixture
def five():
    l = 0.205  # noqa
    b = 0.125
    five = fb.FiveBar(np.array([-b, l, l]), np.array([b, l, l]))
    five.endPose = np.array([0.0, 0.3, 0.0])
    five.assembly = 1
    return five


def test_inverse(five):
    # ++
    assert five.ikine() == pytest.approx(np.array(
        [0.52040625, 1.30998849, 0.0]))
    # -+
    five.arms[0].working = -1
    assert five.ikine() == pytest.approx(np.array(
        [1.83160416, 1.30998849, 0.0]))
    # --
    five.arms[1].working = -1
    assert five.ikine() == pytest.approx(np.array(
        [1.83160416, 2.6211864, 0.0]))
    # +-
    five.arms[0].working = 1
    assert five.ikine() == pytest.approx(np.array(
        [0.52040625, 2.6211864, 0.0]))


def test_forward(five):
    # ++
    five.joints = np.array([0.52040625, 1.30998849, 0.0])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3, 0.0]), rtol=1e-8, atol=1e-7)
    # -+
    five.arms[0].working = -1
    five.joints = np.array([1.83160416, 1.30998849, 0.0])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3, 0.0]), rtol=1e-8, atol=1e-7)
    # --
    five.arms[1].working = -1
    five.joints = np.array([1.83160416, 2.6211864, 0.0])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3, 0.0]), rtol=1e-8, atol=1e-7)
    # +-
    five.joints = np.array([0.52040625, 2.6211864, 0.0])
    five.arms[0].working = 1
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3, 0.0]), rtol=1e-8, atol=1e-7)


def test_forward_ass_2(five):
    # -+
    five.assembly = -1
    five.arms[0].working = -1
    five.joints = np.array([1.83160416, 1.30998849, 0.0])
    assert AssertionError != np.testing.assert_allclose(five.fkine(), np.array(
        [-1.98021962e-17, 9.61346135e-02, 0.0]), rtol=1e-5, atol=1e-5)
