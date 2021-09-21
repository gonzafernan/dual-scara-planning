import numpy as np
import five_bar_robot as fb
import pytest
import sys
sys.path.append('../motion_planning')


@pytest.fixture
def five():
    l = 0.205
    b = 0.125
    five = fb.FiveBar(np.array([-b, l, l]), np.array([b, l, l]))
    five.endEff = np.array([0.0, 0.3])
    five.assMode = 1
    return five


def test_inverse(five):
    # ++
    assert five.ikine() == pytest.approx(np.array([0.52040625, 1.30998849]))
    # -+
    five.arms[0].wkMode_ = -1
    assert five.ikine() == pytest.approx(np.array([1.83160416, 1.30998849]))
    # --
    five.arms[1].wkMode_ = -1
    assert five.ikine() == pytest.approx(np.array([1.83160416, 2.6211864]))
    # +-
    five.arms[0].wkMode_ = 1
    assert five.ikine() == pytest.approx(np.array([0.52040625, 2.6211864]))


def test_forward(five):
    # ++
    five.q = np.array([0.52040625, 1.30998849])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3]), rtol=1e-8, atol=1e-7)
    # -+
    five.arms[0].wkMode_ = -1
    five.q = np.array([1.83160416, 1.30998849])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3]), rtol=1e-8, atol=1e-7)
    # --
    five.arms[1].wkMode_ = -1
    five.q = np.array([1.83160416, 2.6211864])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3]), rtol=1e-8, atol=1e-7)
    # +-
    five.q = np.array([0.52040625, 2.6211864])
    five.arms[0].wkMode_ = 1
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.3]), rtol=1e-8, atol=1e-7)


def test_forward_ass_2(five):
    # -+
    five.assMode = -1
    five.arms[0].wkMode_ = -1
    five.q = np.array([1.83160416, 1.30998849])
    assert AssertionError != np.testing.assert_allclose(five.fkine(), np.array(
        [-1.98021962e-17, 9.61346135e-02]), rtol=1e-5, atol=1e-5)
