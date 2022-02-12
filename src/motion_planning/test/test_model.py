import numpy as np
import pytest
import fivebar as fb


@pytest.fixture
def five():
    l1 = 0.23
    l2 = 0.35
    b = 0.0
    five = fb.FiveBar(np.array([b, l1, l2]), np.array([b, l1, l2]))
    five.endPose = np.array([0.0, 0.4, 0.5])
    five.assembly = 1
    return five


def test_inverse(five):
    # ++
    value = np.array([2.62800598, 0.51358667, 0.5])
    for i, q in enumerate(five.ikine()):
        assert np.round(q, decimals=5) == np.round(value[i], decimals=5)
    # -+


def test_forward(five):
    # ++
    five.joints = np.array([2.62800598, 0.51358667, 0.5])
    assert AssertionError != np.testing.assert_allclose(
        five.fkine(), np.array([0.0, 0.4, 0.5]), rtol=1e-8, atol=1e-7)
    # -+
