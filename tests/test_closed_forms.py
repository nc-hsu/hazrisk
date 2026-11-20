"""Self-consistency tests of the closed forms.

- forward and inverse closed forms undo each other over a grid of inputs;
- the second-order forms reduce to the first-order ones at k2 = 0;
- the analytical hazard derivatives match finite differences.

The conversions between PoE, MAFE and return period are tested in
test_convert.py.
"""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from hazrisk import hazard, risk

# A grid covering typical fitted hazard curves and fragility dispersions.
K0S = (1e-5, 1e-4, 2e-3)
K1S = (1.5, 2.5, 3.5)
K2S = (0.0, 0.05, 0.15, 0.3)
BETAS = (0.2, 0.4, 0.6)
TARGETS = np.array([1e-5, 1e-4, 2e-4, 1e-3])


# --- inverse consistency ---------------------------------------------------


@pytest.mark.parametrize("k0, k1, beta", list(itertools.product(K0S, K1S, BETAS)))
def test_first_order_capacity_inverts_the_mafe(k0, k1, beta):
    theta = risk.required_capacity_first_order(TARGETS, beta, k0, k1)
    assert risk.mafe_first_order(theta, beta, k0, k1) == approx(TARGETS, rel=1e-12)


@pytest.mark.parametrize("k0, k1, beta", list(itertools.product(K0S, K1S, BETAS)))
def test_first_order_mafe_inverts_the_capacity(k0, k1, beta):
    theta = np.array([0.05, 0.2, 0.5, 1.5])
    mafe = risk.mafe_first_order(theta, beta, k0, k1)
    assert risk.required_capacity_first_order(mafe, beta, k0, k1) == approx(
        theta, rel=1e-12
    )


@pytest.mark.parametrize(
    "k0, k1, k2, beta", list(itertools.product(K0S, K1S, K2S, BETAS))
)
def test_second_order_capacity_inverts_the_mafe(k0, k1, k2, beta):
    # Only targets the fitted curve can reach: below the MAFE at the turnover
    # of the parabola (higher targets raise, see test_risk_checks.py).
    targets = TARGETS[TARGETS < _peak_mafe(k0, k1, k2, beta)]
    theta = risk.required_capacity_second_order(targets, beta, k0, k1, k2)
    assert risk.mafe_second_order(theta, beta, k0, k1, k2) == approx(targets, rel=1e-11)


def _turnover(k1, k2):
    """im at which the second-order hazard turns over, ln im = -k1 / (2 k2).

    Below it the fitted hazard increases with im, which is not physical; the
    closed forms are only meaningful above it. No turnover for k2 = 0.
    """
    return np.exp(-k1 / (2 * k2)) if k2 > 0 else 0.0


def _peak_mafe(k0, k1, k2, beta):
    """The largest MAFE the second-order closed form reaches (at the turnover)."""
    if k2 == 0:
        return np.inf
    return risk.mafe_second_order(_turnover(k1, k2), beta, k0, k1, k2)


@pytest.mark.parametrize(
    "k0, k1, k2, beta", list(itertools.product(K0S, K1S, K2S, BETAS))
)
def test_second_order_mafe_inverts_the_capacity(k0, k1, k2, beta):
    # Capacities above the turnover of the parabola, where the inverse is
    # the root that required_capacity_second_order returns.
    theta = np.array([0.05, 0.2, 0.5, 1.5])
    theta = theta[theta > _turnover(k1, k2)]
    mafe = risk.mafe_second_order(theta, beta, k0, k1, k2)
    assert risk.required_capacity_second_order(mafe, beta, k0, k1, k2) == approx(
        theta, rel=1e-10
    )


# --- k2 = 0 limits -----------------------------------------------------------


@pytest.mark.parametrize("beta", BETAS)
def test_p_is_one_at_k2_zero(beta):
    assert risk.p_second_order(0.0, beta) == 1.0


@pytest.mark.parametrize("k0, k1, beta", list(itertools.product(K0S, K1S, BETAS)))
def test_second_order_mafe_at_k2_zero_is_first_order(k0, k1, beta):
    theta = np.array([0.05, 0.2, 0.5, 1.5])
    assert risk.mafe_second_order(theta, beta, k0, k1, 0.0) == approx(
        risk.mafe_first_order(theta, beta, k0, k1), rel=1e-14
    )


@pytest.mark.parametrize("k0, k1, beta", list(itertools.product(K0S, K1S, BETAS)))
def test_second_order_capacity_at_k2_zero_is_first_order(k0, k1, beta):
    assert_array_equal(
        risk.required_capacity_second_order(TARGETS, beta, k0, k1, 0.0),
        risk.required_capacity_first_order(TARGETS, beta, k0, k1),
    )


def test_second_order_hazard_at_k2_zero_is_first_order():
    im = np.geomspace(0.01, 3.0, 20)
    assert hazard.second_order_approx(im, 1e-4, 2.5, 0.0) == approx(
        hazard.first_order_approx(im, 1e-4, 2.5), rel=1e-14
    )
    assert hazard.second_order_dHdx(im, 1e-4, 2.5, 0.0) == approx(
        hazard.first_order_dHdx(im, 1e-4, 2.5), rel=1e-14
    )


# --- derivatives against finite differences -----------------------------------


def _central_difference(f, x, rel_step=1e-6):
    """dH/d(im) by a central difference with a step relative to x."""
    h = x * rel_step
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("k1", K1S)
def test_first_order_derivative_matches_finite_differences(k1):
    im = np.geomspace(0.01, 3.0, 25)
    numeric = _central_difference(lambda x: hazard.first_order_approx(x, 1e-4, k1), im)
    assert hazard.first_order_dHdx(im, 1e-4, k1) == approx(numeric, rel=1e-7)


@pytest.mark.parametrize("k1, k2", list(itertools.product(K1S, K2S)))
def test_second_order_derivative_matches_finite_differences(k1, k2):
    im = np.geomspace(0.01, 3.0, 25)
    numeric = _central_difference(
        lambda x: hazard.second_order_approx(x, 1e-4, k1, k2), im
    )
    assert hazard.second_order_dHdx(im, 1e-4, k1, k2) == approx(numeric, rel=1e-7)
