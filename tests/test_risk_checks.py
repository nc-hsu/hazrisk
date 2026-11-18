"""Tests of risk_contribution, the ratio functions, the second-order inverse
and the input checks in hazrisk.risk.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from hazrisk import hazard, risk
from hazrisk.fragility import lognormal_fragility_curve

K0, K1, K2 = 0.002, 2.4, 0.15
THETA_C, BETA = 0.3, 0.4


@pytest.fixture
def im():
    """A coarse, log-spaced grid of levels, typical of a PSHA output."""
    return np.logspace(-3, 1, 25)


@pytest.fixture
def curve(im):
    """Second-order hazard and a lognormal fragility on the same levels."""
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    return im, H_im, lognormal_fragility_curve(im, THETA_C, BETA)


# --- risk_contribution --------------------------------------------------------


def test_contributions_add_up_to_numerical_mafe(curve):
    im, H_im, frag = curve
    contributions = risk.risk_contribution(im, H_im, frag, normalised=False)
    # Same terms, summed in a different order: equal up to rounding.
    assert contributions.sum() == approx(risk.numerical_mafe(H_im, frag), rel=1e-12)


def test_normalised_contributions_add_up_to_one(curve):
    im, H_im, frag = curve
    assert risk.risk_contribution(im, H_im, frag).sum() == approx(1.0, rel=1e-12)


def test_one_contribution_per_level_and_the_last_is_the_tail(curve):
    im, H_im, frag = curve
    contributions = risk.risk_contribution(im, H_im, frag, normalised=False)
    assert contributions.shape == im.shape
    assert contributions[-1] == H_im[-1] * frag[-1]
    # Interval 0 by hand: average fragility times the hazard drop.
    assert contributions[0] == approx(
        0.5 * (frag[0] + frag[1]) * (H_im[0] - H_im[1]), rel=1e-15
    )


def test_contributions_of_a_stack_of_fragilities(curve):
    im, H_im, _ = curve
    thetas = np.array([0.1, 0.3, 1.0])
    frag = lognormal_fragility_curve(im, thetas[:, None], BETA)
    contributions = risk.risk_contribution(im, H_im, frag, normalised=False)
    assert contributions.shape == frag.shape
    assert contributions.sum(axis=-1) == approx(
        risk.numerical_mafe(H_im, frag), rel=1e-12
    )
    assert risk.risk_contribution(im, H_im, frag).sum(axis=-1) == approx(1.0)


def test_contributions_converge_to_the_closed_form():
    """On a fine grid the total approaches the second-order closed form."""
    im = np.logspace(-3, 1, 400)
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    frag = lognormal_fragility_curve(im, THETA_C, BETA)
    total = risk.risk_contribution(im, H_im, frag, normalised=False).sum()
    assert total == approx(risk.mafe_second_order(THETA_C, K0, K1, K2, BETA), rel=1e-3)


def test_contributions_accept_a_trailing_run_of_zeros(curve):
    im, H_im, frag = curve
    H_im = H_im.copy()
    H_im[-3:] = 0.0
    contributions = risk.risk_contribution(im, H_im, frag, normalised=False)
    assert contributions[-3:] == approx(0.0)
    assert contributions.sum() == approx(risk.numerical_mafe(H_im, frag), rel=1e-12)


# --- ratio functions ----------------------------------------------------------


def test_risk_ratio_agrees_with_two_closed_form_mafes():
    theta_1, theta_2, beta_1, beta_2 = 0.3, 0.4, 0.3, 0.45
    ratio = risk.mafe_first_order(theta_2, K0, K1, beta_2) / risk.mafe_first_order(
        theta_1, K0, K1, beta_1
    )
    assert risk.risk_ratio_first_order(
        theta_2 / theta_1, K1, beta_1, beta_2 / beta_1
    ) == approx(ratio, rel=1e-13)


def test_theta_ratio_inverts_risk_ratio():
    R_theta = risk.theta_ratio_first_order(0.5, K1, 0.3, 1.2)
    assert risk.risk_ratio_first_order(R_theta, K1, 0.3, 1.2) == approx(0.5, rel=1e-13)


@pytest.mark.parametrize(
    "fn", [risk.risk_ratio_first_order, risk.theta_ratio_first_order]
)
def test_ratio_of_scalars_is_a_scalar(fn):
    assert np.ndim(fn(1.5, 2.5, 0.4, 1.2)) == 0
    assert np.ndim(fn(1.5, 2.5, 0.4)) == 0


@pytest.mark.parametrize(
    "fn", [risk.risk_ratio_first_order, risk.theta_ratio_first_order]
)
def test_ratio_broadcasts(fn):
    first = np.array([0.8, 1.0, 1.25])
    out = fn(first, 2.5, 0.4, 1.2)
    assert out.shape == (3,)
    assert out == approx([fn(x, 2.5, 0.4, 1.2) for x in first], rel=1e-15)
    # A column against a row gives the outer combination.
    assert fn(first[:, None], np.array([2.0, 3.0]), 0.4).shape == (3, 2)


def test_ratio_grid_with_np_ix_matches_elementwise_calls():
    """The nb 071 pattern: every combination of four vectors, one axis each."""
    R_lambda = np.linspace(0.5, 2.0, 16)
    k1 = np.array([1.5, 2.0, 2.5, 3.0])
    beta_1 = np.array([0.15, 0.2, 0.25, 0.4, 0.5])
    R_beta = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    grid = risk.theta_ratio_first_order(*np.ix_(R_lambda, k1, beta_1, R_beta))
    assert grid.shape == (16, 4, 5, 6)
    i, j, k, m = 3, 2, 1, 4
    assert grid[i, j, k, m] == risk.theta_ratio_first_order(
        R_lambda[i], k1[j], beta_1[k], R_beta[m]
    )


def test_ratio_of_one_dimensional_vectors_of_different_lengths_raises():
    """Plain broadcasting: unequal 1-D vectors do not combine implicitly."""
    with pytest.raises(ValueError):
        risk.theta_ratio_first_order([0.5, 1.0, 2.0], [2.0, 3.0], 0.4)


# --- required_capacity_second_order: k2 -> 0 ----------------------------------


def test_second_order_capacity_at_k2_zero_is_exactly_first_order():
    targets = np.array([1e-4, 2e-4, 1e-3])
    assert_array_equal(
        risk.required_capacity_second_order(targets, K0, K1, 0.0, BETA),
        risk.required_capacity_first_order(targets, K0, K1, BETA),
    )
    assert risk.required_capacity_second_order(
        2e-4, K0, K1, 0, BETA
    ) == risk.required_capacity_first_order(2e-4, K0, K1, BETA)


def test_second_order_capacity_is_accurate_for_tiny_k2():
    first = risk.required_capacity_first_order(2e-4, K0, K1, BETA)
    assert risk.required_capacity_second_order(2e-4, K0, K1, 1e-12, BETA) == (
        approx(first, rel=1e-10)
    )


@pytest.mark.parametrize("k2", [1e-8, 1e-4, 0.05, 0.15, 0.4])
def test_second_order_capacity_inverts_the_closed_form(k2):
    theta = risk.required_capacity_second_order(2e-4, K0, K1, k2, BETA)
    assert risk.mafe_second_order(theta, K0, K1, k2, BETA) == approx(2e-4, rel=1e-12)


# --- ValueErrors --------------------------------------------------------------


def test_numerical_mafe_rejects_a_reversed_curve(curve):
    _, H_im, frag = curve
    with pytest.raises(ValueError, match="must not increase"):
        risk.numerical_mafe(H_im[::-1], frag[::-1])


def test_numerical_mafe_rejects_a_length_mismatch(curve):
    _, H_im, frag = curve
    with pytest.raises(ValueError, match="last axis of fragility"):
        risk.numerical_mafe(H_im, frag[:-1])


def test_numerical_mafe_rejects_a_two_dimensional_hazard(curve):
    _, H_im, frag = curve
    with pytest.raises(ValueError, match="one-dimensional"):
        risk.numerical_mafe(np.stack([H_im, H_im]), frag)


def test_risk_contribution_rejects_a_reversed_curve(curve):
    im, H_im, frag = curve
    with pytest.raises(ValueError, match="strictly increasing"):
        risk.risk_contribution(im[::-1], H_im[::-1], frag[::-1])
    with pytest.raises(ValueError, match="must not increase"):
        risk.risk_contribution(im, H_im[::-1], frag)


def test_risk_contribution_rejects_length_mismatches(curve):
    im, H_im, frag = curve
    with pytest.raises(ValueError, match="length of H_im"):
        risk.risk_contribution(im[:-1], H_im, frag)
    with pytest.raises(ValueError, match="last axis of fragility"):
        risk.risk_contribution(im, H_im, frag[:-1])


@pytest.mark.parametrize(
    "call",
    [
        lambda k2, beta: risk.p_second_order(k2, beta),
        lambda k2, beta: risk.mafe_second_order(THETA_C, K0, K1, k2, beta),
        lambda k2, beta: risk.required_capacity_second_order(2e-4, K0, K1, k2, beta),
    ],
    ids=["p_second_order", "mafe_second_order", "required_capacity_second_order"],
)
def test_non_positive_p_denominator_raises(call):
    # 1 + 2 * (-10) * 0.4**2 = -2.2
    with pytest.raises(ValueError, match=r"1 \+ 2\*k2\*beta\*\*2 > 0.*1 of 1 "):
        call(-10.0, BETA)
    # One bad element out of three raises for the whole call.
    with pytest.raises(ValueError, match="1 of 3 element"):
        call(np.array([0.1, -10.0, 0.2]), BETA)


def test_unreachable_target_raises():
    # Above the peak of the fitted log-quadratic: no real capacity exists.
    with pytest.raises(ValueError, match=r"out of reach.*1 of 1"):
        risk.required_capacity_second_order(10.0, K0, K1, 0.5, BETA)
    with pytest.raises(ValueError, match=r"out of reach.*1 of 2"):
        risk.required_capacity_second_order(np.array([2e-4, 10.0]), K0, K1, 0.5, BETA)
