import numpy as np
import pytest
from pytest import approx

from hazrisk import hazard as haz
from hazrisk import risk
from hazrisk.fragility import lognormal_fragility_curve


K0, K1, K2 = 0.002, 2.4, 0.15
THETA_C, BETA = 0.3, 0.4


@pytest.fixture
def im():
    """A coarse, log-spaced set of levels, typical of a PSHA output grid.

    Wide enough that the fragility is ~0 at the bottom and ~1 at the top, so the
    truncation of the integral at either end is negligible.
    """
    return np.logspace(-3, 1, 25)


# --- PCHIP hazard interpolation ---------------------------------------------

def test_pchip_passes_through_the_tabulated_points(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    assert haz.pchip_approx(im, im, H_im) == approx(H_im, rel=1e-12)


def test_pchip_is_exact_for_a_first_order_curve(im):
    """A power law is a straight line in log-log space; PCHIP reproduces it."""
    H_im = haz.first_order_approx(im, K0, K1)
    iml = np.geomspace(im[0], im[-1], 500)
    assert haz.pchip_approx(iml, im, H_im) == approx(
        haz.first_order_approx(iml, K0, K1), rel=1e-10)
    assert haz.pchip_dHdx(iml, im, H_im) == approx(
        haz.first_order_dHdx(iml, K0, K1), rel=1e-10)


def test_pchip_is_monotone_and_its_slope_continuous(im):
    """No overshoot between levels, and no step in the slope at the levels."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    iml = np.geomspace(im[0], im[-1], 5000)
    assert np.all(np.diff(haz.pchip_approx(iml, im, H_im)) < 0)

    eps = 1e-9
    level = im[12]
    slope = haz.pchip_dHdx(np.array([level * (1 - eps), level * (1 + eps)]),
                           im, H_im)
    assert slope[0] == approx(slope[1], rel=1e-6)


def test_pchip_does_not_extrapolate(im):
    H_im = haz.first_order_approx(im, K0, K1)
    assert np.all(np.isnan(haz.pchip_approx([im[0] / 2, im[-1] * 2], im, H_im)))


def test_pchip_drops_a_trailing_zero_tail(im):
    H_im = haz.first_order_approx(im, K0, K1)
    H_zero = H_im.copy()
    H_zero[-4:] = 0.0
    iml = np.geomspace(im[0], im[-5], 50)
    assert haz.pchip_approx(iml, im, H_zero) == approx(
        haz.pchip_approx(iml, im[:-4], H_im[:-4]))
    assert np.isnan(haz.pchip_approx(im[-2], im, H_zero))


def test_pchip_rejects_a_zero_inside_the_curve(im):
    H_im = haz.first_order_approx(im, K0, K1)
    H_im[10] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        haz.pchip_approx(im[5], im, H_im)


# --- numerical_mafe_pchip: literature (closed-form) benchmarks ---------------

def test_numerical_mafe_pchip_matches_first_order_closed_form(im):
    """Cornell et al. (2002): exact for a power-law hazard and lognormal fragility."""
    H_im = haz.first_order_approx(im, K0, K1)
    assert risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA) == approx(
        risk.mafe_first_order(THETA_C, K0, K1, BETA), rel=1e-3)


def test_numerical_mafe_pchip_matches_second_order_closed_form(im):
    """Vamvatsikos (2013): exact for a log-quadratic hazard and lognormal
    fragility. The coarse 25-level grid is interpolated, so a small error
    remains; it must stay well under a percent."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    assert risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA) == approx(
        risk.mafe_second_order(THETA_C, K0, K1, K2, BETA), rel=5e-3)


def test_pchip_beats_log_log_linear_on_a_curved_hazard(im):
    """The reason for the function: on the same coarse grid, refining the hazard
    with PCHIP is closer to the exact answer than refining it with straight
    lines in log-log space."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    exact = risk.mafe_second_order(THETA_C, K0, K1, K2, BETA)

    iml = np.geomspace(im[0], im[-1], 2000)
    H_linear = np.exp(np.interp(np.log(iml), np.log(im), np.log(H_im)))
    linear = risk.numerical_mafe(
        H_linear, lognormal_fragility_curve(iml, THETA_C, BETA))
    pchip = risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA)

    assert abs(pchip - exact) < abs(linear - exact)


# --- numerical_mafe_pchip: behaviour ----------------------------------------

def test_numerical_mafe_pchip_returns_a_float_for_scalar_input(im):
    H_im = haz.first_order_approx(im, K0, K1)
    assert isinstance(risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA), float)


def test_numerical_mafe_pchip_broadcasts_over_theta_and_beta(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    theta = np.array([0.2, 0.3, 0.5])
    beta = np.array([[0.2], [0.4]])
    result = risk.numerical_mafe_pchip(theta, im, H_im, beta)

    assert result.shape == (2, 3)
    for i, b in enumerate(beta[:, 0]):
        for j, t in enumerate(theta):
            assert result[i, j] == approx(
                risk.numerical_mafe_pchip(t, im, H_im, b))


def test_numerical_mafe_pchip_ignores_a_trailing_zero_tail(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    H_zero = np.append(H_im, [0.0, 0.0])
    im_ext = np.append(im, [im[-1] * 1.5, im[-1] * 2.0])
    assert risk.numerical_mafe_pchip(THETA_C, im_ext, H_zero, BETA) == approx(
        risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA))


def test_numerical_mafe_handles_a_stack_of_fragilities(im):
    """The tail term must index the last level of every curve in the stack."""
    H_im = haz.first_order_approx(im, K0, K1)
    stack = lognormal_fragility_curve(im, np.array([[0.2], [0.4]]), BETA)
    result = risk.numerical_mafe(H_im, stack)
    assert result == approx([risk.numerical_mafe(H_im, f) for f in stack])


# --- risk_contribution_pchip -------------------------------------------------

def test_contribution_integrates_to_the_mafe(im):
    """Area under d(lambda_C)/d(ln im) plus the tail term = numerical_mafe_pchip."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    iml = np.geomspace(im[0], im[-1], 4000)
    density = risk.risk_contribution_pchip(iml, THETA_C, im, H_im, BETA,
                                           normalised=False)
    tail = H_im[-1] * lognormal_fragility_curve(im[-1], THETA_C, BETA)
    area = np.trapezoid(density, np.log(iml)) + tail
    assert area == approx(risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA),
                          rel=1e-4)


def test_normalised_contribution_integrates_to_one(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    iml = np.geomspace(im[0], im[-1], 4000)
    share = risk.risk_contribution_pchip(iml, THETA_C, im, H_im, BETA)
    assert np.trapezoid(share, np.log(iml)) == approx(1.0, abs=1e-3)


def test_contribution_has_no_steps_at_the_tabulated_levels(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    level = im[14]
    eps = 1e-9
    either_side = risk.risk_contribution_pchip(
        np.array([level * (1 - eps), level * (1 + eps)]),
        THETA_C, im, H_im, BETA, normalised=False)
    assert either_side[0] == approx(either_side[1], rel=1e-6)
