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
    im_x = np.geomspace(im[0], im[-1], 500)
    assert haz.pchip_approx(im_x, im, H_im) == approx(
        haz.first_order_approx(im_x, K0, K1), rel=1e-10
    )
    assert haz.pchip_dHdx(im_x, im, H_im) == approx(
        haz.first_order_dHdx(im_x, K0, K1), rel=1e-10
    )


def test_pchip_is_monotone_and_its_slope_continuous(im):
    """No overshoot between levels, and no step in the slope at the levels."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    im_x = np.geomspace(im[0], im[-1], 5000)
    assert np.all(np.diff(haz.pchip_approx(im_x, im, H_im)) < 0)

    eps = 1e-9
    level = im[12]
    slope = haz.pchip_dHdx(np.array([level * (1 - eps), level * (1 + eps)]), im, H_im)
    assert slope[0] == approx(slope[1], rel=1e-6)


def test_pchip_does_not_extrapolate(im):
    H_im = haz.first_order_approx(im, K0, K1)
    assert np.all(np.isnan(haz.pchip_approx([im[0] / 2, im[-1] * 2], im, H_im)))


def test_pchip_drops_a_trailing_zero_tail(im):
    H_im = haz.first_order_approx(im, K0, K1)
    H_zero = H_im.copy()
    H_zero[-4:] = 0.0
    im_x = np.geomspace(im[0], im[-5], 50)
    assert haz.pchip_approx(im_x, im, H_zero) == approx(
        haz.pchip_approx(im_x, im[:-4], H_im[:-4])
    )
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
        risk.mafe_first_order(THETA_C, BETA, K0, K1), rel=1e-3
    )


def test_numerical_mafe_pchip_matches_second_order_closed_form(im):
    """Vamvatsikos (2013): exact for a log-quadratic hazard and lognormal
    fragility. The coarse 25-level grid is interpolated, so a small error
    remains; it must stay well under a percent."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    assert risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA) == approx(
        risk.mafe_second_order(THETA_C, BETA, K0, K1, K2), rel=5e-3
    )


def test_pchip_beats_log_log_linear_on_a_curved_hazard(im):
    """The reason for the function: on the same coarse grid, refining the hazard
    with PCHIP is closer to the exact answer than refining it with straight
    lines in log-log space."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    exact = risk.mafe_second_order(THETA_C, BETA, K0, K1, K2)

    im_x = np.geomspace(im[0], im[-1], 2000)
    H_linear = np.exp(np.interp(np.log(im_x), np.log(im), np.log(H_im)))
    linear = risk.numerical_mafe(
        H_linear, lognormal_fragility_curve(im_x, THETA_C, BETA)
    )
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
            assert result[i, j] == approx(risk.numerical_mafe_pchip(t, im, H_im, b))


@pytest.mark.parametrize(
    "hazard_curve",
    [
        lambda x: haz.first_order_approx(x, K0, K1),
        lambda x: haz.second_order_approx(x, K0, K1, K2),
    ],
    ids=["first_order", "second_order"],
)
def test_numerical_mafe_pchip_integrates_down_to_the_first_zero(im, hazard_curve):
    """With trailing zeros, the step to the first zero level is integrated.

    The coarse curve ends in two zeros. Its PCHIP result must match
    numerical_mafe on a dense tabulation of the same curve that ends at the
    same zero level. A median capacity near the top of the curve makes the
    step to the zero matter (about 0.1-0.2 % of the total here).

    Tolerance rel=1e-4: the two differ by about 2e-5, the trapezoid error of
    the 2000-level refined grid. Dropping the step would differ by more than
    1e-3, which the second assertion checks.
    """
    theta_c = 5.0
    zero_levels = [15.0, 20.0]
    im_zero = np.append(im, zero_levels)
    H_zero = np.append(hazard_curve(im), [0.0, 0.0])
    with_zeros = risk.numerical_mafe_pchip(theta_c, im_zero, H_zero, BETA)

    # Dense tabulation up to the last positive level, then the first zero.
    dense = np.append(np.geomspace(im[0], im[-1], 20001), zero_levels[0])
    H_dense = np.append(hazard_curve(dense[:-1]), 0.0)
    reference = risk.numerical_mafe(
        H_dense, lognormal_fragility_curve(dense, theta_c, BETA)
    )
    assert with_zeros == approx(reference, rel=1e-4)

    # Without the zeros the tail term is used instead, which differs.
    without_zeros = risk.numerical_mafe_pchip(theta_c, im, hazard_curve(im), BETA)
    assert abs(with_zeros / without_zeros - 1) > 1e-3


def test_numerical_mafe_pchip_uses_only_the_first_zero(im):
    """Zeros after the first one add nothing."""
    H_im = haz.second_order_approx(im, K0, K1, K2)
    one_zero = risk.numerical_mafe_pchip(
        5.0, np.append(im, 15.0), np.append(H_im, 0.0), BETA
    )
    three_zeros = risk.numerical_mafe_pchip(
        5.0, np.append(im, [15.0, 20.0, 30.0]), np.append(H_im, [0.0, 0.0, 0.0]), BETA
    )
    assert three_zeros == one_zero


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
    im_x = np.geomspace(im[0], im[-1], 4000)
    density = risk.risk_contribution_pchip(
        im_x, THETA_C, im, H_im, BETA, normalised=False
    )
    tail = H_im[-1] * lognormal_fragility_curve(im[-1], THETA_C, BETA)
    area = np.trapezoid(density, np.log(im_x)) + tail
    assert area == approx(risk.numerical_mafe_pchip(THETA_C, im, H_im, BETA), rel=1e-4)


def test_normalised_contribution_integrates_to_one(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    im_x = np.geomspace(im[0], im[-1], 4000)
    share = risk.risk_contribution_pchip(im_x, THETA_C, im, H_im, BETA)
    assert np.trapezoid(share, np.log(im_x)) == approx(1.0, abs=1e-3)


def test_normalised_contribution_with_zeros_leaves_out_the_step_to_the_zero(im):
    """Evaluated up to the last positive level, the normalised contribution
    integrates to one minus the share of the step down to the first zero."""
    theta_c = 5.0
    H_im = haz.second_order_approx(im, K0, K1, K2)
    im_zero, H_zero = np.append(im, 15.0), np.append(H_im, 0.0)
    im_x = np.geomspace(im[0], im[-1], 4000)
    share = risk.risk_contribution_pchip(im_x, theta_c, im_zero, H_zero, BETA)

    total = risk.numerical_mafe_pchip(theta_c, im_zero, H_zero, BETA)
    fragility_ends = lognormal_fragility_curve(np.array([im[-1], 15.0]), theta_c, BETA)
    step = 0.5 * fragility_ends.sum() * H_im[-1]
    assert np.trapezoid(share, np.log(im_x)) == approx(1 - step / total, abs=1e-4)
    # Above the last positive level the contribution is not evaluated.
    assert np.isnan(risk.risk_contribution_pchip(12.0, theta_c, im_zero, H_zero, BETA))


def test_contribution_has_no_steps_at_the_tabulated_levels(im):
    H_im = haz.second_order_approx(im, K0, K1, K2)
    level = im[14]
    eps = 1e-9
    either_side = risk.risk_contribution_pchip(
        np.array([level * (1 - eps), level * (1 + eps)]),
        THETA_C,
        im,
        H_im,
        BETA,
        normalised=False,
    )
    assert either_side[0] == approx(either_side[1], rel=1e-6)
