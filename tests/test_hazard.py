import numpy as np
import pytest
from pytest import approx

from hazrisk import hazard as haz
from hazrisk import risk

K0, K1, K2 = 0.002, 2.4, 0.15


@pytest.fixture
def im():
    """A log-spaced set of intensity measure levels."""
    return np.logspace(-2, 0.5, 40)


@pytest.fixture
def first_order_curve(im):
    """A hazard curve that is exactly first order, so the fit must recover K0, K1."""
    return im, haz.first_order_approx(im, K0, K1)


@pytest.fixture
def second_order_curve(im):
    """A hazard curve that is exactly second order."""
    return im, haz.second_order_approx(im, K0, K1, K2)


# --- round-trip recovery of the generating coefficients ----------------------


def test_first_order_points_recovers_coefficients(first_order_curve):
    im, H_im = first_order_curve
    k0, k1 = haz.fit_first_order_points(im, H_im, iml=[0.05, 0.5])
    assert (k0, k1) == approx((K0, K1))


def test_second_order_points_recovers_coefficients(second_order_curve):
    """Anchors taken from the curve itself are exact; see
    ``test_off_grid_anchors_are_close`` for the interpolated case."""
    im, H_im = second_order_curve
    k0, k1, k2 = haz.fit_second_order_points(im, H_im, iml=im[[5, 20, 35]])
    assert (k0, k1, k2) == approx((K0, K1, K2))


def test_first_order_regression_recovers_coefficients(first_order_curve):
    im, H_im = first_order_curve
    k0, k1, r2 = haz.fit_first_order_regression(im, H_im)
    assert (k0, k1) == approx((K0, K1))
    assert r2 == approx(1.0)


def test_second_order_regression_recovers_coefficients(second_order_curve):
    im, H_im = second_order_curve
    k0, k1, k2, r2 = haz.fit_second_order_regression(im, H_im)
    assert (k0, k1, k2) == approx((K0, K1, K2))
    assert r2 == approx(1.0)


def test_second_order_regression_of_first_order_curve_gives_zero_k2(first_order_curve):
    """A power law is a second order curve with k2 = 0."""
    im, H_im = first_order_curve
    k0, k1, k2, r2 = haz.fit_second_order_regression(im, H_im)
    assert (k0, k1) == approx((K0, K1))
    assert k2 == approx(0.0, abs=1e-9)
    assert r2 == approx(1.0)


# --- the fitted curve passes through the requested points -------------------


def test_points_fit_passes_through_the_anchors(second_order_curve):
    """The fit reproduces the hazard at the anchors, as log-log interpolated
    from the supplied curve."""
    im, H_im = second_order_curve
    anchors = np.array([0.05, 0.2, 1.0])
    k0, k1, k2 = haz.fit_second_order_points(im, H_im, iml=anchors)

    expected = np.exp(np.interp(np.log(anchors), np.log(im), np.log(H_im)))
    fitted = haz.second_order_approx(anchors, k0, k1, k2)
    assert fitted == approx(expected, rel=1e-9)


# --- iml and mafe anchoring are equivalent ----------------------------------


def test_iml_and_mafe_anchoring_agree(second_order_curve):
    im, H_im = second_order_curve
    anchors = im[[5, 20, 35]]
    mafe = haz.second_order_approx(anchors, K0, K1, K2)

    by_iml = haz.fit_second_order_points(im, H_im, iml=anchors)
    by_mafe = haz.fit_second_order_points(im, H_im, mafe=mafe)
    assert by_mafe == approx(by_iml)


def test_first_order_mafe_anchoring(first_order_curve):
    im, H_im = first_order_curve
    mafe = haz.first_order_approx(np.array([0.05, 0.5]), K0, K1)
    k0, k1 = haz.fit_first_order_points(im, H_im, mafe=mafe)
    assert (k0, k1) == approx((K0, K1))


# --- off-grid anchors are interpolated in log-log space ---------------------


def test_off_grid_anchors_are_close():
    """Anchors between the supplied levels are log-log interpolated. That is a
    straight-line interpolation of a curve with log-log curvature, so the
    recovered coefficients carry a small interpolation error rather than being
    exact; on a well-discretised curve it stays well under a percent."""
    im = np.logspace(-2, 0.5, 40)
    H_im = haz.second_order_approx(im, K0, K1, K2)

    off_grid = np.sqrt(im[:-1] * im[1:])[[5, 20, 34]]  # midpoints in log space
    assert not np.any(np.isin(off_grid, im))

    k0, k1, k2 = haz.fit_second_order_points(im, H_im, iml=off_grid)
    assert (k0, k1, k2) == approx((K0, K1, K2), rel=1e-2)


def test_off_grid_anchors_are_exact_for_a_first_order_curve():
    """A first order curve is a straight line in log-log space, so log-log
    interpolation of it is exact and off-grid anchors lose nothing."""
    im = np.logspace(-2, 0.5, 6)
    H_im = haz.first_order_approx(im, K0, K1)

    off_grid = np.sqrt(im[:-1] * im[1:])[[0, 3]]
    k0, k1 = haz.fit_first_order_points(im, H_im, iml=off_grid)
    assert (k0, k1) == approx((K0, K1))


# --- band selection ---------------------------------------------------------


def test_iml_and_mafe_bands_select_the_same_points(second_order_curve):
    im, H_im = second_order_curve
    low, high = im[10], im[25]
    by_iml = haz.fit_second_order_regression(im, H_im, iml_band=(low, high))
    by_mafe = haz.fit_second_order_regression(im, H_im, mafe_band=(H_im[25], H_im[10]))
    assert by_mafe == approx(by_iml)


def test_band_is_inclusive_of_its_bounds(second_order_curve):
    im, H_im = second_order_curve
    x, _ = haz._band_points(im, H_im, (im[10], im[25]), None, 3)
    assert x.size == 16
    assert np.exp(x[[0, -1]]) == approx([im[10], im[25]])


def test_narrow_band_improves_the_first_order_fit(second_order_curve):
    """A power law cannot follow the curvature of a second order curve, so
    restricting the band to the rare end must improve the fit there."""
    im, H_im = second_order_curve
    _, _, r2_all = haz.fit_first_order_regression(im, H_im)
    _, _, r2_band = haz.fit_first_order_regression(im, H_im, mafe_band=(1e-4, 1e-2))
    assert r2_band > r2_all
    assert r2_all < 1.0


def test_band_bounds_may_be_given_either_way_round(second_order_curve):
    im, H_im = second_order_curve
    forwards = haz.fit_first_order_regression(im, H_im, iml_band=(0.05, 0.5))
    backwards = haz.fit_first_order_regression(im, H_im, iml_band=(0.5, 0.05))
    assert backwards == approx(forwards)


# --- error cases ------------------------------------------------------------


def test_both_iml_and_mafe_raises(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="exactly one"):
        haz.fit_first_order_points(im, H_im, iml=[0.05, 0.5], mafe=[1e-3, 1e-4])


def test_neither_iml_nor_mafe_raises(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="exactly one"):
        haz.fit_first_order_points(im, H_im)


def test_wrong_number_of_points_raises(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="expected 2 fitting points"):
        haz.fit_first_order_points(im, H_im, iml=[0.05, 0.2, 0.5])
    with pytest.raises(ValueError, match="expected 3 fitting points"):
        haz.fit_second_order_points(im, H_im, iml=[0.05, 0.5])


def test_repeated_points_raise(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="distinct"):
        haz.fit_first_order_points(im, H_im, iml=[0.05, 0.05])


def test_anchor_outside_the_curve_raises(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="within the supplied curve"):
        haz.fit_first_order_points(im, H_im, iml=[0.05, 10.0])
    with pytest.raises(ValueError, match="within the supplied curve"):
        haz.fit_first_order_points(im, H_im, mafe=[1e-2, 1e-12])


def test_both_bands_raise(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="at most one"):
        haz.fit_first_order_regression(
            im, H_im, iml_band=(0.05, 0.5), mafe_band=(1e-4, 1e-2)
        )


def test_too_few_points_in_band_raises(second_order_curve):
    im, H_im = second_order_curve
    with pytest.raises(ValueError, match="at least 3 are needed"):
        haz.fit_second_order_regression(im, H_im, iml_band=(im[5], im[6]))


# --- trailing zeros (as OpenQuake writes them) ---------------------------------

# Each fit with its fitting points or band, as keyword arguments.
FITS_WITH_ARGUMENTS = [
    (haz.fit_first_order_points, {"mafe": [1e-2, 1e-3]}),
    (haz.fit_first_order_points, {"iml": [0.05, 0.5]}),
    (haz.fit_second_order_points, {"mafe": [1e-1, 1e-2, 1e-3]}),
    (haz.fit_second_order_points, {"iml": [0.02, 0.2, 1.0]}),
    (haz.fit_first_order_regression, {}),
    (haz.fit_first_order_regression, {"mafe_band": (1e-5, 1e-2)}),
    (haz.fit_second_order_regression, {}),
    (haz.fit_second_order_regression, {"iml_band": (0.02, 2.0)}),
]


@pytest.mark.parametrize("fit, kwargs", FITS_WITH_ARGUMENTS)
def test_fits_drop_a_trailing_run_of_zeros(second_order_curve, fit, kwargs):
    """The same curve with three zero levels appended gives the same fit."""
    im, H_im = second_order_curve
    im_zero = np.append(im, im[-1] * np.array([1.5, 2.0, 3.0]))
    H_zero = np.append(H_im, [0.0, 0.0, 0.0])
    assert fit(im_zero, H_zero, **kwargs) == fit(im, H_im, **kwargs)


@pytest.mark.parametrize("fit, kwargs", FITS_WITH_ARGUMENTS)
def test_fits_reject_a_zero_inside_the_curve(second_order_curve, fit, kwargs):
    """A zero followed by a positive value is not a trailing run, so it raises."""
    im, H_im = second_order_curve
    H_hole = H_im.copy()
    H_hole[-5] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit(im, H_hole, **kwargs)


def test_fits_reject_curves_of_different_lengths(second_order_curve):
    """Trimming the zeros must not hide a length mismatch."""
    im, H_im = second_order_curve
    H_zero = np.append(H_im, [0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        haz.fit_first_order_regression(im, H_zero)


def test_non_monotonic_input_raises(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="strictly increasing"):
        haz.fit_first_order_regression(im[::-1], H_im)
    with pytest.raises(ValueError, match="strictly decreasing"):
        haz.fit_first_order_regression(im, H_im[::-1])


def test_mismatched_lengths_raise(first_order_curve):
    im, H_im = first_order_curve
    with pytest.raises(ValueError, match="same length"):
        haz.fit_first_order_regression(im[:-1], H_im)


# --- the fitted coefficients feed straight into risk.py ---------------------


def test_fitted_coefficients_are_ordered_for_the_risk_functions(second_order_curve):
    """The (k0, k1, k2) tuple must drop straight into the closed-form risk
    functions, and reproduce the numerical result for the same curve."""
    im, H_im = second_order_curve
    beta, theta_c = 0.4, 0.3

    k0, k1, k2, _ = haz.fit_second_order_regression(im, H_im)
    closed_form = risk.mafe_second_order(theta_c, beta, k0, k1, k2)

    fine_im = np.logspace(-2, 1.5, 4000)
    fine_H = haz.second_order_approx(fine_im, K0, K1, K2)
    from hazrisk.fragility import lognormal_fragility_curve

    numerical = risk.numerical_mafe(
        fine_H, lognormal_fragility_curve(fine_im, theta_c, beta)
    )

    assert closed_form == approx(numerical, rel=0.02)
