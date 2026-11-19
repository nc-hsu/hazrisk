"""Tests of the public interface: __all__, pchip_interpolant and input types.

Two promises are checked here. First, that each module's ``__all__`` names
exactly the functions it exposes. Second, that integer arrays and Python lists
are accepted wherever float arrays are, and give the same numbers.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

import hazrisk
from hazrisk import convert, fragility, hazard, risk

K0, K1, K2 = 0.002, 2.4, 0.15
THETA_C, BETA = 0.3, 0.4


@pytest.fixture
def im():
    """A coarse, log-spaced grid of levels, typical of a PSHA output."""
    return np.logspace(-3, 1, 25)


# --- package and module interface -------------------------------------------


def test_package_exposes_the_four_modules_and_the_version():
    assert hazrisk.__all__ == ["hazard", "fragility", "risk", "convert", "__version__"]
    assert isinstance(hazrisk.__version__, str)


@pytest.mark.parametrize("module", [hazard, fragility, risk, convert])
def test_all_lists_every_public_function_and_nothing_else(module):
    """__all__ must match the module's own public functions (not imports)."""
    public = {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }
    assert sorted(module.__all__) == sorted(public)


# --- pchip_interpolant -------------------------------------------------------


def test_pchip_interpolant_reproduces_pchip_approx(im):
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    im_x = np.geomspace(im[0], im[-1], 300)
    f = hazard.pchip_interpolant(im, H_im)
    # Same object underneath, so the agreement is exact, not approximate.
    assert_array_equal(np.exp(f(np.log(im_x))), hazard.pchip_approx(im_x, im, H_im))


def test_pchip_interpolant_reproduces_pchip_dHdx(im):
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    im_x = np.geomspace(im[0], im[-1], 300)
    f = hazard.pchip_interpolant(im, H_im)
    x = np.log(im_x)
    dHdx = np.exp(f(x)) * f(x, 1) / im_x
    assert_array_equal(dHdx, hazard.pchip_dHdx(im_x, im, H_im))


def test_pchip_interpolant_works_in_log_log_space(im):
    """Its breakpoints are ln(im) and its values at them ln(H)."""
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    f = hazard.pchip_interpolant(im, H_im)
    assert f.x == approx(np.log(im), rel=1e-14)
    assert f(np.log(im)) == approx(np.log(H_im), rel=1e-12)


def test_pchip_interpolant_drops_trailing_zeros_and_does_not_extrapolate(im):
    H_im = hazard.first_order_approx(im, K0, K1)
    H_im[-3:] = 0.0
    f = hazard.pchip_interpolant(im, H_im)
    # The breakpoints stop at the last positive level.
    assert f.x[-1] == approx(np.log(im[-4]))
    assert np.isnan(f(np.log(im[-1])))
    assert np.isnan(f(np.log(im[0] / 2)))


def test_pchip_interpolant_rejects_an_increasing_curve(im):
    H_im = hazard.first_order_approx(im, K0, K1)
    with pytest.raises(ValueError, match="decreasing"):
        hazard.pchip_interpolant(im, H_im[::-1])


# --- integer arrays and lists give the same results as float arrays -----------

INT_IM = np.array([1, 2, 3, 5])


@pytest.mark.parametrize(
    "call",
    [
        lambda x: hazard.first_order_approx(x, K0, 2),
        lambda x: hazard.first_order_dHdx(x, K0, 2),
        lambda x: hazard.second_order_approx(x, K0, K1, K2),
        lambda x: hazard.second_order_dHdx(x, K0, K1, K2),
        lambda x: fragility.lognormal_fragility_curve(x, 2, BETA),
        lambda x: risk.mafe_first_order(x, BETA, K0, 2),
        lambda x: risk.mafe_second_order(x, BETA, K0, K1, K2),
    ],
    ids=[
        "first_order_approx",
        "first_order_dHdx",
        "second_order_approx",
        "second_order_dHdx",
        "lognormal_fragility_curve",
        "mafe_first_order",
        "mafe_second_order",
    ],
)
@pytest.mark.parametrize("as_input", [lambda a: a, list], ids=["int_array", "int_list"])
def test_integer_inputs_match_float_inputs(call, as_input):
    assert_array_equal(call(as_input(INT_IM)), call(INT_IM.astype(float)))


def test_required_capacity_first_order_accepts_a_list():
    targets = [1e-4, 2e-4, 1e-3]
    assert_array_equal(
        risk.required_capacity_first_order(targets, BETA, K0, K1),
        risk.required_capacity_first_order(np.array(targets), BETA, K0, K1),
    )


def test_numerical_mafe_and_pchip_accept_lists(im):
    H_im = hazard.second_order_approx(im, K0, K1, K2)
    frag = fragility.lognormal_fragility_curve(im, THETA_C, BETA)
    H_list, frag_list = np.asarray(H_im).tolist(), np.asarray(frag).tolist()
    assert risk.numerical_mafe(H_list, frag_list) == risk.numerical_mafe(H_im, frag)
    im_x = [0.01, 0.1, 1.0]
    assert_array_equal(
        hazard.pchip_approx(im_x, im.tolist(), H_list),
        hazard.pchip_approx(np.array(im_x), im, H_im),
    )


def test_ratio_functions_accept_integer_lists():
    assert_array_equal(
        risk.risk_ratio_first_order([1, 2], [2, 3], [1], [1]),
        risk.risk_ratio_first_order([1.0, 2.0], [2.0, 3.0], [1.0], [1.0]),
    )


def test_scalar_inputs_still_give_scalars():
    """Python numbers in, a scalar (not an array) out."""
    assert np.ndim(hazard.first_order_approx(0.3, K0, K1)) == 0
    assert np.ndim(risk.mafe_second_order(0.3, BETA, K0, K1, K2)) == 0
    assert isinstance(risk.p_second_order(K2, BETA), float)
    assert isinstance(hazard.first_order_approx(0.3, K0, K1), float)
