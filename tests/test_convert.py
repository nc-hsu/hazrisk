"""Tests of hazrisk.convert: PoE, mean annual frequency and return period.

- every pair of opposite conversions undoes each other (round trips);
- the conversions agree with each other (PoE -> MAFE -> return period gives
  the same as PoE -> return period directly);
- textbook values: the 10 %, 5 % and 2 % in 50 years design levels and their
  rounded return periods of 475, 975 and 2475 years;
- lists are accepted wherever arrays are.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from hazrisk import convert

POES = np.array([1e-4, 0.02, 0.1, 0.5, 0.9])
MAFES = np.array([1e-6, 1e-4, 2e-3, 0.1])
RTPS = np.array([10.0, 475.0, 2475.0, 1e5])

# The design levels used in practice: probability in 50 years and the
# rounded return period they are quoted with.
DESIGN_LEVELS = [(0.10, 475), (0.05, 975), (0.02, 2475)]


# --- round trips ---------------------------------------------------------------


@pytest.mark.parametrize("t", [1, 50, 100])
def test_poe_and_mafe_round_trip(t):
    assert convert.mafe_to_poe(convert.poe_to_mafe(POES, t), t) == approx(
        POES, rel=1e-12
    )
    assert convert.poe_to_mafe(convert.mafe_to_poe(MAFES, t), t) == approx(
        MAFES, rel=1e-9
    )


@pytest.mark.parametrize("t", [1, 50, 100])
def test_poe_and_return_period_round_trip(t):
    assert convert.rtp_to_poe(convert.poe_to_rtp(POES, t), t) == approx(POES, rel=1e-12)
    assert convert.poe_to_rtp(convert.rtp_to_poe(RTPS, t), t) == approx(RTPS, rel=1e-9)


def test_mafe_and_return_period_round_trip():
    assert convert.rtp_to_mafe(convert.mafe_to_rtp(MAFES)) == approx(MAFES, rel=1e-15)
    assert convert.mafe_to_rtp(convert.rtp_to_mafe(RTPS)) == approx(RTPS, rel=1e-15)


# --- the conversions agree with each other -------------------------------------


@pytest.mark.parametrize("t", [1, 50])
def test_return_period_is_the_reciprocal_of_the_mafe(t):
    poe = np.array([0.02, 0.1, 0.5])
    assert convert.poe_to_rtp(poe, t) == approx(
        convert.mafe_to_rtp(convert.poe_to_mafe(poe, t))
    )


@pytest.mark.parametrize("t", [1, 50])
def test_rtp_to_poe_goes_through_the_mafe(t):
    assert convert.rtp_to_poe(RTPS, t) == approx(
        convert.mafe_to_poe(convert.rtp_to_mafe(RTPS), t), rel=1e-14
    )


# --- textbook values -----------------------------------------------------------


@pytest.mark.parametrize("poe, return_period", DESIGN_LEVELS)
def test_textbook_design_levels_in_50_years(poe, return_period):
    """10 %, 5 % and 2 % in 50 years: the familiar rounded return periods."""
    assert convert.poe_to_rtp(poe, 50) == approx(return_period, abs=0.5)


@pytest.mark.parametrize("poe, return_period", DESIGN_LEVELS)
def test_textbook_return_periods_give_the_design_probabilities(poe, return_period):
    """The rounded return periods give back the 50-year probabilities."""
    assert convert.rtp_to_poe(return_period, 50) == approx(poe, abs=1e-4)


def test_textbook_return_period_and_frequency():
    """The 2475-year return period is a frequency of about 4.04e-4 per year."""
    assert convert.rtp_to_mafe(2475) == approx(4.0404e-4, rel=1e-4)
    assert convert.mafe_to_rtp(2e-4) == approx(5000)


def test_poisson_relation_for_small_probabilities():
    """For a small PoE in one year, the PoE and the MAFE nearly coincide."""
    assert convert.poe_to_mafe(1e-6) == approx(1e-6, rel=1e-6)


# --- lists are accepted ----------------------------------------------------------


@pytest.mark.parametrize(
    "name, values",
    [
        ("poe_to_mafe", [0.02, 0.1, 0.5]),
        ("poe_to_rtp", [0.02, 0.1, 0.5]),
        ("mafe_to_poe", [1e-4, 2e-3, 1e-2]),
        ("rtp_to_poe", [475, 975, 2475]),
    ],
)
def test_conversions_with_a_time_span_accept_lists(name, values):
    fn = getattr(convert, name)
    assert_array_equal(fn(values, 50), fn(np.array(values), 50))


@pytest.mark.parametrize(
    "name, values",
    [("mafe_to_rtp", [1e-4, 2e-3, 1e-2]), ("rtp_to_mafe", [475, 975, 2475])],
)
def test_conversions_without_a_time_span_accept_lists(name, values):
    fn = getattr(convert, name)
    assert_array_equal(fn(values), fn(np.array(values, dtype=float)))
