"""Hazard curves: analytical approximations, interpolation and conversions.

The hazard curve is the first link of the hazard -> fragility -> risk chain:
``H(im)`` is the mean annual frequency of exceeding each level of an intensity
measure ``im`` (for example spectral acceleration). This module provides

- the first-order (power law, Cornell et al. 2002) and second-order
  (log-quadratic, Vamvatsikos 2013) approximations of ``H(im)`` and their
  derivatives, which the closed-form solutions in :mod:`hazrisk.risk` need;
- functions that fit those approximations to a numerical hazard curve, either
  through chosen points or by regression over a band of the curve;
- PCHIP interpolation of a numerical hazard curve in log-log space, used by
  the numerical risk integration in :mod:`hazrisk.risk`;
- conversions between probability of exceedance in ``t`` years, mean annual
  frequency of exceedance and return period.

Tabulated curves (``im``, ``H_im``) must be ordered by increasing ``im``, so
``H_im`` decreases along the array.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from hazrisk._typing import as_float, scalar, vector

# The public interface of the module, in the order the docs list it.
__all__ = [
    "first_order_approx",
    "first_order_dHdx",
    "second_order_approx",
    "second_order_dHdx",
    "poe_to_mafe",
    "mafe_to_poe",
    "poe_to_rtp",
    "fit_first_order_points",
    "fit_second_order_points",
    "fit_first_order_regression",
    "fit_second_order_regression",
    "pchip_interpolant",
    "pchip_approx",
    "pchip_dHdx",
]

def first_order_approx(im: scalar|vector, k0: scalar, k1: scalar
                       ) -> scalar|vector:
    """First order hazard function."""
    # Cast arrays to float: numpy refuses negative powers of integer arrays.
    im, k0, k1 = (as_float(a) for a in (im, k0, k1))
    return k0 * im ** (-k1)


def first_order_dHdx(im: scalar|vector, k0: scalar, k1: scalar
                     ) -> scalar|vector:
    """Derivative of the first order hazard function."""
    im, k0, k1 = (as_float(a) for a in (im, k0, k1))
    return -k0 * k1 * im ** (-k1 - 1)


def second_order_approx(im: scalar|vector, k0: scalar, k1: scalar, k2: scalar
                        ) -> scalar|vector:
    """Second order hazard function."""
    im, k0, k1, k2 = (as_float(a) for a in (im, k0, k1, k2))
    return k0 * np.exp(-k2 * np.log(im) ** 2 - k1 * np.log(im))


def second_order_dHdx(im: scalar|vector, k0: scalar, k1: scalar, k2: scalar
                      ) -> scalar|vector:
    """Derivative of the second order hazard function.
    --> derived using the chain rule with:
                    g(s) = -k2 * log(s) ** 2 - k1 * log(s)
    """
    im, k0, k1, k2 = (as_float(a) for a in (im, k0, k1, k2))
    return -(2 * k2 * np.log(im) + k1) / im * second_order_approx(im, k0, k1, k2)


def poe_to_mafe(poe: scalar|vector, t: scalar=1) -> float:
    """ converts probability of exceedance in time span `t` to mean annual
    frequency of exceedance"""
    poe, t = as_float(poe), as_float(t)
    return -np.log(1-poe) / t


def mafe_to_poe(mafe: scalar|vector, t:scalar=1) -> float:
    """ converts mean annual frequency of exceedance to
    probability of exceedance in time span `t` """
    mafe, t = as_float(mafe), as_float(t)
    return 1 - np.exp((-mafe * t))


def poe_to_rtp(poe: scalar|vector, t: scalar=1) -> float:
    """ converts probability of exceedance in time span `t` to mean return period"""
    # poe_to_mafe casts the inputs to float.
    return 1 / poe_to_mafe(poe, t)


def _validate_curve(im: vector, H_im: vector) -> tuple[vector, vector]:
    """Check that a numerical hazard curve is usable for a log-log fit.

    Parameters
    ----------
    im : vector
        Intensity measure levels, ordered by increasing ``im``.
    H_im : vector
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im``.

    Returns
    -------
    tuple of vector
        ``im`` and ``H_im`` cast to float arrays.
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)

    if im.ndim != 1 or H_im.ndim != 1:
        raise ValueError("im and H_im must be one-dimensional")
    if im.size != H_im.size:
        raise ValueError(f"im and H_im must be the same length, got {im.size} "
                         f"and {H_im.size}")
    if np.any(im <= 0):
        raise ValueError("im must be strictly positive to fit in log space")
    if np.any(H_im <= 0):
        raise ValueError("H_im must be strictly positive to fit in log space; "
                         "trim the zero-hazard tail of the curve first")
    if np.any(np.diff(im) <= 0):
        raise ValueError("im must be strictly increasing")
    if np.any(np.diff(H_im) >= 0):
        raise ValueError("H_im must be strictly decreasing with increasing im")

    return im, H_im


def _x_from_iml(im: vector, H_im: vector, iml: vector) -> tuple[vector, vector]:
    """Interpolate the hazard curve at the requested intensity measure levels.

    The interpolation is linear in ``ln(im)`` vs ``ln(H)``, consistent with the
    space the hazard approximations are fitted in. Values outside the range of
    the supplied curve raise rather than extrapolate.

    Parameters
    ----------
    im, H_im : vector
        The numerical hazard curve.
    iml : vector
        Intensity measure levels at which to anchor the fit.

    Returns
    -------
    tuple of vector
        ``(x, y) = (ln(iml), ln(H(iml)))``.
    """
    iml = np.atleast_1d(np.asarray(iml, dtype=float))
    if np.any(iml <= 0):
        raise ValueError("fitting points given as iml must be strictly positive")
    if np.any(iml < im[0]) or np.any(iml > im[-1]):
        raise ValueError(f"fitting points must lie within the supplied curve, "
                         f"im = [{im[0]:g}, {im[-1]:g}]")

    x = np.log(iml)
    y = np.interp(x, np.log(im), np.log(H_im))
    return x, y


def _x_from_mafe(im: vector, H_im: vector, mafe: vector) -> tuple[vector, vector]:
    """Invert the hazard curve at the requested mean annual frequencies.

    As in :func:`_x_from_iml` the interpolation is linear in ``ln(im)`` vs
    ``ln(H)``, here solving for ``im`` given ``H``. ``np.interp`` requires an
    increasing sample-point array, so the curve is reversed: ``ln(H)`` decreases
    with increasing ``im``.

    Parameters
    ----------
    im, H_im : vector
        The numerical hazard curve.
    mafe : vector
        Mean annual frequencies of exceedance at which to anchor the fit.

    Returns
    -------
    tuple of vector
        ``(x, y) = (ln(im(mafe)), ln(mafe))``.
    """
    mafe = np.atleast_1d(np.asarray(mafe, dtype=float))
    if np.any(mafe <= 0):
        raise ValueError("fitting points given as mafe must be strictly positive")
    if np.any(mafe > H_im[0]) or np.any(mafe < H_im[-1]):
        raise ValueError(f"fitting points must lie within the supplied curve, "
                         f"H_im = [{H_im[-1]:g}, {H_im[0]:g}]")

    y = np.log(mafe)
    x = np.interp(y, np.log(H_im)[::-1], np.log(im)[::-1])
    return x, y


def _fitting_points(im: vector, H_im: vector, iml, mafe, n: int
                    ) -> tuple[vector, vector]:
    """Resolve the ``iml`` / ``mafe`` arguments into ``n`` log-log fitting points."""
    if (iml is None) == (mafe is None):
        raise ValueError("give exactly one of iml or mafe as the fitting points")

    if iml is not None:
        x, y = _x_from_iml(im, H_im, iml)
    else:
        x, y = _x_from_mafe(im, H_im, mafe)

    if x.size != n:
        raise ValueError(f"expected {n} fitting points, got {x.size}")
    if np.unique(x).size != n:
        raise ValueError("fitting points must be distinct")

    return x, y


def _band_points(im: vector, H_im: vector, iml_band, mafe_band, n: int
                 ) -> tuple[vector, vector]:
    """Select the portion of the curve to regress, in log-log coordinates.

    Parameters
    ----------
    im, H_im : vector
        The numerical hazard curve.
    iml_band, mafe_band : tuple of scalar, optional
        A ``(low, high)`` pair bounding the fitting range in intensity measure
        or in mean annual frequency terms (inclusive). At most one may be given;
        if neither is, the whole curve is used.
    n : int
        Minimum number of points required by the fit (``degree + 1``).

    Returns
    -------
    tuple of vector
        ``(x, y) = (ln(im), ln(H_im))`` over the selected band.
    """
    if iml_band is not None and mafe_band is not None:
        raise ValueError("give at most one of iml_band or mafe_band")

    if iml_band is not None:
        low, high = sorted(float(b) for b in iml_band)
        mask = (im >= low) & (im <= high)
        band = f"iml_band = ({low:g}, {high:g})"
    elif mafe_band is not None:
        low, high = sorted(float(b) for b in mafe_band)
        mask = (H_im >= low) & (H_im <= high)
        band = f"mafe_band = ({low:g}, {high:g})"
    else:
        mask = np.ones(im.size, dtype=bool)
        band = "the supplied curve"

    if mask.sum() < n:
        raise ValueError(f"{band} contains {mask.sum()} points of the curve, "
                         f"but at least {n} are needed for this fit")

    return np.log(im[mask]), np.log(H_im[mask])


def _fit_log_log(x: vector, y: vector, degree: int) -> vector:
    """Fit ``y = c0 + c1*x + ...`` of the given degree, ascending coefficients.

    With exactly ``degree + 1`` points the Vandermonde system is square and is
    solved exactly, so the fitted curve passes through every point. With more
    points it is a least-squares fit.
    """
    if x.size == degree + 1:
        return np.linalg.solve(np.vander(x, degree + 1, increasing=True), y)
    return np.polynomial.polynomial.polyfit(x, y, degree)


def _r_squared(x: vector, y: vector, coefficients: vector) -> float:
    """Coefficient of determination of a log-log fit, on the ``ln(H)`` residuals.

    This is measured in the space the least-squares fit actually minimises, so
    every decade of mean annual frequency carries equal weight.
    """
    residuals = y - np.polynomial.polynomial.polyval(x, coefficients)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def fit_first_order_points(im: vector, H_im: vector, iml=None, mafe=None
                           ) -> tuple[float, float]:
    """Fit the first order hazard approximation through two chosen points.

    The first order form (Cornell et al. 2002, the SAC/FEMA power law)

        H(im) = k0 * im ** (-k1)

    is a straight line in log-log space,

        ln(H) = ln(k0) - k1 * ln(im),

    so two points determine it exactly and the fitted curve passes through both.
    This is the usual way the approximation is anchored in practice: pick the two
    hazard levels that bracket the limit state of interest.

    Points that do not fall on a value of the supplied curve are interpolated
    linearly in ``ln(im)`` vs ``ln(H)``; points outside the curve raise rather
    than being extrapolated.

    Parameters
    ----------
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing).
    iml : vector, optional
        The two intensity measure levels to fit through.
    mafe : vector, optional
        The two mean annual frequencies of exceedance to fit through. Exactly
        one of ``iml`` and ``mafe`` must be given.

    Returns
    -------
    tuple of float
        ``(k0, k1)`` of the fitted first order approximation.
    """
    im, H_im = _validate_curve(im, H_im)
    x, y = _fitting_points(im, H_im, iml, mafe, 2)
    c0, c1 = _fit_log_log(x, y, 1)
    return float(np.exp(c0)), float(-c1)


def fit_second_order_points(im: vector, H_im: vector, iml=None, mafe=None
                            ) -> tuple[float, float, float]:
    """Fit the second order hazard approximation through three chosen points.

    The second order form of Vamvatsikos (2013), "Derivation of new SAC-FEMA
    performance evaluation solutions with second-order hazard approximation",

        H(im) = k0 * exp(-k2 * ln(im)**2 - k1 * ln(im))

    is a parabola in log-log space,

        ln(H) = ln(k0) - k1 * ln(im) - k2 * ln(im)**2,

    so three points determine it exactly and the fitted curve passes through all
    three. The extra ``k2`` term captures the curvature that a power law misses,
    which is why the anchor points should span the range of hazard of interest
    rather than the whole curve.


    Note that the fitted parabola turns over at ``ln(im) = -k1 / (2 * k2)`` and
    decreases in hazard on the other side of that maximum, which is not physical.
    That point normally sits below the fitted range, but the approximation should
    only be trusted over the range it was fitted to.

    Points that do not fall on a value of the supplied curve are interpolated
    linearly in ``ln(im)`` vs ``ln(H)``; points outside the curve raise rather
    than being extrapolated.

    Parameters
    ----------
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing).
    iml : vector, optional
        The three intensity measure levels to fit through.
    mafe : vector, optional
        The three mean annual frequencies of exceedance to fit through. Exactly
        one of ``iml`` and ``mafe`` must be given.

    Returns
    -------
    tuple of float
        ``(k0, k1, k2)`` of the fitted second order approximation.
    """
    im, H_im = _validate_curve(im, H_im)
    x, y = _fitting_points(im, H_im, iml, mafe, 3)
    c0, c1, c2 = _fit_log_log(x, y, 2)
    return float(np.exp(c0)), float(-c1), float(-c2)


def fit_first_order_regression(im: vector, H_im: vector, iml_band=None,
                               mafe_band=None) -> tuple[float, float, float]:
    """Least-squares fit of the first order hazard approximation over a band.

    Fits ``H(im) = k0 * im ** (-k1)`` (Cornell et al. 2002) by linear regression
    of ``ln(H)`` on ``ln(im)`` over the requested portion of the curve, rather
    than forcing the line through two chosen points. Restricting the band to the
    hazard levels of engineering interest matters: a power law fitted over the
    whole curve is usually a poor approximation at the rare end.

    ``r2`` is the coefficient of determination of the ``ln(H)`` residuals, i.e.
    measured in the space the fit is performed in, so all decades of mean annual
    frequency weigh equally.

    Parameters
    ----------
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing).
    iml_band : tuple of scalar, optional
        ``(low, high)`` intensity measure bounds of the fitting range, inclusive.
    mafe_band : tuple of scalar, optional
        ``(low, high)`` mean annual frequency bounds of the fitting range,
        inclusive. At most one of ``iml_band`` and ``mafe_band`` may be given; if
        neither is, the whole curve is fitted.

    Returns
    -------
    tuple of float
        ``(k0, k1, r2)`` of the fitted first order approximation.
    """
    im, H_im = _validate_curve(im, H_im)
    x, y = _band_points(im, H_im, iml_band, mafe_band, 2)
    coefficients = _fit_log_log(x, y, 1)
    c0, c1 = coefficients
    return float(np.exp(c0)), float(-c1), _r_squared(x, y, coefficients)


def fit_second_order_regression(im: vector, H_im: vector, iml_band=None,
                                mafe_band=None
                                ) -> tuple[float, float, float, float]:
    """Least-squares fit of the second order hazard approximation over a band.

    Fits ``H(im) = k0 * exp(-k2 * ln(im)**2 - k1 * ln(im))`` (Vamvatsikos 2013)
    by quadratic regression of ``ln(H)`` on ``ln(im)`` over the requested portion
    of the curve, rather than forcing the parabola through three chosen points.
    The quadratic term follows the curvature of a real hazard curve, so this
    typically holds over a wider band than the first order fit does.


    Note that the fitted parabola turns over at ``ln(im) = -k1 / (2 * k2)`` and
    decreases in hazard on the other side of that maximum, which is not physical.
    That point normally sits below the fitted range, but the approximation should
    only be trusted over the range it was fitted to.

    ``r2`` is the coefficient of determination of the ``ln(H)`` residuals, i.e.
    measured in the space the fit is performed in, so all decades of mean annual
    frequency weigh equally.

    Parameters
    ----------
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing).
    iml_band : tuple of scalar, optional
        ``(low, high)`` intensity measure bounds of the fitting range, inclusive.
    mafe_band : tuple of scalar, optional
        ``(low, high)`` mean annual frequency bounds of the fitting range,
        inclusive. At most one of ``iml_band`` and ``mafe_band`` may be given; if
        neither is, the whole curve is fitted.

    Returns
    -------
    tuple of float
        ``(k0, k1, k2, r2)`` of the fitted second order approximation.
    """
    im, H_im = _validate_curve(im, H_im)
    x, y = _band_points(im, H_im, iml_band, mafe_band, 3)
    coefficients = _fit_log_log(x, y, 2)
    c0, c1, c2 = coefficients
    return float(np.exp(c0)), float(-c1), float(-c2), _r_squared(x, y, coefficients)


def _trim_zero_tail(im: vector, H_im: vector) -> tuple[vector, vector]:
    """Drop the trailing non-positive values of a numerical hazard curve.

    OpenQuake writes a hazard of exactly zero above the truncation ceiling of the
    ground motion model, and also wherever the probability of exceedance falls
    below the precision it is stored at. Neither can be represented in log space,
    so the curve is cut at its last positive value. Only the *tail* is trimmed: a
    zero followed by a positive value is not a real hazard curve and is left for
    :func:`_validate_curve` to reject.

    Parameters
    ----------
    im : vector
        Intensity measure levels, ordered by increasing ``im``.
    H_im : vector
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im``.

    Returns
    -------
    tuple of vector
        ``im`` and ``H_im`` up to and including the last positive ``H_im``.
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)
    positive = np.flatnonzero(H_im > 0)
    if positive.size < 2:
        raise ValueError("H_im needs at least two positive values")
    return im[:positive[-1] + 1], H_im[:positive[-1] + 1]


def pchip_interpolant(im: vector, H_im: vector) -> PchipInterpolator:
    r"""PCHIP interpolant of a numerical hazard curve in log-log space.

    Returns the piecewise cubic Hermite interpolating polynomial (PCHIP;
    Fritsch & Carlson 1980) ``f`` of ``ln(H)`` against ``ln(im)``, so that

    .. math::

        H(im) = \exp\left(f(\ln im)\right).

    This is the object :func:`pchip_approx`, :func:`pchip_dHdx` and
    :func:`hazrisk.risk.numerical_mafe_pchip` evaluate. Building it once and
    evaluating it many times saves repeating the checks on the curve, and
    gives access to everything a :class:`scipy.interpolate.PchipInterpolator`
    offers (derivatives, the breakpoints ``f.x``, and so on).

    The interpolant works in log-log space: its argument is ``ln(im)`` and its
    value is ``ln(H)``. Use :func:`pchip_approx` for the hazard itself.

    Before the interpolant is built, trailing zeros of ``H_im`` (as OpenQuake
    writes them above the truncation level of the ground motion model, or
    below the precision the probability is stored at) are dropped, since
    ``ln(0)`` is not defined. The rest of the curve must be strictly positive
    and strictly decreasing, with ``im`` strictly increasing; otherwise a
    ``ValueError`` is raised.

    The interpolant does not extrapolate: outside ``ln(im[0])`` to ``ln`` of
    the last level with positive hazard it returns ``nan``.

    The hazard derivative follows from the chain rule. With ``f'`` the first
    derivative of the interpolant (``f.derivative()``, or ``f(x, 1)``),

    .. math::

        \frac{dH}{d\,im} = H(im) \, \frac{f'(\ln im)}{im},

    which is what :func:`pchip_dHdx` returns.

    Parameters
    ----------
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing
        apart from a trailing run of zeros).

    Returns
    -------
    scipy.interpolate.PchipInterpolator
        ``f`` with ``ln(H) = f(ln(im))``, built with ``extrapolate=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.array([0.1, 0.2, 0.5, 1.0])
    >>> H_im = np.array([1e-2, 3e-3, 4e-4, 5e-5])
    >>> f = hazard.pchip_interpolant(im, H_im)
    >>> round(float(np.exp(f(np.log(0.2)))), 12)  # the tabulated value
    0.003
    >>> x = np.log(0.3)
    >>> dHdx = np.exp(f(x)) * f(x, 1) / 0.3
    >>> bool(np.isclose(dHdx, hazard.pchip_dHdx(0.3, im, H_im)))
    True
    >>> bool(np.isnan(f(np.log(2.0))))  # outside the curve
    True
    """
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    return PchipInterpolator(np.log(im), np.log(H_im), extrapolate=False)


def pchip_approx(iml: scalar|vector, im: vector, H_im: vector) -> scalar|vector:
    """Numerical hazard curve interpolated by PCHIP in log-log space.

    The hazard is interpolated as ``ln(H)`` against ``ln(im)`` with a piecewise
    cubic Hermite interpolating polynomial (PCHIP; Fritsch & Carlson 1980,
    "Monotone piecewise cubic interpolation", SIAM J. Numer. Anal. 17(2),
    238-246). Between each pair of tabulated levels the curve is a cubic fixed by
    the two end values and the two end slopes, and the slope at each level is a
    weighted harmonic mean of the slopes of the neighbouring segments.

    Compared with the straight-line log-log interpolation used elsewhere in this
    module (:func:`_x_from_iml`):

    - the slope ``d(ln H)/d(ln im)`` is continuous across the tabulated levels,
      so the hazard derivative :func:`pchip_dHdx`, and any risk integrand built
      from it, has no steps at those levels;
    - the interpolant never overshoots the data, so the interpolated hazard is
      strictly decreasing wherever the tabulated hazard is. An ordinary cubic
      spline does not guarantee this.

    The values at the tabulated levels are reproduced exactly, and a curve that
    is already a straight line in log-log space (first order) is reproduced
    exactly everywhere.

    The interpolant is :func:`pchip_interpolant`: trailing zeros in ``H_im``
    are dropped, and the curve is not extrapolated: ``iml`` outside ``im[0]`` to the last
    positive level returns ``nan``.

    Parameters
    ----------
    iml : scalar or vector
        Intensity measure levels at which to evaluate the hazard.
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing
        apart from a trailing run of zeros).

    Returns
    -------
    scalar or vector
        The interpolated mean annual frequency of exceeding ``iml``.
    """
    interpolant = pchip_interpolant(im, H_im)
    iml = np.asarray(iml, dtype=float)
    return np.exp(interpolant(np.log(iml)))


def pchip_dHdx(iml: scalar|vector, im: vector, H_im: vector) -> scalar|vector:
    """Derivative of the PCHIP-interpolated hazard curve, ``dH/d(im)``.

    The interpolant is ``ln(H) = f(ln(im))`` (see :func:`pchip_approx`), so by
    the chain rule

        dH/d(im) = H(im) * f'(ln(im)) / im,

    with ``f'`` the analytical derivative of the piecewise cubic. It is negative
    and continuous. As for :func:`pchip_approx`, ``iml`` outside the curve
    returns ``nan``.

    Parameters
    ----------
    iml : scalar or vector
        Intensity measure levels at which to evaluate the derivative.
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im``.

    Returns
    -------
    scalar or vector
        The derivative of the mean annual frequency of exceedance with respect
        to ``im``, at ``iml``.
    """
    interpolant = pchip_interpolant(im, H_im)
    iml = np.asarray(iml, dtype=float)
    x = np.log(iml)
    # Chain rule on ln(H) = f(ln(im)): dH/d(im) = H * f'(ln im) / im.
    return np.exp(interpolant(x)) * interpolant(x, 1) / iml
