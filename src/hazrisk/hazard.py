r"""Hazard curves: analytical approximations, fitting and interpolation.

The hazard curve is the first link of the hazard -> fragility -> risk chain:
:math:`H(im)` is the mean annual frequency of exceeding each level of an
intensity measure ``im`` (for example spectral acceleration). This module
provides

- the first-order (power law, Cornell et al. 2002) and second-order
  (log-quadratic, Vamvatsikos 2013) approximations of :math:`H(im)` and their
  derivatives, which the closed-form solutions in :mod:`hazrisk.risk` need;
- functions that fit those approximations to a numerical hazard curve, either
  through chosen points or by regression over a band of the curve;
- PCHIP interpolation of a numerical hazard curve in log-log space, used by
  the numerical risk integration in :mod:`hazrisk.risk`.

Conversions between probability of exceedance, mean annual frequency and
return period are in :mod:`hazrisk.convert`.

Tabulated curves (``im``, ``H_im``) must be ordered by increasing ``im``, so
``H_im`` decreases along the array. ``im`` can be in any unit; the hazard is
a mean annual frequency (per year).
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from hazrisk._typing import FloatArray, FloatOrArray, as_float

# The public interface of the module, in the order the docs list it.
__all__ = [
    "first_order_approx",
    "first_order_dHdx",
    "second_order_approx",
    "second_order_dHdx",
    "fit_first_order_points",
    "fit_second_order_points",
    "fit_first_order_regression",
    "fit_second_order_regression",
    "pchip_interpolant",
    "pchip_approx",
    "pchip_dHdx",
]


# ---------------------------------------------------------------------------
# Analytical approximations
# ---------------------------------------------------------------------------


def first_order_approx(im: ArrayLike, k0: ArrayLike, k1: ArrayLike) -> FloatOrArray:
    r"""First-order (power law) approximation of the hazard curve.

    .. math::

        H(im) = k_0 \, im^{-k_1}

    This is the form behind the SAC/FEMA closed-form risk solution (Cornell
    et al. 2002): a straight line of slope :math:`-k_1` in log-log space. It
    is usually only accurate over the band of the curve it was fitted to; see
    :func:`fit_first_order_points` and :func:`fit_first_order_regression`.

    Parameters
    ----------
    im : array_like
        Intensity measure levels (any unit, > 0).
    k0 : array_like
        Hazard at ``im = 1`` in the unit of ``im`` (per year).
    k1 : array_like
        Slope of the hazard curve in log-log space (> 0 for a decreasing
        curve).

    Returns
    -------
    float or ndarray
        Mean annual frequency of exceeding ``im`` (per year), with the
        broadcast shape of the inputs.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    >>> from hazrisk import hazard
    >>> print(f"{hazard.first_order_approx(0.5, k0=1e-4, k1=2.5):.4e}")
    5.6569e-04
    """
    # Cast arrays to float: numpy refuses negative powers of integer arrays.
    im, k0, k1 = (as_float(a) for a in (im, k0, k1))
    return k0 * im ** (-k1)


def first_order_dHdx(im: ArrayLike, k0: ArrayLike, k1: ArrayLike) -> FloatOrArray:
    r"""Derivative of the first-order hazard approximation with respect to ``im``.

    .. math::

        \frac{dH}{d\,im} = -k_0 k_1 \, im^{-k_1 - 1}

    It is negative, because the hazard decreases with ``im``. Parameters as
    in :func:`first_order_approx` (Cornell et al. 2002).

    Parameters
    ----------
    im : array_like
        Intensity measure levels (any unit, > 0).
    k0 : array_like
        Hazard at ``im = 1`` (per year).
    k1 : array_like
        Slope of the hazard curve in log-log space.

    Returns
    -------
    float or ndarray
        :math:`dH/d\,im` (per year per unit of ``im``), with the broadcast
        shape of the inputs.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    >>> from hazrisk import hazard
    >>> print(f"{hazard.first_order_dHdx(0.5, k0=1e-4, k1=2.5):.4e}")
    -2.8284e-03
    """
    im, k0, k1 = (as_float(a) for a in (im, k0, k1))
    return -k0 * k1 * im ** (-k1 - 1)


def second_order_approx(
    im: ArrayLike, k0: ArrayLike, k1: ArrayLike, k2: ArrayLike
) -> FloatOrArray:
    r"""Second-order (log-quadratic) approximation of the hazard curve.

    .. math::

        H(im) = k_0 \exp\left(-k_2 (\ln im)^2 - k_1 \ln im\right)

    the form of Vamvatsikos (2013): a parabola in log-log space. The
    :math:`k_2` term follows the curvature of a real hazard curve, so the
    fit holds over a wider band than the first-order one; with
    :math:`k_2 = 0` it is :func:`first_order_approx`.

    Parameters
    ----------
    im : array_like
        Intensity measure levels (any unit, > 0).
    k0 : array_like
        Hazard at ``im = 1`` in the unit of ``im`` (per year).
    k1 : array_like
        Slope of the hazard curve in log-log space at ``im = 1``.
    k2 : array_like
        Curvature of the hazard curve in log-log space.

    Returns
    -------
    float or ndarray
        Mean annual frequency of exceeding ``im`` (per year), with the
        broadcast shape of the inputs.

    Notes
    -----
    The parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` (for
    :math:`k_2 > 0`): below that point the hazard *increases* with ``im``,
    which is not physical. This function does not check for it. The point
    normally lies well below the band the curve was fitted to, and the
    approximation should only be used over that band.

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> from hazrisk import hazard
    >>> print(f"{hazard.second_order_approx(0.5, k0=1e-4, k1=2.5, k2=0.2):.4e}")
    5.1386e-04
    """
    im, k0, k1, k2 = (as_float(a) for a in (im, k0, k1, k2))
    return k0 * np.exp(-k2 * np.log(im) ** 2 - k1 * np.log(im))


def second_order_dHdx(
    im: ArrayLike, k0: ArrayLike, k1: ArrayLike, k2: ArrayLike
) -> FloatOrArray:
    r"""Derivative of the second-order hazard approximation with respect to ``im``.

    By the chain rule on :math:`H = k_0 \exp(g)`, with
    :math:`g = -k_2 (\ln im)^2 - k_1 \ln im`,

    .. math::

        \frac{dH}{d\,im} = -\frac{2 k_2 \ln im + k_1}{im} \, H(im).

    Parameters as in :func:`second_order_approx` (Vamvatsikos 2013).

    Parameters
    ----------
    im : array_like
        Intensity measure levels (any unit, > 0).
    k0, k1, k2 : array_like
        Coefficients of the second-order approximation.

    Returns
    -------
    float or ndarray
        :math:`dH/d\,im` (per year per unit of ``im``), with the broadcast
        shape of the inputs.

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> from hazrisk import hazard
    >>> print(f"{hazard.second_order_dHdx(0.5, k0=1e-4, k1=2.5, k2=0.2):.4e}")
    -2.2843e-03
    """
    im, k0, k1, k2 = (as_float(a) for a in (im, k0, k1, k2))
    return -(2 * k2 * np.log(im) + k1) / im * second_order_approx(im, k0, k1, k2)


# ---------------------------------------------------------------------------
# Fitting the approximations to a numerical hazard curve
# ---------------------------------------------------------------------------


def _trim_zero_tail(im: ArrayLike, H_im: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Drop the trailing non-positive values of a numerical hazard curve.

    OpenQuake writes a hazard of exactly zero above the truncation ceiling of the
    ground motion model, and also wherever the probability of exceedance falls
    below the precision it is stored at. Neither can be represented in log space,
    so the fitting functions and the PCHIP interpolant cut the curve at its last
    positive value. Only the *tail* is trimmed: a zero followed by a positive
    value is not a real hazard curve and is left for :func:`_validate_curve` to
    reject.

    Parameters
    ----------
    im : array_like
        Intensity measure levels, ordered by increasing ``im``.
    H_im : array_like
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im``.

    Returns
    -------
    tuple of ndarray
        ``im`` and ``H_im`` up to and including the last positive ``H_im``.

    Raises
    ------
    ValueError
        If ``im`` or ``H_im`` is not one-dimensional, their lengths differ, or
        fewer than two values of ``H_im`` are positive.
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)
    # Check the shapes before slicing: cutting both arrays at the same index
    # would otherwise hide a length mismatch from _validate_curve.
    if im.ndim != 1 or H_im.ndim != 1:
        raise ValueError("im and H_im must be one-dimensional")
    if im.size != H_im.size:
        raise ValueError(
            f"im and H_im must be the same length, got {im.size} and {H_im.size}"
        )
    positive = np.flatnonzero(H_im > 0)
    if positive.size < 2:
        raise ValueError("H_im needs at least two positive values")
    return im[: positive[-1] + 1], H_im[: positive[-1] + 1]


def _validate_curve(im: ArrayLike, H_im: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Check that a numerical hazard curve is usable for a log-log fit.

    Parameters
    ----------
    im : array_like
        Intensity measure levels, ordered by increasing ``im``.
    H_im : array_like
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im``.

    Returns
    -------
    tuple of ndarray
        ``im`` and ``H_im`` cast to float arrays.

    Raises
    ------
    ValueError
        If the curve is not one-dimensional, the lengths differ, any value is
        not positive, ``im`` is not strictly increasing or ``H_im`` not
        strictly decreasing.
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)

    if im.ndim != 1 or H_im.ndim != 1:
        raise ValueError("im and H_im must be one-dimensional")
    if im.size != H_im.size:
        raise ValueError(
            f"im and H_im must be the same length, got {im.size} and {H_im.size}"
        )
    if np.any(im <= 0):
        raise ValueError("im must be strictly positive to fit in log space")
    if np.any(H_im <= 0):
        raise ValueError(
            "H_im must be strictly positive to fit in log space; only a "
            "trailing run of zeros is allowed"
        )
    if np.any(np.diff(im) <= 0):
        raise ValueError("im must be strictly increasing")
    if np.any(np.diff(H_im) >= 0):
        raise ValueError("H_im must be strictly decreasing with increasing im")

    return im, H_im


def _x_from_iml(
    im: FloatArray, H_im: FloatArray, iml: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Interpolate the hazard curve at the requested intensity measure levels.

    The interpolation is linear in ``ln(im)`` vs ``ln(H)``, consistent with the
    space the hazard approximations are fitted in. Values outside the range of
    the supplied curve raise rather than extrapolate.

    Parameters
    ----------
    im, H_im : ndarray
        The numerical hazard curve.
    iml : array_like
        Intensity measure levels at which to anchor the fit.

    Returns
    -------
    tuple of ndarray
        ``(x, y) = (ln(iml), ln(H(iml)))``.
    """
    iml = np.atleast_1d(np.asarray(iml, dtype=float))
    if np.any(iml <= 0):
        raise ValueError("fitting points given as iml must be strictly positive")
    if np.any(iml < im[0]) or np.any(iml > im[-1]):
        raise ValueError(
            f"fitting points must lie within the supplied curve, "
            f"im = [{im[0]:g}, {im[-1]:g}]"
        )

    x = np.log(iml)
    y = np.interp(x, np.log(im), np.log(H_im))
    return x, y


def _x_from_mafe(
    im: FloatArray, H_im: FloatArray, mafe: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Invert the hazard curve at the requested mean annual frequencies.

    As in :func:`_x_from_iml` the interpolation is linear in ``ln(im)`` vs
    ``ln(H)``, here solving for ``im`` given ``H``. ``np.interp`` requires an
    increasing sample-point array, so the curve is reversed: ``ln(H)`` decreases
    with increasing ``im``.

    Parameters
    ----------
    im, H_im : ndarray
        The numerical hazard curve.
    mafe : array_like
        Mean annual frequencies of exceedance at which to anchor the fit.

    Returns
    -------
    tuple of ndarray
        ``(x, y) = (ln(im(mafe)), ln(mafe))``.
    """
    mafe = np.atleast_1d(np.asarray(mafe, dtype=float))
    if np.any(mafe <= 0):
        raise ValueError("fitting points given as mafe must be strictly positive")
    if np.any(mafe > H_im[0]) or np.any(mafe < H_im[-1]):
        raise ValueError(
            f"fitting points must lie within the supplied curve, "
            f"H_im = [{H_im[-1]:g}, {H_im[0]:g}]"
        )

    y = np.log(mafe)
    x = np.interp(y, np.log(H_im)[::-1], np.log(im)[::-1])
    return x, y


def _fitting_points(
    im: FloatArray,
    H_im: FloatArray,
    iml: ArrayLike | None,
    mafe: ArrayLike | None,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Resolve the ``iml`` / ``mafe`` arguments into ``n`` log-log fitting points."""
    if (iml is None) == (mafe is None):
        raise ValueError("give exactly one of iml or mafe as the fitting points")

    if iml is not None:
        x, y = _x_from_iml(im, H_im, iml)
    else:
        # mafe is not None here: exactly one of the two was given.
        assert mafe is not None
        x, y = _x_from_mafe(im, H_im, mafe)

    if x.size != n:
        raise ValueError(f"expected {n} fitting points, got {x.size}")
    if np.unique(x).size != n:
        raise ValueError("fitting points must be distinct")

    return x, y


def _band_points(
    im: FloatArray,
    H_im: FloatArray,
    iml_band: tuple[float, float] | None,
    mafe_band: tuple[float, float] | None,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Select the portion of the curve to regress, in log-log coordinates.

    Parameters
    ----------
    im, H_im : ndarray
        The numerical hazard curve.
    iml_band, mafe_band : tuple of float, optional
        A ``(low, high)`` pair bounding the fitting range in intensity measure
        or in mean annual frequency terms (inclusive). At most one may be given;
        if neither is, the whole curve is used.
    n : int
        Minimum number of points required by the fit (``degree + 1``).

    Returns
    -------
    tuple of ndarray
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
        raise ValueError(
            f"{band} contains {mask.sum()} points of the curve, "
            f"but at least {n} are needed for this fit"
        )

    return np.log(im[mask]), np.log(H_im[mask])


def _fit_log_log(x: FloatArray, y: FloatArray, degree: int) -> FloatArray:
    """Fit ``y = c0 + c1*x + ...`` of the given degree, ascending coefficients.

    With exactly ``degree + 1`` points the Vandermonde system is square and is
    solved exactly, so the fitted curve passes through every point. With more
    points it is a least-squares fit.
    """
    if x.size == degree + 1:
        return np.linalg.solve(np.vander(x, degree + 1, increasing=True), y)
    # polyfit is typed as returning a generic float array; it is float64 here.
    return np.asarray(np.polynomial.polynomial.polyfit(x, y, degree), dtype=np.float64)


def _r_squared(x: FloatArray, y: FloatArray, coefficients: FloatArray) -> float:
    """Coefficient of determination of a log-log fit, on the ``ln(H)`` residuals.

    This is measured in the space the least-squares fit actually minimises, so
    every decade of mean annual frequency carries equal weight.
    """
    residuals = y - np.polynomial.polynomial.polyval(x, coefficients)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def fit_first_order_points(
    im: ArrayLike,
    H_im: ArrayLike,
    iml: ArrayLike | None = None,
    mafe: ArrayLike | None = None,
) -> tuple[float, float]:
    r"""Fit the first-order hazard approximation through two chosen points.

    The first-order form (Cornell et al. 2002, the SAC/FEMA power law,
    :func:`first_order_approx`) is a straight line in log-log space,

    .. math::

        \ln H = \ln k_0 - k_1 \ln im,

    so two points determine it exactly and the fitted curve passes through
    both. This is the usual way the approximation is anchored in practice:
    pick the two hazard levels that bracket the limit state of interest.

    The points are given either as intensity levels (``iml``) or as mean
    annual frequencies (``mafe``). Points that do not fall on a value of the
    supplied curve are interpolated linearly in ``ln(im)`` vs ``ln(H)``;
    points outside the curve raise rather than being extrapolated.

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        positive and strictly decreasing, apart from a trailing run of zeros,
        which is dropped before fitting.
    iml : array_like, optional
        The two intensity measure levels to fit through.
    mafe : array_like, optional
        The two mean annual frequencies of exceedance to fit through. Exactly
        one of ``iml`` and ``mafe`` must be given.

    Returns
    -------
    tuple of float
        ``(k0, k1)`` of the fitted first-order approximation.

    Raises
    ------
    ValueError
        If the curve, without its trailing zeros, is not strictly positive
        and ordered as above, if not exactly one of ``iml`` and ``mafe`` is
        given, if they do not hold two distinct points, or if a point lies
        outside the curve.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    Anchor on the 475-year and 2475-year hazard levels of a curve that is
    exactly first order, which recovers its coefficients:

    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.geomspace(0.01, 3.0, 30)
    >>> H_im = hazard.first_order_approx(im, 1e-4, 2.5)
    >>> k0, k1 = hazard.fit_first_order_points(im, H_im, mafe=[1 / 475, 1 / 2475])
    >>> print(f"k0 = {k0:.3g}, k1 = {k1:.3g}")
    k0 = 0.0001, k1 = 2.5
    """
    # Trailing zeros carry no information for a log-log fit; drop them, then
    # check that what is left is a usable curve.
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    x, y = _fitting_points(im, H_im, iml, mafe, 2)
    c0, c1 = _fit_log_log(x, y, 1)
    return float(np.exp(c0)), float(-c1)


def fit_second_order_points(
    im: ArrayLike,
    H_im: ArrayLike,
    iml: ArrayLike | None = None,
    mafe: ArrayLike | None = None,
) -> tuple[float, float, float]:
    r"""Fit the second-order hazard approximation through three chosen points.

    The second-order form of Vamvatsikos (2013) (:func:`second_order_approx`)
    is a parabola in log-log space,

    .. math::

        \ln H = \ln k_0 - k_1 \ln im - k_2 (\ln im)^2,

    so three points determine it exactly and the fitted curve passes through
    all three. The extra :math:`k_2` term captures the curvature that a power
    law misses, which is why the anchor points should span the range of
    hazard of interest rather than the whole curve.

    Points that do not fall on a value of the supplied curve are interpolated
    linearly in ``ln(im)`` vs ``ln(H)``; points outside the curve raise rather
    than being extrapolated.

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        positive and strictly decreasing, apart from a trailing run of zeros,
        which is dropped before fitting.
    iml : array_like, optional
        The three intensity measure levels to fit through.
    mafe : array_like, optional
        The three mean annual frequencies of exceedance to fit through.
        Exactly one of ``iml`` and ``mafe`` must be given.

    Returns
    -------
    tuple of float
        ``(k0, k1, k2)`` of the fitted second-order approximation.

    Raises
    ------
    ValueError
        If the curve, without its trailing zeros, is not strictly positive
        and ordered as above, if not exactly one of ``iml`` and ``mafe`` is
        given, if they do not hold three distinct points, or if a point lies
        outside the curve.

    Notes
    -----
    The fitted parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` (for
    :math:`k_2 > 0`): below that point the hazard increases with ``im``, which
    is not physical. The fit does not check for it. The point normally lies
    below the fitted range, and the approximation should only be used over
    the range it was fitted to (see :func:`second_order_approx`).

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    Three tabulated levels of a curve that is exactly second order recover
    its coefficients. (Anchors between tabulated levels are interpolated
    with straight lines in log-log space, so on a curved hazard they give
    slightly different coefficients.)

    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.geomspace(0.01, 3.0, 30)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> k0, k1, k2 = hazard.fit_second_order_points(im, H_im, iml=im[[10, 18, 26]])
    >>> print(f"k0 = {k0:.3g}, k1 = {k1:.3g}, k2 = {k2:.3g}")
    k0 = 0.0001, k1 = 2.5, k2 = 0.2
    """
    # Trailing zeros carry no information for a log-log fit; drop them, then
    # check that what is left is a usable curve.
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    x, y = _fitting_points(im, H_im, iml, mafe, 3)
    c0, c1, c2 = _fit_log_log(x, y, 2)
    return float(np.exp(c0)), float(-c1), float(-c2)


def fit_first_order_regression(
    im: ArrayLike,
    H_im: ArrayLike,
    iml_band: tuple[float, float] | None = None,
    mafe_band: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    r"""Least-squares fit of the first-order hazard approximation over a band.

    Fits :math:`H = k_0 \, im^{-k_1}` (Cornell et al. 2002) by linear
    regression of :math:`\ln H` on :math:`\ln im` over the requested portion
    of the curve, rather than forcing the line through two chosen points.
    Restricting the band to the hazard levels of engineering interest
    matters: a power law fitted over the whole curve is usually a poor
    approximation at the rare end.

    ``r2`` is the coefficient of determination of the :math:`\ln H`
    residuals, i.e. measured in the space the fit is performed in, so every
    decade of mean annual frequency weighs equally:

    .. math::

        r^2 = 1 - \frac{\sum (\ln H_i - \ln \hat H_i)^2}
                       {\sum (\ln H_i - \overline{\ln H})^2}.

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        positive and strictly decreasing, apart from a trailing run of zeros,
        which is dropped before fitting.
    iml_band : tuple of float, optional
        ``(low, high)`` intensity measure bounds of the fitting range,
        inclusive.
    mafe_band : tuple of float, optional
        ``(low, high)`` mean annual frequency bounds of the fitting range,
        inclusive. At most one of ``iml_band`` and ``mafe_band`` may be given;
        if neither is, the whole curve is fitted.

    Returns
    -------
    tuple of float
        ``(k0, k1, r2)`` of the fitted first-order approximation.

    Raises
    ------
    ValueError
        If the curve, without its trailing zeros, is not strictly positive
        and ordered as above, if both bands are given, or if the band holds
        fewer than two levels.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    A power law fitted to a curved (second-order) hazard over the 10 to
    10 000 year band:

    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.geomspace(0.01, 3.0, 30)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> k0, k1, r2 = hazard.fit_first_order_regression(
    ...     im, H_im, mafe_band=(1e-4, 1e-1))
    >>> print(f"k0 = {k0:.3g}, k1 = {k1:.3g}, r2 = {r2:.4f}")
    k0 = 0.000174, k1 = 1.68, r2 = 0.9843
    """
    # Trailing zeros carry no information for a log-log fit; drop them, then
    # check that what is left is a usable curve.
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    x, y = _band_points(im, H_im, iml_band, mafe_band, 2)
    coefficients = _fit_log_log(x, y, 1)
    c0, c1 = coefficients
    return float(np.exp(c0)), float(-c1), _r_squared(x, y, coefficients)


def fit_second_order_regression(
    im: ArrayLike,
    H_im: ArrayLike,
    iml_band: tuple[float, float] | None = None,
    mafe_band: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    r"""Least-squares fit of the second-order hazard approximation over a band.

    Fits :math:`H = k_0 \exp(-k_2 (\ln im)^2 - k_1 \ln im)` (Vamvatsikos 2013)
    by quadratic regression of :math:`\ln H` on :math:`\ln im` over the
    requested portion of the curve, rather than forcing the parabola through
    three chosen points. The quadratic term follows the curvature of a real
    hazard curve, so this typically holds over a wider band than the
    first-order fit does.

    ``r2`` is the coefficient of determination of the :math:`\ln H` residuals
    (see :func:`fit_first_order_regression`).

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        positive and strictly decreasing, apart from a trailing run of zeros,
        which is dropped before fitting.
    iml_band : tuple of float, optional
        ``(low, high)`` intensity measure bounds of the fitting range,
        inclusive.
    mafe_band : tuple of float, optional
        ``(low, high)`` mean annual frequency bounds of the fitting range,
        inclusive. At most one of ``iml_band`` and ``mafe_band`` may be given;
        if neither is, the whole curve is fitted.

    Returns
    -------
    tuple of float
        ``(k0, k1, k2, r2)`` of the fitted second-order approximation.

    Raises
    ------
    ValueError
        If the curve, without its trailing zeros, is not strictly positive
        and ordered as above, if both bands are given, or if the band holds
        fewer than three levels.

    Notes
    -----
    The fitted parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` (for
    :math:`k_2 > 0`): below that point the hazard increases with ``im``, which
    is not physical. The fit does not check for it. The point normally lies
    below the fitted range, and the approximation should only be used over
    the range it was fitted to (see :func:`second_order_approx`).

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.geomspace(0.01, 3.0, 30)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> k0, k1, k2, r2 = hazard.fit_second_order_regression(
    ...     im, H_im, mafe_band=(1e-4, 1e-1))
    >>> print(f"k0 = {k0:.3g}, k1 = {k1:.3g}, k2 = {k2:.3g}, r2 = {r2:.4f}")
    k0 = 0.0001, k1 = 2.5, k2 = 0.2, r2 = 1.0000
    """
    # Trailing zeros carry no information for a log-log fit; drop them, then
    # check that what is left is a usable curve.
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    x, y = _band_points(im, H_im, iml_band, mafe_band, 3)
    coefficients = _fit_log_log(x, y, 2)
    c0, c1, c2 = coefficients
    return float(np.exp(c0)), float(-c1), float(-c2), _r_squared(x, y, coefficients)


# ---------------------------------------------------------------------------
# PCHIP interpolation of a numerical hazard curve
# ---------------------------------------------------------------------------


def pchip_interpolant(im: ArrayLike, H_im: ArrayLike) -> PchipInterpolator:
    r"""PCHIP interpolant of a numerical hazard curve in log-log space.

    Returns the piecewise cubic Hermite interpolating polynomial (PCHIP;
    Fritsch & Carlson 1980) :math:`f` of :math:`\ln H` against
    :math:`\ln im`, so that

    .. math::

        H(im) = \exp\left(f(\ln im)\right).

    This is the object :func:`pchip_approx`, :func:`pchip_dHdx` and
    :func:`hazrisk.risk.numerical_mafe_pchip` evaluate. Building it once and
    evaluating it many times saves repeating the checks on the curve, and
    gives access to everything a :class:`scipy.interpolate.PchipInterpolator`
    offers (derivatives, the breakpoints ``f.x``, and so on).

    The interpolant works in log-log space: its argument is :math:`\ln im`
    and its value is :math:`\ln H`. Use :func:`pchip_approx` for the hazard
    itself.

    Before the interpolant is built, trailing zeros of ``H_im`` (as OpenQuake
    writes them above the truncation level of the ground motion model, or
    below the precision the probability is stored at) are dropped, since
    :math:`\ln 0` is not defined. The rest of the curve must be strictly
    positive and strictly decreasing, with ``im`` strictly increasing.

    The interpolant does not extrapolate: outside :math:`\ln im_0` to the
    logarithm of the last level with positive hazard it returns ``nan``.

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        decreasing apart from a trailing run of zeros.

    Returns
    -------
    scipy.interpolate.PchipInterpolator
        :math:`f` with :math:`\ln H = f(\ln im)`, built with
        ``extrapolate=False``.

    Raises
    ------
    ValueError
        If fewer than two values of ``H_im`` are positive, or the curve
        (without its zero tail) is not strictly positive, ``im`` not strictly
        increasing or ``H_im`` not strictly decreasing.

    References
    ----------
    Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic
    interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.
    https://doi.org/10.1137/0717021

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.array([0.1, 0.2, 0.5, 1.0])
    >>> H_im = np.array([1e-2, 3e-3, 4e-4, 5e-5])
    >>> f = hazard.pchip_interpolant(im, H_im)
    >>> round(float(np.exp(f(np.log(0.2)))), 12)  # the tabulated value
    0.003
    >>> bool(np.isnan(f(np.log(2.0))))  # outside the curve
    True
    """
    # ln(0) is undefined, so the zero tail is dropped before the log-log fit.
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    return PchipInterpolator(np.log(im), np.log(H_im), extrapolate=False)


def pchip_approx(im_x: ArrayLike, im: ArrayLike, H_im: ArrayLike) -> FloatOrArray:
    r"""Numerical hazard curve interpolated by PCHIP in log-log space.

    Evaluates :math:`H(im_x) = \exp(f(\ln im_x))`, with :math:`f` the
    interpolant of :func:`pchip_interpolant`: a piecewise cubic Hermite
    interpolating polynomial (PCHIP; Fritsch & Carlson 1980) of
    :math:`\ln H` against :math:`\ln im`. Between each pair of tabulated
    levels the curve is a cubic fixed by the two end values and the two end
    slopes, and the slope at each level is a weighted harmonic mean of the
    slopes of the neighbouring segments.

    Unlike linear interpolation in log-log space the slope
    :math:`d(\ln H)/d(\ln im)` is continuous across the tabulated points.

    The interpolant never overshoots the data, so the interpolated hazard is
    strictly decreasing wherever the tabulated hazard is. An ordinary cubic
    spline does not guarantee this.

    The values at the tabulated points are reproduced exactly, and a curve that
    is already a straight line in log-log space (first order) is reproduced
    exactly everywhere.

    Trailing zeros in ``H_im`` are dropped, and the curve is not
    extrapolated: ``im_x`` outside ``im[0]`` to the last point with positive
    hazard returns ``nan``.

    Parameters
    ----------
    im_x : array_like
        Intensity measure levels at which to evaluate the hazard.
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        decreasing apart from a trailing run of zeros.

    Returns
    -------
    float or ndarray
        The interpolated mean annual frequency of exceeding ``im_x`` (per
        year), with the shape of ``im_x``.

    Raises
    ------
    ValueError
        As :func:`pchip_interpolant`, for an unusable curve.

    References
    ----------
    Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic
    interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.
    https://doi.org/10.1137/0717021

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.array([0.1, 0.2, 0.5, 1.0])
    >>> H_im = np.array([1e-2, 3e-3, 4e-4, 5e-5])
    >>> print(np.round(hazard.pchip_approx([0.15, 0.3, 2.0], im, H_im), 6))
    [0.00511  0.001319      nan]
    """
    interpolant = pchip_interpolant(im, H_im)
    im_x = np.asarray(im_x, dtype=float)
    # The interpolant maps ln(im) to ln(H), so evaluate in logs and exponentiate.
    H_im_x: FloatOrArray = np.exp(interpolant(np.log(im_x)))
    return H_im_x


def pchip_dHdx(im_x: ArrayLike, im: ArrayLike, H_im: ArrayLike) -> FloatOrArray:
    r"""Derivative of the PCHIP-interpolated hazard curve, :math:`dH/d\,im`.

    The interpolant is :math:`\ln H = f(\ln im)` (see
    :func:`pchip_interpolant`), i.e. :math:`H(im) = \exp(f(\ln im))`.
    Differentiating with the chain rule, the outer exponential gives
    :math:`H(im)`, the cubic gives its slope :math:`f'(\ln im)`, and
    :math:`d(\ln im)/d\,im = 1/im`:

    .. math::

        \frac{dH}{d\,im} = H(im) \, \frac{f'(\ln im)}{im}.

    :math:`f'` is the analytical derivative of the piecewise cubic,
    available from the interpolant as ``f.derivative()`` or ``f(x, 1)``. The
    result is negative and continuous. As for :func:`pchip_approx`, ``im_x``
    outside the curve returns ``nan``.

    Parameters
    ----------
    im_x : array_like
        Intensity measure levels at which to evaluate the derivative.
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        decreasing apart from a trailing run of zeros.

    Returns
    -------
    float or ndarray
        :math:`dH/d\,im` at ``im_x`` (per year per unit of ``im``), with the
        shape of ``im_x``.

    Raises
    ------
    ValueError
        As :func:`pchip_interpolant`, for an unusable curve.

    References
    ----------
    Fritsch, F. N. & Carlson, R. E. (1980). Monotone piecewise cubic
    interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.
    https://doi.org/10.1137/0717021

    Examples
    --------
    On a first-order curve PCHIP is exact, so the derivative matches
    :func:`first_order_dHdx`:

    >>> import numpy as np
    >>> from hazrisk import hazard
    >>> im = np.geomspace(0.01, 3.0, 30)
    >>> H_im = hazard.first_order_approx(im, 1e-4, 2.5)
    >>> exact = hazard.first_order_dHdx(0.5, 1e-4, 2.5)
    >>> bool(np.isclose(hazard.pchip_dHdx(0.5, im, H_im), exact))
    True

    The same value from the interpolant :math:`f` of
    :func:`pchip_interpolant` and its first derivative ``f(x, 1)``:

    >>> f = hazard.pchip_interpolant(im, H_im)
    >>> x = np.log(0.5)
    >>> dHdx = np.exp(f(x)) * f(x, 1) / 0.5
    >>> bool(np.isclose(dHdx, hazard.pchip_dHdx(0.5, im, H_im)))
    True
    """
    interpolant = pchip_interpolant(im, H_im)
    im_x = np.asarray(im_x, dtype=float)
    x = np.log(im_x)
    # Chain rule on ln(H) = f(ln(im)): dH/d(im) = H * f'(ln im) / im.
    dHdx: FloatOrArray = np.exp(interpolant(x)) * interpolant(x, 1) / im_x
    return dHdx
