r"""Hazard curves: analytical approximations, interpolation and conversions.

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
  the numerical risk integration in :mod:`hazrisk.risk`;
- conversions between probability of exceedance in ``t`` years, mean annual
  frequency of exceedance and return period.

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

    The parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` (for
    :math:`k_2 > 0`) and the hazard *increases* with ``im`` below that point,
    which is not physical. That point normally lies well below the fitted
    band, but the approximation should only be trusted over the band it was
    fitted to.

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
# Probability, frequency and return period
# ---------------------------------------------------------------------------


def poe_to_mafe(poe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Mean annual frequency of exceedance from a probability of exceedance.

    Earthquakes are taken to occur as a Poisson process, so the probability
    of at least one exceedance in ``t`` years is
    :math:`P = 1 - e^{-\lambda t}`, and

    .. math::

        \lambda = -\frac{\ln(1 - P)}{t}.

    Parameters
    ----------
    poe : array_like
        Probability of exceedance in ``t`` years, in [0, 1).
    t : array_like, optional
        Time span in years (default 1).

    Returns
    -------
    float or ndarray
        Mean annual frequency of exceedance (per year).

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    10 % in 50 years:

    >>> from hazrisk import hazard
    >>> print(f"{hazard.poe_to_mafe(0.10, t=50):.4e}")
    2.1072e-03
    """
    poe, t = as_float(poe), as_float(t)
    return -np.log(1 - poe) / t


def mafe_to_poe(mafe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Probability of exceedance in ``t`` years from a mean annual frequency.

    The inverse of :func:`poe_to_mafe`, for a Poisson process:

    .. math::

        P = 1 - e^{-\lambda t}.

    Parameters
    ----------
    mafe : array_like
        Mean annual frequency of exceedance (per year, >= 0).
    t : array_like, optional
        Time span in years (default 1).

    Returns
    -------
    float or ndarray
        Probability of at least one exceedance in ``t`` years.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    >>> from hazrisk import hazard
    >>> print(f"{hazard.mafe_to_poe(1 / 475, t=50):.4f}")
    0.0999
    """
    mafe, t = as_float(mafe), as_float(t)
    return 1 - np.exp(-mafe * t)


def poe_to_rtp(poe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Mean return period from a probability of exceedance in ``t`` years.

    The return period is the reciprocal of the mean annual frequency of
    exceedance (:func:`poe_to_mafe`):

    .. math::

        T_R = \frac{1}{\lambda} = -\frac{t}{\ln(1 - P)}.

    Parameters
    ----------
    poe : array_like
        Probability of exceedance in ``t`` years, in (0, 1).
    t : array_like, optional
        Time span in years (default 1).

    Returns
    -------
    float or ndarray
        Mean return period (years).

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    The familiar design levels, 10 % and 2 % in 50 years:

    >>> from hazrisk import hazard
    >>> print([round(float(T)) for T in hazard.poe_to_rtp([0.10, 0.02], t=50)])
    [475, 2475]
    """
    # poe_to_mafe casts the inputs to float.
    return 1 / poe_to_mafe(poe, t)


# ---------------------------------------------------------------------------
# Fitting the approximations to a numerical hazard curve
# ---------------------------------------------------------------------------


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
            "H_im must be strictly positive to fit in log space; "
            "trim the zero-hazard tail of the curve first"
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
    pick the two hazard levels that bracket the limit state of interest. The
    fit is only trustworthy between (and near) the two anchors.

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
        positive and strictly decreasing. Trim a zero tail first.
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
        If the curve is not strictly positive and ordered as above, if not
        exactly one of ``iml`` and ``mafe`` is given, if they do not hold two
        distinct points, or if a point lies outside the curve.

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
    im, H_im = _validate_curve(im, H_im)
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

    The fitted parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` and the
    hazard increases with ``im`` below that point, which is not physical.
    That point normally lies below the fitted range, but the approximation
    should only be trusted over the range it was fitted to.

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
        positive and strictly decreasing. Trim a zero tail first.
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
        If the curve is not strictly positive and ordered as above, if not
        exactly one of ``iml`` and ``mafe`` is given, if they do not hold three
        distinct points, or if a point lies outside the curve.

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
    im, H_im = _validate_curve(im, H_im)
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
        positive and strictly decreasing. Trim a zero tail first.
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
        If the curve is not strictly positive and ordered as above, if both
        bands are given, or if the band holds fewer than two levels.

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
    im, H_im = _validate_curve(im, H_im)
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

    The fitted parabola turns over at :math:`\ln im = -k_1 / (2 k_2)` and the
    hazard increases with ``im`` below that point, which is not physical.
    That point normally lies below the fitted range, but the approximation
    should only be trusted over the range it was fitted to.

    ``r2`` is the coefficient of determination of the :math:`\ln H` residuals
    (see :func:`fit_first_order_regression`).

    Parameters
    ----------
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        positive and strictly decreasing. Trim a zero tail first.
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
        If the curve is not strictly positive and ordered as above, if both
        bands are given, or if the band holds fewer than three levels.

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
    im, H_im = _validate_curve(im, H_im)
    x, y = _band_points(im, H_im, iml_band, mafe_band, 3)
    coefficients = _fit_log_log(x, y, 2)
    c0, c1, c2 = coefficients
    return float(np.exp(c0)), float(-c1), float(-c2), _r_squared(x, y, coefficients)


# ---------------------------------------------------------------------------
# PCHIP interpolation of a numerical hazard curve
# ---------------------------------------------------------------------------


def _trim_zero_tail(im: ArrayLike, H_im: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Drop the trailing non-positive values of a numerical hazard curve.

    OpenQuake writes a hazard of exactly zero above the truncation ceiling of the
    ground motion model, and also wherever the probability of exceedance falls
    below the precision it is stored at. Neither can be represented in log space,
    so the curve is cut at its last positive value. Only the *tail* is trimmed: a
    zero followed by a positive value is not a real hazard curve and is left for
    :func:`_validate_curve` to reject.

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
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)
    positive = np.flatnonzero(H_im > 0)
    if positive.size < 2:
        raise ValueError("H_im needs at least two positive values")
    return im[: positive[-1] + 1], H_im[: positive[-1] + 1]


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

    The hazard derivative follows from the chain rule. With :math:`f'` the
    first derivative of the interpolant (``f.derivative()``, or ``f(x, 1)``),

    .. math::

        \frac{dH}{d\,im} = H(im) \, \frac{f'(\ln im)}{im},

    which is what :func:`pchip_dHdx` returns.

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
    >>> x = np.log(0.3)
    >>> dHdx = np.exp(f(x)) * f(x, 1) / 0.3
    >>> bool(np.isclose(dHdx, hazard.pchip_dHdx(0.3, im, H_im)))
    True
    >>> bool(np.isnan(f(np.log(2.0))))  # outside the curve
    True
    """
    im, H_im = _validate_curve(*_trim_zero_tail(im, H_im))
    return PchipInterpolator(np.log(im), np.log(H_im), extrapolate=False)


def pchip_approx(iml: ArrayLike, im: ArrayLike, H_im: ArrayLike) -> FloatOrArray:
    r"""Numerical hazard curve interpolated by PCHIP in log-log space.

    Evaluates :math:`H(iml) = \exp(f(\ln iml))`, with :math:`f` the
    interpolant of :func:`pchip_interpolant`: a piecewise cubic Hermite
    interpolating polynomial (PCHIP; Fritsch & Carlson 1980) of
    :math:`\ln H` against :math:`\ln im`. Between each pair of tabulated
    levels the curve is a cubic fixed by the two end values and the two end
    slopes, and the slope at each level is a weighted harmonic mean of the
    slopes of the neighbouring segments.

    Compared with straight-line interpolation in log-log space:

    - the slope :math:`d(\ln H)/d(\ln im)` is continuous across the
      tabulated levels, so the hazard derivative :func:`pchip_dHdx`, and any
      risk integrand built from it, has no steps at those levels;
    - the interpolant never overshoots the data, so the interpolated hazard is
      strictly decreasing wherever the tabulated hazard is. An ordinary cubic
      spline does not guarantee this.

    The values at the tabulated levels are reproduced exactly, and a curve that
    is already a straight line in log-log space (first order) is reproduced
    exactly everywhere.

    Trailing zeros in ``H_im`` are dropped, and the curve is not
    extrapolated: ``iml`` outside ``im[0]`` to the last level with positive
    hazard returns ``nan``.

    Parameters
    ----------
    iml : array_like
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
        The interpolated mean annual frequency of exceeding ``iml`` (per
        year), with the shape of ``iml``.

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
    iml = np.asarray(iml, dtype=float)
    H_iml: FloatOrArray = np.exp(interpolant(np.log(iml)))
    return H_iml


def pchip_dHdx(iml: ArrayLike, im: ArrayLike, H_im: ArrayLike) -> FloatOrArray:
    r"""Derivative of the PCHIP-interpolated hazard curve, :math:`dH/d\,im`.

    The interpolant is :math:`\ln H = f(\ln im)` (see
    :func:`pchip_interpolant`), so by the chain rule

    .. math::

        \frac{dH}{d\,im} = H(im) \, \frac{f'(\ln im)}{im},

    with :math:`f'` the analytical derivative of the piecewise cubic. It is
    negative and continuous. As for :func:`pchip_approx`, ``iml`` outside
    the curve returns ``nan``.

    Parameters
    ----------
    iml : array_like
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
        :math:`dH/d\,im` at ``iml`` (per year per unit of ``im``), with the
        shape of ``iml``.

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
    """
    interpolant = pchip_interpolant(im, H_im)
    iml = np.asarray(iml, dtype=float)
    x = np.log(iml)
    # Chain rule on ln(H) = f(ln(im)): dH/d(im) = H * f'(ln im) / im.
    dHdx: FloatOrArray = np.exp(interpolant(x)) * interpolant(x, 1) / iml
    return dHdx
