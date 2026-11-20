r"""Risk: the mean annual frequency of a limit state, and related quantities.

This is the last link of the hazard -> fragility -> risk chain. It combines a
hazard curve (:mod:`hazrisk.hazard`) with a fragility curve
(:mod:`hazrisk.fragility`) to give the mean annual frequency of exceeding a
limit state such as collapse (written :math:`\lambda_C` below; "MAFC" for
collapse). It provides

- the SAC/FEMA closed-form solutions for a lognormal fragility on the
  first-order (Cornell et al. 2002) and second-order (Vamvatsikos 2013)
  hazard approximations, in both directions: capacity -> MAFE (``mafe_*``)
  and target MAFE -> required median capacity (``required_capacity_*``);
- numerical integration of a fragility against a tabulated hazard curve
  (``numerical_mafe``, ``numerical_mafe_pchip``);
- where the risk comes from (``risk_contribution``,
  ``risk_contribution_pchip``);
- first-order ratios between two designs (``risk_ratio_first_order``,
  ``theta_ratio_first_order``).

Units: ``im`` and the median capacity ``theta_c`` in the same (any) unit;
frequencies per year. Tabulated inputs must be ordered by increasing ``im``,
so ``H_im`` decreases along the array; the numerical integration relies on it.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hazrisk import hazard as haz
from hazrisk._typing import FloatArray, FloatOrArray, as_float
from hazrisk.fragility import lognormal_fragility_curve

# The public interface of the module, in the order the docs list it.
__all__ = [
    "mafe_first_order",
    "required_capacity_first_order",
    "p_second_order",
    "mafe_second_order",
    "required_capacity_second_order",
    "numerical_mafe",
    "numerical_mafe_pchip",
    "risk_contribution",
    "risk_contribution_pchip",
    "risk_ratio_first_order",
    "theta_ratio_first_order",
]


# ---------------------------------------------------------------------------
# Input checks shared by several functions
# ---------------------------------------------------------------------------


def _count(bad: NDArray[np.bool_]) -> str:
    """'n of m element(s)' for an error message about an array condition."""
    return f"{int(bad.sum())} of {bad.size} element(s)"


def _check_p_denominator(k2: FloatOrArray, beta: FloatOrArray) -> FloatOrArray:
    """Return ``1 + 2 k2 beta^2``, raising if it is not positive anywhere.

    The second-order solutions take the square root of ``p``, the reciprocal
    of this quantity, and are derived for ``p > 0``. Without the check a
    non-positive value gives ``nan`` (or ``inf``) silently. One bad element
    raises for the whole call, so a partly valid result is never returned.
    """
    denominator = 1 + 2 * k2 * beta**2
    bad = np.asarray(denominator <= 0)
    if bad.any():
        raise ValueError(
            "the second-order solutions need 1 + 2*k2*beta**2 > 0, but it is "
            f"<= 0 for {_count(bad)}"
        )
    return denominator


def _check_tabulated(H_im: FloatArray, fragility: FloatArray) -> None:
    """Check a tabulated hazard curve and the fragility evaluated on it.

    The numerical integration assumes the levels are ordered by increasing
    ``im``; if they are reversed the trapezoid sum changes sign and a negative
    MAFE comes out without complaint. Equal neighbours are allowed: OpenQuake
    curves often end in a run of zeros.
    """
    if np.ndim(H_im) != 1:
        raise ValueError(f"H_im must be one-dimensional, got shape {np.shape(H_im)}")
    increasing = np.diff(H_im) > 0
    if increasing.any():
        raise ValueError(
            "H_im must not increase: order the levels by increasing im, so "
            f"H_im decreases; it increases at {int(increasing.sum())} of its "
            f"{increasing.size} steps"
        )
    if np.ndim(fragility) == 0 or np.shape(fragility)[-1] != np.size(H_im):
        raise ValueError(
            f"the last axis of fragility (shape {np.shape(fragility)}) must "
            f"have the length of H_im ({np.size(H_im)})"
        )


# ---------------------------------------------------------------------------
# Closed-form solutions, first order
# ---------------------------------------------------------------------------


def mafe_first_order(
    theta_c: ArrayLike, beta: ArrayLike, k0: ArrayLike, k1: ArrayLike
) -> FloatOrArray:
    r"""MAFE of a limit state on the first-order hazard (SAC/FEMA closed form).

    For a lognormal fragility with median :math:`\theta_c` and dispersion
    :math:`\beta`, and the first-order hazard
    :math:`H(im) = k_0 \, im^{-k_1}` (:func:`hazrisk.hazard.first_order_approx`),
    the risk integral has the closed form of Cornell et al. (2002):

    .. math::

        \lambda_C = H(\theta_c) \exp\left(\tfrac{1}{2} k_1^2 \beta^2\right)
                  = k_0 \, \theta_c^{-k_1}
                    \exp\left(\tfrac{1}{2} k_1^2 \beta^2\right).

    Parameters
    ----------
    theta_c : array_like
        Median capacity, in the unit of ``im``.
    beta : array_like
        Lognormal dispersion of the fragility.
    k0, k1 : array_like
        Coefficients of the first-order hazard approximation.

    Returns
    -------
    float or ndarray
        Mean annual frequency of the limit state (per year), with the
        broadcast shape of the inputs.

    Notes
    -----
    The closed form assumes a lognormal fragility and that the hazard follows
    the fitted first-order shape over the intensities that matter for the
    limit state, roughly :math:`\theta_c` and a few :math:`\beta` below it.
    Fit the coefficients over those hazard levels, for example with
    :func:`hazrisk.hazard.fit_first_order_regression`. Where the real hazard curve
    bends away from the fitted shape the answer is biased; check it against
    :func:`numerical_mafe_pchip` on the tabulated curve.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    >>> from hazrisk import risk
    >>> print(f"{risk.mafe_first_order(0.6, beta=0.4, k0=1e-4, k1=2.5):.4e}")
    5.9125e-04
    """
    # Cast arrays to float: numpy refuses negative powers of integer arrays.
    theta_c, beta, k0, k1 = (as_float(a) for a in (theta_c, beta, k0, k1))
    mafe: FloatOrArray = k0 * theta_c ** (-k1) * np.exp(0.5 * (beta * k1) ** 2)
    return mafe


def required_capacity_first_order(
    target_mafe: ArrayLike, beta: ArrayLike, k0: ArrayLike, k1: ArrayLike
) -> FloatOrArray:
    r"""Median capacity that gives a target MAFE on the first-order hazard.

    The inverse of :func:`mafe_first_order` (Cornell et al. 2002): solving
    :math:`\lambda_C = k_0 \theta_c^{-k_1} \exp(\tfrac{1}{2} k_1^2 \beta^2)`
    for :math:`\theta_c`,

    .. math::

        \theta_c = \exp\left(\tfrac{1}{2} k_1 \beta^2
                   - \frac{\ln(\lambda_C / k_0)}{k_1}\right).

    Parameters
    ----------
    target_mafe : array_like
        Target mean annual frequency of the limit state (per year, > 0).
    beta : array_like
        Lognormal dispersion of the fragility.
    k0, k1 : array_like
        Coefficients of the first-order hazard approximation.

    Returns
    -------
    float or ndarray
        Required median capacity, in the unit of ``im``, with the broadcast
        shape of the inputs.

    Notes
    -----
    The closed form assumes a lognormal fragility and that the hazard follows
    the fitted first-order shape over the intensities that matter for the
    limit state, roughly :math:`\theta_c` and a few :math:`\beta` below it.
    Fit the coefficients over those hazard levels, for example with
    :func:`hazrisk.hazard.fit_first_order_regression`. Where the real hazard curve
    bends away from the fitted shape the answer is biased; check it against
    :func:`numerical_mafe_pchip` on the tabulated curve.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    >>> from hazrisk import risk
    >>> theta = risk.required_capacity_first_order(2e-4, beta=0.4, k0=1e-4, k1=2.5)
    >>> print(f"{theta:.4f}")
    0.9257
    >>> print(f"{risk.mafe_first_order(theta, 0.4, 1e-4, 2.5):.1e}")
    2.0e-04
    """
    target_mafe, beta, k0, k1 = (as_float(a) for a in (target_mafe, beta, k0, k1))
    return np.exp((0.5 * k1 * beta**2) - (np.log(target_mafe / k0) / k1))


# ---------------------------------------------------------------------------
# Closed-form solutions, second order
# ---------------------------------------------------------------------------


def p_second_order(k2: ArrayLike, beta: ArrayLike) -> FloatOrArray:
    r"""The factor :math:`p` of the second-order closed-form solutions.

    .. math::

        p = \frac{1}{1 + 2 k_2 \beta^2}

    (Vamvatsikos 2013). It measures how much the curvature :math:`k_2` of the
    hazard and the dispersion :math:`\beta` of the fragility interact;
    :math:`p = 1` for a first-order hazard (:math:`k_2 = 0`). The
    second-order solutions need :math:`p > 0`, i.e.
    :math:`1 + 2 k_2 \beta^2 > 0`, which always holds for :math:`k_2 \ge 0`.

    Parameters
    ----------
    k2 : array_like
        Curvature coefficient of the second-order hazard approximation.
    beta : array_like
        Lognormal dispersion of the fragility.

    Returns
    -------
    float or ndarray
        :math:`p`, with the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If :math:`1 + 2 k_2 \beta^2 \le 0` for any element.

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> from hazrisk import risk
    >>> print(f"{risk.p_second_order(0.2, 0.4):.4f}")
    0.9398
    """
    k2, beta = as_float(k2), as_float(beta)
    return 1 / _check_p_denominator(k2, beta)


def mafe_second_order(
    theta_c: ArrayLike, beta: ArrayLike, k0: ArrayLike, k1: ArrayLike, k2: ArrayLike
) -> FloatOrArray:
    r"""MAFE of a limit state on the second-order hazard (closed form).

    For a lognormal fragility with median :math:`\theta_c` and dispersion
    :math:`\beta`, and the second-order hazard
    :math:`H(im) = k_0 \exp(-k_2 (\ln im)^2 - k_1 \ln im)`
    (:func:`hazrisk.hazard.second_order_approx`), Vamvatsikos (2013) gives

    .. math::

        \lambda_C = \sqrt{p} \, k_0^{1-p} \, H(\theta_c)^p
                    \exp\left(\tfrac{1}{2} p k_1^2 \beta^2\right),

    with :math:`p` from :func:`p_second_order`. At :math:`k_2 = 0`,
    :math:`p = 1` and this is :func:`mafe_first_order`.

    Parameters
    ----------
    theta_c : array_like
        Median capacity, in the unit of ``im``.
    beta : array_like
        Lognormal dispersion of the fragility.
    k0, k1, k2 : array_like
        Coefficients of the second-order hazard approximation.

    Returns
    -------
    float or ndarray
        Mean annual frequency of the limit state (per year), with the
        broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If :math:`1 + 2 k_2 \beta^2 \le 0` for any element.

    Notes
    -----
    The closed form assumes a lognormal fragility and that the hazard follows
    the fitted second-order shape over the intensities that matter for the
    limit state, roughly :math:`\theta_c` and a few :math:`\beta` below it.
    Fit the coefficients over those hazard levels, for example with
    :func:`hazrisk.hazard.fit_second_order_regression`. Where the real hazard curve
    bends away from the fitted shape the answer is biased; check it against
    :func:`numerical_mafe_pchip` on the tabulated curve.

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> from hazrisk import risk
    >>> print(f"{risk.mafe_second_order(0.6, 0.4, 1e-4, 2.5, 0.2):.4e}")
    4.9043e-04
    """
    theta_c, beta, k0, k1, k2 = (as_float(a) for a in (theta_c, beta, k0, k1, k2))
    p = p_second_order(k2, beta)
    H_theta = haz.second_order_approx(theta_c, k0, k1, k2)
    mafe: FloatOrArray = (
        np.sqrt(p) * k0 ** (1 - p) * H_theta**p * np.exp(0.5 * p * k1**2 * beta**2)
    )
    return mafe


def required_capacity_second_order(
    target_mafe: ArrayLike,
    beta: ArrayLike,
    k0: ArrayLike,
    k1: ArrayLike,
    k2: ArrayLike,
) -> FloatOrArray:
    r"""Median capacity that gives a target MAFE on the second-order hazard.

    The inverse of :func:`mafe_second_order` (Vamvatsikos 2013). With
    :math:`p` from :func:`p_second_order`, the target :math:`\lambda_C`, and

    .. math::

        L = \ln\frac{\lambda_C}{k_0 \sqrt{p}}, \qquad
        D = \frac{k_1^2 - 4 k_2 L}{p},

    the required median capacity is :math:`\theta_c = e^x` with

    .. math::

        x = \frac{\tfrac{1}{2} k_1 \beta^2 - L / (p k_1)}
                 {(k_1 + \sqrt{D}) / (2 k_1)}.

    At :math:`k_2 = 0` this is exactly :func:`required_capacity_first_order`.
    The derivation (a quadratic in :math:`x = \ln\theta_c`, rearranged so it
    does not divide by :math:`k_2`) is in
    :ref:`theory-required-capacity-second-order`.

    A negative :math:`D` means no capacity gives the target: the fitted
    second-order curve never reaches that frequency.

    Parameters
    ----------
    target_mafe : array_like
        Target mean annual frequency of the limit state (per year, > 0).
    beta : array_like
        Lognormal dispersion of the fragility.
    k0, k1, k2 : array_like
        Coefficients of the second-order hazard approximation.

    Returns
    -------
    float or ndarray
        Required median capacity, in the unit of ``im``, with the broadcast
        shape of the inputs.

    Raises
    ------
    ValueError
        If :math:`1 + 2 k_2 \beta^2 \le 0`, or :math:`D < 0`, for any
        element.

    Notes
    -----
    The closed form assumes a lognormal fragility and that the hazard follows
    the fitted second-order shape over the intensities that matter for the
    limit state, roughly :math:`\theta_c` and a few :math:`\beta` below it.
    Fit the coefficients over those hazard levels, for example with
    :func:`hazrisk.hazard.fit_second_order_regression`. Where the real hazard curve
    bends away from the fitted shape the answer is biased; check it against
    :func:`numerical_mafe_pchip` on the tabulated curve.

    References
    ----------
    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    >>> from hazrisk import risk
    >>> theta = risk.required_capacity_second_order(2e-4, 0.4, 1e-4, 2.5, 0.2)
    >>> print(f"{theta:.4f}")
    0.8966
    >>> print(f"{risk.mafe_second_order(theta, 0.4, 1e-4, 2.5, 0.2):.1e}")
    2.0e-04
    """
    target_mafe, beta, k0, k1, k2 = (
        as_float(a) for a in (target_mafe, beta, k0, k1, k2)
    )
    p = p_second_order(k2, beta)
    L = np.log(target_mafe / (k0 * np.sqrt(p)))
    D = k1**2 / p - 4 * k2 / p * L
    bad = np.asarray(D < 0)
    if bad.any():
        raise ValueError(
            "the target MAFE is out of reach of the second-order solution "
            "(negative discriminant k1**2/p - 4*k2*L/p) for " + _count(bad)
        )
    # The rationalised root: no division by k2, so it is exact at k2 = 0.
    return np.exp((0.5 * k1 * beta**2 - L / (p * k1)) / ((k1 + np.sqrt(D)) / (2 * k1)))


# ---------------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------------


def numerical_mafe(H_im: ArrayLike, fragility: ArrayLike) -> FloatOrArray:
    r"""MAFE of a limit state by numerical integration of a tabulated curve.

    The risk is the hazard-fragility convolution integral

    .. math::

        \lambda_C = \int_0^\infty P(C \mid im) \,
                    \left|\frac{dH}{d\,im}\right| \, d\,im,

    where :math:`P(C \mid im)` is the fragility (probability of the limit
    state given the intensity measure ``im``) and :math:`H(im)` = ``H_im`` is
    the hazard curve (mean annual frequency of exceeding ``im``). It is
    evaluated with the trapezoid rule over the tabulated pairs
    :math:`(H_i, P_i)`, plus a tail term for the intensities above the last
    level (see Notes).

    ``fragility`` and ``H_im`` must be evaluated at the same intensity levels
    (ordered by increasing ``im``) so that each ``(H_im[i], fragility[i])``
    is a matched pair; ``im`` itself is not needed. Several fragility curves
    on the same hazard curve can be integrated in one call by stacking them
    along the leading axes of ``fragility``: the integration runs over the
    last axis, which must match ``H_im``.

    On a coarse PSHA grid (a few tens of levels) the straight lines between
    levels bias the result; :func:`numerical_mafe_pchip` refines the hazard
    first.

    Parameters
    ----------
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), with the
        levels ordered by increasing ``im``, so non-increasing.
    fragility : array_like
        Fragility curve, i.e. the probability of the limit state at each
        ``im``. Shape ``(n,)`` for one curve or ``(..., n)`` for a stack of
        curves.

    Returns
    -------
    float or ndarray
        Mean annual frequency of the limit state (per year), one per
        fragility curve.

    Raises
    ------
    ValueError
        If ``H_im`` is not one-dimensional or increases anywhere (equal
        neighbours, such as a run of zeros at the end of an OpenQuake curve,
        are allowed), or if the last axis of ``fragility`` does not have the
        length of ``H_im``.

    Notes
    -----
    Rather than estimate the hazard derivative :math:`dH/d\,im` and
    integrate it, this changes the variable of integration from ``im`` to
    the hazard :math:`H`. Because the hazard decreases monotonically with
    ``im``, :math:`|dH/d\,im| \, d\,im = -dH`, and the integral becomes

    .. math::

        \lambda_C = -\int P(C \mid im) \, dH,

    i.e. simply the fragility integrated against the hazard curve.
    Geometrically, if the fragility is plotted on the y-axis against
    ``H_im`` on the x-axis, the risk is the area under that curve.
    ``np.trapezoid(fragility, H_im)`` returns the signed area under the
    polyline through the points ``(H_im[i], fragility[i])``; since the arrays
    are ordered by increasing ``im`` (hence decreasing ``H_im``) the x-axis is
    traversed right-to-left and the signed area is negative, so the result is
    negated.

    This formulation:

    - integrates the smooth, bounded fragility curve with the trapezoid rule,
      rather than the sharply peaked hazard derivative, which straight lines
      represent far less faithfully;
    - remains correct for non-uniformly spaced intensity measure levels,
      because the area uses the actual ``H_im`` spacing and ``im`` itself
      drops out.

    The final term ``H_im[-1] * fragility[-1]`` is the contribution of all
    intensities beyond the last level ``im[-1]``. That is the region where
    the hazard runs from 0 up to ``H_im[-1]`` and the fragility is taken to
    stay at ``fragility[-1]``; geometrically it is the rectangle that extends
    the area down to ``H_im = 0``. If the curve ends in zeros (as OpenQuake
    writes them) this term is zero, and the step from the last positive level
    to the first zero is an ordinary trapezoid.

    Limits of the integration range: below ``im[0]`` the fragility is taken
    as zero, so the curve should start well below the fragility median; above
    the last level the fragility is held at its last value, which
    underestimates the risk slightly if the fragility is still rising there.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.
    https://doi.org/10.1017/9781108425056
    Section 9.4.1, pp. 382-385.

    Examples
    --------
    On a fine grid the result approaches the closed form:

    >>> import numpy as np
    >>> from hazrisk import hazard, fragility, risk
    >>> im = np.geomspace(0.01, 10.0, 2000)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> P = fragility.lognormal_fragility_curve(im, 0.6, 0.4)
    >>> print(f"{risk.numerical_mafe(H_im, P):.4e}")
    4.9043e-04

    Several fragility curves on the same hazard curve are integrated in one
    call by stacking them along the leading axes of ``fragility``; the
    integration runs over the last axis, which matches ``H_im``. Here three
    median capacities, as a ``(3, 2000)`` stack:

    >>> theta_c = np.array([0.4, 0.6, 0.9])
    >>> P_stack = fragility.lognormal_fragility_curve(im, theta_c[:, None], 0.4)
    >>> P_stack.shape
    (3, 2000)
    >>> mafe = risk.numerical_mafe(H_im, P_stack)
    >>> print(np.array2string(mafe, formatter={"float": "{:.4e}".format}))
    [1.1405e-03 4.9043e-04 1.9826e-04]
    """
    H_im = np.asarray(H_im, dtype=float)
    fragility = np.asarray(fragility, dtype=float)
    _check_tabulated(H_im, fragility)
    # -trapz because H_im decreases with im; add the tail beyond the last level.
    mafe: FloatOrArray = (
        -np.trapezoid(fragility, H_im, axis=-1) + H_im[-1] * fragility[..., -1]
    )
    return mafe


def numerical_mafe_pchip(
    theta_c: ArrayLike,
    im: ArrayLike,
    H_im: ArrayLike,
    beta: ArrayLike,
    n_im: int = 2000,
) -> FloatOrArray:
    r"""MAFE for a lognormal fragility on a PCHIP-refined numerical hazard curve.

    PCHIP interpolation is used to smooth out a coarsely sampled hazard curve
    to calculate the MAFE more accurately. The hazard is interpolated onto
    ``n_im`` log-spaced levels with :func:`hazrisk.hazard.pchip_approx`,
    which uses scipy's :class:`scipy.interpolate.PchipInterpolator`
    (Fritsch & Butland 1984; Moler 2004), the lognormal fragility is
    evaluated on the same levels, and the pair is integrated with
    :func:`numerical_mafe`.

    PCHIP is preferred to linear interpolation in log-log space because with
    linear interpolation the hazard slope is constant within each interval and
    changes abruptly at each point (i.e. the slope is discontinuous).
    PCHIP keeps the slope continuous.

    The refined grid runs from ``im[0]`` to the last positive value of
    ``H_im``, since PCHIP works on :math:`\ln H` and :math:`\ln 0` is not
    defined. Trailing zeros in ``H_im`` (as OpenQuake writes them above the
    truncation ceiling of the ground motion model, or below the precision the
    probability is stored at) still carry information: the hazard has fallen
    to zero by the first zero level. So if the curve has trailing zeros, the
    first zero level (its ``im``, with :math:`H = 0`) is appended to the
    refined grid before integrating. The step from the last positive level to
    that zero is then a trapezoid with the fragility averaged over the
    interval, as :func:`numerical_mafe` does on the raw curve, and the tail
    term is zero. Zeros after the first one add nothing and are not used.
    Without trailing zeros, everything above the last level enters through
    the tail term of :func:`numerical_mafe`.

    ``theta_c`` and ``beta`` broadcast against each other, so a whole set of
    fragility curves (e.g. a Monte Carlo sample of ``(theta_c, beta)``, or
    the boundary of a confidence region) is integrated against the same
    refined hazard in one call.

    Parameters
    ----------
    theta_c : array_like
        Median capacity, in the unit of ``im``.
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        decreasing apart from a trailing run of zeros.
    beta : array_like
        Lognormal dispersion of the fragility.
    n_im : int, optional
        Number of log-spaced levels of the refined integration grid. The
        default is 2000.

    Returns
    -------
    float or ndarray
        Mean annual frequency of the limit state (per year): a Python float
        for scalar ``theta_c`` and ``beta``, otherwise an array with their
        broadcast shape.

    Raises
    ------
    ValueError
        If the hazard curve is unusable (see
        :func:`hazrisk.hazard.pchip_interpolant`).

    Notes
    -----
    Limits of the integration range: below ``im[0]`` the fragility is taken
    as zero, so the curve should start well below the fragility median. For
    a curve without trailing zeros, the fragility is held at its value at the
    last level for every intensity above it (the tail term of
    :func:`numerical_mafe`), which underestimates the risk slightly if the
    fragility is still rising there.

    References
    ----------
    Fritsch, F. N. & Butland, J. (1984). A method for constructing local
    monotone piecewise cubic interpolants. *SIAM Journal on Scientific and
    Statistical Computing*, 5(2), 300-304. https://doi.org/10.1137/0905021

    Moler, C. (2004). *Numerical Computing with MATLAB*. Society for
    Industrial and Applied Mathematics.
    https://doi.org/10.1137/1.9780898717952

    Examples
    --------
    On a coarse 30-level curve that follows the second-order form, the
    closed-form second-order MAFE is 4.9043e-04, and the numerical
    integration matches it:

    >>> import numpy as np
    >>> from hazrisk import hazard, risk
    >>> im = np.geomspace(0.01, 10.0, 30)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> print(f"{risk.numerical_mafe_pchip(0.6, im, H_im, 0.4):.4e}")
    4.9043e-04

    ``theta_c`` and ``beta`` broadcast against each other, so a set of
    fragility curves is integrated against the same refined hazard in one
    call. Here three median capacities (a column) times two dispersions (a
    row) give a ``(3, 2)`` array:

    >>> theta_c = np.array([[0.4], [0.6], [0.9]])
    >>> beta = np.array([0.3, 0.5])
    >>> mafe = risk.numerical_mafe_pchip(theta_c, im, H_im, beta)
    >>> mafe.shape
    (3, 2)
    >>> print(np.array2string(mafe, formatter={"float": "{:.4e}".format}))
    [[1.0003e-03 1.3362e-03]
     [4.2043e-04 5.9068e-04]
     [1.6585e-04 2.4597e-04]]
    """
    im = np.asarray(im, dtype=float)
    im_pos, H_pos = haz._trim_zero_tail(im, H_im)

    # Refine the positive part of the curve with PCHIP on a log-spaced grid.
    im_x = np.geomspace(im_pos[0], im_pos[-1], n_im)
    H_im_x = haz.pchip_approx(im_x, im_pos, H_pos)

    # If the curve has trailing zeros, the hazard has reached zero by the first
    # zero level: append that level with H = 0, so the step down to it is
    # integrated as a trapezoid (as numerical_mafe does on the raw curve) and
    # the tail term vanishes. Later zeros would add nothing.
    if im_pos.size < im.size:
        im_x = np.append(im_x, im[im_pos.size])
        H_im_x = np.append(H_im_x, 0.0)

    # Trailing axis = intensity level, leading axes = the (theta_c, beta) set.
    theta_c, beta = np.broadcast_arrays(
        np.asarray(theta_c, dtype=float), np.asarray(beta, dtype=float)
    )
    fragility = lognormal_fragility_curve(im_x, theta_c[..., None], beta[..., None])

    mafe = numerical_mafe(H_im_x, fragility)
    return float(mafe) if np.ndim(mafe) == 0 else mafe


# ---------------------------------------------------------------------------
# Where the risk comes from
# ---------------------------------------------------------------------------


def risk_contribution(
    im: ArrayLike, H_im: ArrayLike, fragility: ArrayLike, normalised: bool = True
) -> FloatArray:
    r"""Share of the risk that comes from each tabulated level.

    The discrete counterpart of :func:`risk_contribution_pchip`. It splits the
    trapezoid sum of :func:`numerical_mafe` into its terms, so the
    contributions add up to that MAFE. For ``n`` levels ordered by
    increasing ``im``, with :math:`H_i` the hazard and :math:`P_i` the
    fragility at level ``i``:

    - element ``i`` (for ``i = 0 ... n-2``) is the risk from the interval
      between level ``i`` and level ``i+1``,

      .. math::

          \Delta\lambda_i = \tfrac{1}{2} (P_i + P_{i+1}) (H_i - H_{i+1}),

      i.e. the annual frequency of ground motions that fall in that interval
      (:math:`H_i - H_{i+1}`) times the average probability of the limit
      state over it;
    - the last element is the tail term :math:`H_{n-1} P_{n-1}`, the risk
      from every intensity above the last level, with the fragility held at
      its last value (as in :func:`numerical_mafe`).

    So the result has one value per level, and their sum equals
    ``numerical_mafe(H_im, fragility)`` up to floating-point rounding.

    Parameters
    ----------
    im : array_like
        Intensity measure levels, strictly increasing. Used only to check the
        ordering.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year),
        non-increasing.
    fragility : array_like
        Probability of the limit state at each ``im``. Shape ``(n,)`` for one
        fragility curve, or ``(..., n)`` for a stack of curves on the same
        hazard.
    normalised : bool, optional
        If True (the default) divide by the total, so the contributions add
        up to one. If False return them as frequencies (per year).

    Returns
    -------
    ndarray
        The contributions, with the shape of ``fragility``.

    Raises
    ------
    ValueError
        If ``im`` is not strictly increasing, ``H_im`` increases anywhere, or
        the lengths of ``im``, ``H_im`` and the last axis of ``fragility``
        differ.

    Notes
    -----
    The contributions share the limits of :func:`numerical_mafe`: nothing is
    counted below ``im[0]`` (the fragility is taken as zero there), and the
    last element holds the fragility at its last value for every intensity
    above the last level. If the curve ends in zeros, that last element is
    zero and the step to the first zero level is an ordinary interval.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.
    https://doi.org/10.1017/9781108425056
    Section 9.4.1, pp. 382-385.

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import hazard, fragility, risk
    >>> im = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    >>> H_im = hazard.first_order_approx(im, 1e-4, 2.5)
    >>> P = fragility.lognormal_fragility_curve(im, 0.6, 0.4)
    >>> share = risk.risk_contribution(im, H_im, P)
    >>> print(np.round(share, 3))
    [0.042 0.39  0.4   0.135 0.033]
    >>> print(f"{share.sum():.6f}")
    1.000000
    """
    im = np.asarray(im, dtype=float)
    H_im = np.asarray(H_im, dtype=float)
    fragility = np.asarray(fragility, dtype=float)
    if np.ndim(im) != 1 or np.size(im) != np.size(H_im):
        raise ValueError(
            f"im must be one-dimensional with the length of H_im "
            f"({np.size(H_im)}), got shape {np.shape(im)}"
        )
    if np.any(np.diff(im) <= 0):
        raise ValueError("im must be strictly increasing")
    _check_tabulated(H_im, fragility)

    # Trapezoid terms of the integral over the hazard, one per interval, then
    # the tail beyond the last level: the same terms numerical_mafe sums.
    intervals = (
        0.5 * (fragility[..., :-1] + fragility[..., 1:]) * (H_im[:-1] - H_im[1:])
    )
    tail = H_im[-1] * fragility[..., -1:]
    contributions = np.concatenate([intervals, tail], axis=-1)

    if normalised:
        shares: FloatArray = contributions / contributions.sum(axis=-1, keepdims=True)
        return shares
    raw: FloatArray = contributions
    return raw


def risk_contribution_pchip(
    im_x: ArrayLike,
    theta_c: float,
    im: ArrayLike,
    H_im: ArrayLike,
    beta: float,
    normalised: bool = True,
) -> FloatOrArray:
    r"""Risk per unit :math:`\ln im`, with the hazard interpolated by PCHIP.

    This is the integrand of the hazard-fragility convolution expressed per
    unit of :math:`\ln im`,

    .. math::

        \frac{d\lambda_C}{d \ln im} = P(C \mid im) \,
            \left|\frac{dH}{d\,im}\right| \, im,

    with :math:`dH/d\,im` from :func:`hazrisk.hazard.pchip_dHdx` (scipy's
    :class:`scipy.interpolate.PchipInterpolator`). Plotted
    against ``im`` on a log axis, the area under it is the risk, so it shows
    which intensities the risk comes from: the continuous counterpart of
    :func:`risk_contribution`. Per unit :math:`\ln im` rather than per unit
    ``im`` because hazard curves are tabulated and plotted on log-spaced
    levels.

    Because the PCHIP hazard slope is continuous, the result has no steps at
    the tabulated levels of the hazard curve, which a straight-line log-log
    interpolation produces.

    The contribution is evaluated by PCHIP from ``im[0]`` up to the last
    positive level of the curve, and is ``nan`` above it. With
    ``normalised=True`` it is divided by the total from
    :func:`numerical_mafe_pchip`, so it integrates (over :math:`\ln im`) to
    one minus the share of that total which lies above the last positive
    level: the tail term for a curve without trailing zeros, or the step
    down to the first zero level for a curve with them.

    Parameters
    ----------
    im_x : array_like
        Intensity measure levels at which to evaluate the contribution.
        Levels outside ``im[0]`` to the last positive level of the curve
        return ``nan``.
    theta_c : float
        Median capacity, in the unit of ``im``.
    im : array_like
        Intensity measure levels of the numerical hazard curve, strictly
        increasing.
    H_im : array_like
        Mean annual frequency of exceeding each ``im`` (per year), strictly
        decreasing apart from a trailing run of zeros.
    beta : float
        Lognormal dispersion of the fragility.
    normalised : bool, optional
        If True (the default) divide by the total mean annual frequency, as
        :func:`risk_contribution` does.

    Returns
    -------
    float or ndarray
        :math:`d\lambda_C / d \ln im` at each ``im_x`` (per year), or its share
        of the total if ``normalised``.

    Raises
    ------
    ValueError
        If the hazard curve is unusable (see
        :func:`hazrisk.hazard.pchip_interpolant`).

    Notes
    -----
    The total it is normalised by has the limits of
    :func:`numerical_mafe_pchip`: the fragility is taken as zero below
    ``im[0]``, and, for a curve without trailing zeros, held at its last
    value above the last level.

    References
    ----------
    Fritsch, F. N. & Butland, J. (1984). A method for constructing local
    monotone piecewise cubic interpolants. *SIAM Journal on Scientific and
    Statistical Computing*, 5(2), 300-304. https://doi.org/10.1137/0905021

    Moler, C. (2004). *Numerical Computing with MATLAB*. Society for
    Industrial and Applied Mathematics.
    https://doi.org/10.1137/1.9780898717952

    Examples
    --------
    The normalised contribution integrates to one minus the small share of
    the risk above the last level of the curve:

    >>> import numpy as np
    >>> from hazrisk import hazard, risk
    >>> im = np.geomspace(0.01, 10.0, 30)
    >>> H_im = hazard.second_order_approx(im, 1e-4, 2.5, 0.2)
    >>> im_x = np.geomspace(im[0], im[-1], 4000)
    >>> c = risk.risk_contribution_pchip(im_x, 0.6, im, H_im, 0.4)
    >>> print(f"{np.trapezoid(c, np.log(im_x)):.4f}")
    0.9998
    """
    im_x = np.asarray(im_x, dtype=float)
    # Integrand per unit ln(im): P(C|im) * |dH/d im| * im, with the PCHIP slope.
    contribution = (
        lognormal_fragility_curve(im_x, theta_c, beta)
        * -haz.pchip_dHdx(im_x, im, H_im)
        * im_x
    )
    if normalised:
        # Divide by the same total that numerical_mafe_pchip reports.
        return contribution / numerical_mafe_pchip(theta_c, im, H_im, beta)
    return contribution


# ---------------------------------------------------------------------------
# Ratios between two designs, first order
# ---------------------------------------------------------------------------


def risk_ratio_first_order(
    R_theta: ArrayLike,
    k1: ArrayLike,
    beta_1: ArrayLike,
    R_beta: ArrayLike = 1,
) -> FloatOrArray:
    r"""Ratio of the MAFEs of two designs, on the first-order hazard.

    Two lognormal fragilities, with medians :math:`\theta_1, \theta_2` and
    dispersions :math:`\beta_1, \beta_2`, on the same first-order hazard
    curve. Dividing the closed form of Cornell et al. (2002) (see
    :func:`mafe_first_order`) for case 2 by that for case 1, :math:`k_0`
    cancels and

    .. math::

        R_\lambda = \frac{\lambda_2}{\lambda_1}
                  = R_\theta^{-k_1}
                    \exp\left(\tfrac{1}{2} k_1^2 \beta_1^2 (R_\beta^2 - 1)\right),

    with :math:`R_\theta = \theta_2 / \theta_1` and
    :math:`R_\beta = \beta_2 / \beta_1`.

    The inputs broadcast against each other by the usual numpy rules:
    scalars give a scalar. For the full grid of every combination, give
    each input its own axis with :func:`numpy.ix_`.

    Parameters
    ----------
    R_theta : array_like
        Ratio of median capacities, :math:`\theta_2 / \theta_1`.
    k1 : array_like
        Exponent of the first-order hazard curve.
    beta_1 : array_like
        Dispersion of the reference (case 1) fragility.
    R_beta : array_like, optional
        Ratio of dispersions, :math:`\beta_2 / \beta_1`. The default of one
        means the two fragilities have the same dispersion.

    Returns
    -------
    float or ndarray
        :math:`R_\lambda = \lambda_2 / \lambda_1`, with the broadcast shape
        of the inputs.

    Notes
    -----
    The ratio holds only on the first-order hazard, and assumes both designs
    see the same power-law hazard, with the same slope :math:`k_1`. Where
    that is not a fair description of the real hazard curve over both
    capacities, compute the two MAFEs separately (for example with
    :func:`numerical_mafe_pchip`).

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import risk
    >>> print(f"{risk.risk_ratio_first_order(1.2, 2.5, 0.4):.4f}")
    0.6339
    >>> grid = risk.risk_ratio_first_order(
    ...     *np.ix_([0.9, 1.0, 1.1], [2.0, 3.0], [0.2, 0.4], [1.0, 1.5]))
    >>> grid.shape
    (3, 2, 2, 2)
    """
    R_theta, k1, beta_1, R_beta = (as_float(a) for a in (R_theta, k1, beta_1, R_beta))
    R_lambda: FloatOrArray = R_theta ** (-k1) * np.exp(
        0.5 * k1**2 * (R_beta**2 - 1) * beta_1**2
    )
    return R_lambda


def theta_ratio_first_order(
    R_lambda: ArrayLike,
    k1: ArrayLike,
    beta_1: ArrayLike,
    R_beta: ArrayLike = 1,
) -> FloatOrArray:
    r"""Ratio of median capacities that gives a target ratio of MAFEs.

    The inverse of :func:`risk_ratio_first_order`: how much the median
    capacity has to change, :math:`R_\theta = \theta_2 / \theta_1`, for
    the MAFE to change by :math:`R_\lambda = \lambda_2 / \lambda_1`, on the
    first-order hazard curve (Cornell et al. 2002). Solving the ratio of the
    two closed forms for :math:`R_\theta`,

    .. math::

        R_\theta = \exp\left(\tfrac{1}{2} k_1 \beta_1^2 (R_\beta^2 - 1)
                   - \frac{\ln R_\lambda}{k_1}\right),

    with :math:`R_\beta = \beta_2 / \beta_1`.

    The inputs broadcast against each other by the usual numpy rules:
    scalars give a scalar. For the full grid of every combination, give
    each input its own axis with :func:`numpy.ix_`.

    Parameters
    ----------
    R_lambda : array_like
        Target ratio of MAFEs, :math:`\lambda_2 / \lambda_1`.
    k1 : array_like
        Exponent of the first-order hazard curve.
    beta_1 : array_like
        Dispersion of the reference (case 1) fragility.
    R_beta : array_like, optional
        Ratio of dispersions, :math:`\beta_2 / \beta_1`. The default of one
        means the two fragilities have the same dispersion.

    Returns
    -------
    float or ndarray
        :math:`R_\theta = \theta_2 / \theta_1`, with the broadcast shape of
        the inputs.

    Notes
    -----
    As for :func:`risk_ratio_first_order`, this holds only on the
    first-order hazard, and assumes both designs see the same power-law
    hazard, with the same slope :math:`k_1`.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Examples
    --------
    Halving the MAFE with the same dispersion, for ``k1 = 2.5``:

    >>> import numpy as np
    >>> from hazrisk import risk
    >>> print(f"{risk.theta_ratio_first_order(0.5, 2.5, 0.4):.4f}")
    1.3195

    Every combination of four input vectors, as a 4-D array with one axis
    per input:

    >>> R_lambda = np.linspace(0.5, 2.0, 16)
    >>> k1 = [1.5, 2.0, 2.5, 3.0]
    >>> beta_1 = [0.15, 0.2, 0.25, 0.4, 0.5]
    >>> R_beta = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    >>> R_theta = risk.theta_ratio_first_order(*np.ix_(R_lambda, k1, beta_1, R_beta))
    >>> R_theta.shape
    (16, 4, 5, 6)
    """
    R_lambda, k1, beta_1, R_beta = (as_float(a) for a in (R_lambda, k1, beta_1, R_beta))
    return np.exp(0.5 * k1 * (R_beta**2 - 1) * beta_1**2 - np.log(R_lambda) / k1)
