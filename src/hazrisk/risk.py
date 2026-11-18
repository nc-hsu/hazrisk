"""Risk: the mean annual frequency of a limit state, and related quantities.

This is the last link of the hazard -> fragility -> risk chain. It combines a
hazard curve (:mod:`hazrisk.hazard`) with a fragility curve
(:mod:`hazrisk.fragility`) to give the mean annual frequency of exceeding a
limit state such as collapse (MAFE of the limit state, often written
``lambda_C``). It provides

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

Tabulated inputs must be ordered by increasing ``im``, so ``H_im`` decreases
along the array; the numerical integration relies on it.
"""

import numpy as np

from hazrisk import hazard as haz
from hazrisk._typing import as_float, scalar, vector
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


def required_capacity_first_order(target_mafe: scalar, k0: scalar, k1: scalar,
                                  beta: scalar) -> scalar:
    """Calculate theta_c from mean annual frequency of exceedance for first order 
    hazard."""
    target_mafe, k0, k1, beta = (as_float(a)
                                 for a in (target_mafe, k0, k1, beta))
    return np.exp((0.5 * k1 * beta ** 2) - (np.log(target_mafe / k0) / k1))


def mafe_first_order(theta_c: scalar, k0: scalar, k1: scalar, beta: scalar
                     ) -> scalar:
    """Calculate mean annual frequency of collapse for first order hazard."""
    # Cast arrays to float: numpy refuses negative powers of integer arrays.
    theta_c, k0, k1, beta = (as_float(a)
                             for a in (theta_c, k0, k1, beta))
    return k0 * theta_c ** (-k1) * np.exp(0.5 * (beta * k1) ** 2)


def _count(bad) -> str:
    """'n of m element(s)' for an error message about an array condition."""
    bad = np.asarray(bad)
    return f"{int(bad.sum())} of {bad.size} element(s)"


def _check_p_denominator(k2, beta):
    """Return ``1 + 2 k2 beta^2``, raising if it is not positive anywhere.

    The second-order solutions take the square root of ``p``, the reciprocal
    of this quantity, and are derived for ``p > 0``. Without the check a
    non-positive value gives ``nan`` (or ``inf``) silently. One bad element
    raises for the whole call, so a partly valid result is never returned.
    """
    denominator = 1 + 2 * k2 * beta ** 2
    bad = np.asarray(denominator <= 0)
    if bad.any():
        raise ValueError(
            "the second-order solutions need 1 + 2*k2*beta**2 > 0, but it is "
            f"<= 0 for {_count(bad)}")
    return denominator


def p_second_order(k2: scalar, beta: scalar) -> scalar:
    """Calculate the p factor for second order risk approximation.

    Raises ``ValueError`` if ``1 + 2 k2 beta^2 <= 0`` for any element.
    """
    k2, beta = as_float(k2), as_float(beta)
    return 1 / _check_p_denominator(k2, beta)


def required_capacity_second_order(target_mafe: scalar, k0: scalar, k1: scalar,
                                   k2: scalar, beta: scalar) -> scalar:
    r"""Median capacity that gives a target MAFE on the second-order hazard.

    The inverse of :func:`mafe_second_order` (Vamvatsikos 2013). Taking logs
    of the closed form

    .. math::

        \lambda = \sqrt{p}\, k_0^{1-p}\, H(\theta_c)^p
                  \exp\left(\tfrac{1}{2} p k_1^2 \beta^2\right),
        \qquad
        \ln H(\theta_c) = \ln k_0 - k_1 x - k_2 x^2,
        \qquad x = \ln \theta_c,

    and writing :math:`L = \ln\left(\lambda / (k_0 \sqrt{p})\right)` gives
    a quadratic in :math:`x`,

    .. math::

        k_2 x^2 + k_1 x + \left(\frac{L}{p} - \tfrac{1}{2} k_1^2 \beta^2\right) = 0,

    with discriminant (using :math:`1/p = 1 + 2 k_2 \beta^2`)

    .. math::

        D = \frac{k_1^2 - 4 k_2 L}{p}.

    The root that tends to the first-order answer as :math:`k_2 \to 0` is
    :math:`x = (-k_1 + \sqrt{D}) / (2 k_2)`. Written like that it divides
    by :math:`k_2`: it fails at :math:`k_2 = 0` and loses accuracy for small
    :math:`k_2`, where :math:`\sqrt{D}` is close to :math:`k_1` and the
    numerator cancels. Multiplying numerator and denominator by
    :math:`k_1 + \sqrt{D}` removes the division (since
    :math:`D - k_1^2 = 2 k_2 (k_1^2 \beta^2 - 2L/p)`):

    .. math::

        x = \frac{k_1^2 \beta^2 - 2L/p}{k_1 + \sqrt{D}}
          = \frac{\tfrac{1}{2} k_1 \beta^2 - L / (p k_1)}
                 {(k_1 + \sqrt{D}) / (2 k_1)}.

    The second form is the one evaluated. At :math:`k_2 = 0`, :math:`p = 1`,
    :math:`\sqrt{D} = k_1` and the denominator is exactly one, so the result
    is exactly :func:`required_capacity_first_order`.

    Raises ``ValueError`` if ``1 + 2 k2 beta^2 <= 0``, or if ``D < 0``, which
    means the fitted second-order curve never reaches the target MAFE, for any
    element.
    """
    target_mafe, k0, k1, k2, beta = (as_float(a)
                                     for a in (target_mafe, k0, k1, k2, beta))
    p = p_second_order(k2, beta)
    L = np.log(target_mafe / (k0 * np.sqrt(p)))
    D = k1 ** 2 / p - 4 * k2 / p * L
    bad = np.asarray(D < 0)
    if bad.any():
        raise ValueError(
            "the target MAFE is out of reach of the second-order solution "
            "(negative discriminant k1**2/p - 4*k2*L/p) for " + _count(bad))
    return np.exp((0.5 * k1 * beta ** 2 - L / (p * k1))
                  / ((k1 + np.sqrt(D)) / (2 * k1)))


def mafe_second_order(theta_c: scalar, k0: scalar, k1: scalar, k2: scalar, 
                      beta: scalar) -> scalar:
    """Calculate mean annual frequency of collapse for second order hazard.

    Raises ``ValueError`` if ``1 + 2 k2 beta^2 <= 0`` for any element.
    """
    theta_c, k0, k1, k2, beta = (as_float(a)
                                 for a in (theta_c, k0, k1, k2, beta))
    p = p_second_order(k2, beta)
    H_theta = haz.second_order_approx(theta_c, k0, k1, k2)
    return np.sqrt(p) * k0 ** (1 - p) * H_theta ** p * \
               np.exp(0.5 * p * k1 ** 2 * beta **2)


def _check_tabulated(H_im, fragility) -> None:
    """Check a tabulated hazard curve and the fragility evaluated on it.

    The numerical integration assumes the levels are ordered by increasing
    ``im``; if they are reversed the trapezoid sum changes sign and a negative
    MAFE comes out without complaint. Equal neighbours are allowed: OpenQuake
    curves often end in a run of zeros.
    """
    if np.ndim(H_im) != 1:
        raise ValueError(f"H_im must be one-dimensional, got shape "
                         f"{np.shape(H_im)}")
    increasing = np.diff(H_im) > 0
    if increasing.any():
        raise ValueError(
            "H_im must not increase: order the levels by increasing im, so "
            f"H_im decreases; it increases at {int(increasing.sum())} of its "
            f"{increasing.size} steps")
    if np.ndim(fragility) == 0 or np.shape(fragility)[-1] != np.size(H_im):
        raise ValueError(
            f"the last axis of fragility (shape {np.shape(fragility)}) must "
            f"have the length of H_im ({np.size(H_im)})")


def numerical_mafe(H_im: vector, fragility: vector) -> scalar:
    """Calculate the mean annual frequency of collapse by numerical integration.

    The collapse risk is the standard hazard-fragility convolution integral

        lambda_C = integral of  P(C|im) * |d(lambda)/d(im)|  d(im)

    where P(C|im) is the fragility (probability of collapse given the intensity
    measure ``im``) and lambda(im) = ``H_im`` is the hazard curve (mean annual
    frequency of exceeding ``im``).

    Rather than estimate the hazard derivative ``d(lambda)/d(im)`` and re-integrate
    it, this changes the variable of integration from ``im`` to the hazard
    ``lambda``. Because the hazard is monotonically decreasing in ``im``,
    ``|d(lambda)/d(im)| d(im) = -d(lambda)``, and the integral collapses to

        lambda_C = -integral of  P(C|im)  d(lambda)

    i.e. simply the fragility integrated against the hazard curve. Geometrically,
    if the fragility is plotted on the y-axis against ``H_im`` on the x-axis, the
    collapse risk is the area under that curve. ``np.trapezoid(fragility, H_im)``
    returns the signed area under the polyline through the points
    ``(H_im[i], fragility[i])``; since the arrays are ordered by increasing ``im``
    (hence decreasing ``H_im``) the x-axis is traversed right-to-left and the
    signed area is negative, so the result is negated.

    This formulation:
    - trapezoidally integrates the smooth, bounded fragility curve rather than the
      sharply peaked hazard derivative, which linear interpolation represents far
      more faithfully;
    - remains correct for non-uniformly spaced intensity measure levels, because
      the area uses the actual ``H_im`` spacing and ``im`` itself drops out.

    The final term ``H_im[-1] * fragility[-1]`` captures the contribution of all
    intensities beyond the last level ``im[-1]``. This is the region where the
    hazard runs from ``0`` up to ``H_im[-1]`` and the fragility is taken to stay at
    ``fragility[-1]``; geometrically it is the rectangle that extends the area down
    to ``H_im = 0``.

    Note that ``fragility`` and ``H_im`` must be discretised at the same intensity
    measure levels ``im`` (ordered increasing) so that each point
    ``(H_im[i], fragility[i])`` is a matched pair; ``im`` itself is not needed.

    Several fragility curves on the same hazard curve can be integrated in one
    call by stacking them along the leading axes of ``fragility``: the
    integration runs over the last axis, which must match ``H_im``.

    Parameters
    ----------
    H_im : vector
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im`` (monotonically decreasing).
    fragility : vector
        Fragility curve, i.e. P(collapse | im) at each ``im``. Shape ``(n,)``
        for one curve or ``(..., n)`` for a stack of curves.

    Returns
    -------
    scalar or vector
        The mean annual frequency of collapse, one per fragility curve.

    Raises
    ------
    ValueError
        If ``H_im`` is not one-dimensional or increases anywhere (equal
        neighbours, such as a run of zeros at the end of an OpenQuake curve,
        are allowed), or if the last axis of ``fragility`` does not have the
        length of ``H_im``.
    """
    H_im = as_float(H_im)
    fragility = as_float(fragility)
    _check_tabulated(H_im, fragility)
    # -trapz because H_im decreases with im; add the tail beyond the last level.
    return (-np.trapezoid(fragility, H_im, axis=-1)
            + H_im[-1] * fragility[..., -1])


def numerical_mafe_pchip(theta_c: scalar|vector, im: vector, H_im: vector,
                         beta: scalar|vector, n_im: int=2000) -> scalar|vector:
    """Mean annual frequency of collapse for a lognormal fragility on a numerical
    hazard curve, with the hazard interpolated by PCHIP in log-log space.

    A PSHA hazard curve is usually tabulated at a few tens of intensity levels,
    far too coarse to integrate directly against a fragility curve with a
    dispersion of 0.2-0.4. This function refines it first: the hazard is
    interpolated onto ``n_im`` log-spaced levels with :func:`hazard.pchip_approx`
    (Fritsch & Carlson 1980), the lognormal fragility is evaluated on the same
    levels, and the pair is integrated with :func:`numerical_mafe`.

    Why PCHIP rather than straight lines in log-log space: the risk integrand is
    ``P(C|im) * |dH/d(im)|``, and with straight-line interpolation the hazard
    slope is constant within each tabulated interval and steps at every
    tabulated level. That shows as jumps in the risk integrand (see
    :func:`risk_contribution_pchip`) and biases the integral where the slope of
    ``ln(H)`` is changing quickly, i.e. at the rare end of the curve that
    dominates collapse risk. PCHIP keeps the slope continuous and cannot
    overshoot, so the refined hazard still decreases monotonically. For a first
    order (power law) hazard curve the interpolation is exact and the result
    converges to :func:`mafe_first_order`.

    The integration range is ``im[0]`` to the last positive value of ``H_im``:

    - Below ``im[0]`` the fragility is assumed negligible, so the curve must
      start well below the fragility median.
    - Trailing zeros in ``H_im``, as written by OpenQuake above the truncation
      ceiling or below the precision the probability is stored at, are dropped
      (:func:`hazard._trim_zero_tail`). Everything above the last positive level
      enters through the tail term of :func:`numerical_mafe`, i.e. with the
      fragility held at its value at that level.

    ``theta_c`` and ``beta`` broadcast against each other, so a whole set of
    fragility curves (e.g. a Monte Carlo sample of ``(theta_c, beta)``, or the
    boundary of a confidence region) is integrated against the same refined
    hazard in one call.

    Parameters
    ----------
    theta_c : scalar or vector
        Median collapse capacity, in the units of ``im``.
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing
        apart from a trailing run of zeros).
    beta : scalar or vector
        Lognormal dispersion of the collapse fragility.
    n_im : int, optional
        Number of log-spaced levels of the refined integration grid. The
        default of 2000 puts the integration error well below 0.1 % for
        ``beta`` down to about 0.1.

    Returns
    -------
    scalar or vector
        The mean annual frequency of collapse, with the broadcast shape of
        ``theta_c`` and ``beta``.
    """
    im, H_im = haz._trim_zero_tail(im, H_im)
    iml = np.geomspace(im[0], im[-1], n_im)
    H_iml = haz.pchip_approx(iml, im, H_im)

    # Trailing axis = intensity level, leading axes = the (theta_c, beta) set.
    theta_c, beta = np.broadcast_arrays(np.asarray(theta_c, dtype=float),
                                        np.asarray(beta, dtype=float))
    fragility = lognormal_fragility_curve(iml, theta_c[..., None], beta[..., None])

    mafe = numerical_mafe(H_iml, fragility)
    return float(mafe) if np.ndim(mafe) == 0 else mafe


def risk_contribution_pchip(iml: scalar|vector, theta_c: scalar, im: vector,
                            H_im: vector, beta: scalar,
                            normalised: bool=True) -> scalar|vector:
    """Contribution to the collapse risk per unit ``ln(im)``, with the hazard
    interpolated by PCHIP in log-log space.

    This is the integrand of the hazard-fragility convolution expressed per unit
    of ``ln(im)``,

        d(lambda_C)/d(ln im) = P(C|im) * |dH/d(im)| * im,

    with ``dH/d(im)`` from :func:`hazard.pchip_dHdx`. Plotted against ``im`` on
    a log axis, the area under it is the collapse risk, so it shows which
    intensities the risk comes from: the continuous counterpart of
    :func:`risk_contribution`. Per unit ``ln(im)`` rather than per unit ``im``
    because hazard curves are tabulated and plotted on log-spaced levels.

    Because the PCHIP hazard slope is continuous, the result has no steps at the
    tabulated levels of the hazard curve, which a straight-line log-log
    interpolation produces.

    With ``normalised=True`` the contribution is divided by the total collapse
    risk from :func:`numerical_mafe_pchip`, so it integrates (over ``ln(im)``)
    to one minus the share of the risk that lies above the last positive level
    of the curve.

    Parameters
    ----------
    iml : scalar or vector
        Intensity measure levels at which to evaluate the contribution. Levels
        outside ``im[0]`` to the last positive level of the curve return
        ``nan``.
    theta_c : scalar
        Median collapse capacity, in the units of ``im``.
    im : vector
        Intensity measure levels of the numerical hazard curve, ordered by
        increasing ``im``.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (strictly decreasing
        apart from a trailing run of zeros).
    beta : scalar
        Lognormal dispersion of the collapse fragility.
    normalised : bool, optional
        If True (the default) divide by the total mean annual frequency of
        collapse, as :func:`risk_contribution` does.

    Returns
    -------
    scalar or vector
        ``d(lambda_C)/d(ln im)`` at each ``iml`` (per year), or its share of the
        total if ``normalised``.
    """
    iml = np.asarray(iml, dtype=float)
    contribution = (lognormal_fragility_curve(iml, theta_c, beta) *
                    -haz.pchip_dHdx(iml, im, H_im) * iml)
    if normalised:
        return contribution / numerical_mafe_pchip(theta_c, im, H_im, beta)
    return contribution


def risk_contribution(im: vector, H_im: vector, fragility: vector,
                      normalised: bool = True) -> vector:
    r"""Share of the collapse risk that comes from each tabulated level.

    The discrete counterpart of :func:`risk_contribution_pchip`. It splits the
    trapezoid sum of :func:`numerical_mafe` into its terms, so the
    contributions add up to that MAFE. For ``n`` levels ordered by
    increasing ``im``, with :math:`H_i` the hazard and :math:`P_i` the
    fragility at level ``i``:

    - element ``i`` (for ``i = 0 ... n-2``) is the risk from the interval
      between level ``i`` and level ``i+1``,

      .. math::

          \Delta\lambda_i = \tfrac{1}{2} (P_i + P_{i+1}) (H_i - H_{i+1}),

      i.e. the fraction of the year's ground motions that fall in that
      interval (:math:`H_i - H_{i+1}`) times the average probability of
      collapse over it;
    - the last element is the tail term :math:`H_{n-1} P_{n-1}`, the risk
      from every intensity above the last level, with the fragility held at
      its last value (as in :func:`numerical_mafe`).

    So the result has one value per level, and

    .. math::

        \sum_i \Delta\lambda_i = \texttt{numerical\_mafe(H\_im, fragility)}

    up to floating-point rounding. Plotted as bars against ``im``, element
    ``i`` sits on the interval starting at ``im[i]``; it is approximately the
    integral of :func:`risk_contribution_pchip` over that interval in
    ``ln(im)``.

    Parameters
    ----------
    im : vector
        Intensity measure levels, ordered by increasing ``im``. Used only to
        check the ordering.
    H_im : vector
        Mean annual frequency of exceeding each ``im`` (non-increasing).
    fragility : vector
        Probability of collapse at each ``im``. Shape ``(n,)`` for one
        fragility curve, or ``(..., n)`` for a stack of curves on the same
        hazard.
    normalised : bool, optional
        If True (the default) divide by the total, so the contributions add
        up to one. If False return them in units of the MAFE (per year).

    Returns
    -------
    vector
        The contributions, with the shape of ``fragility``.

    Raises
    ------
    ValueError
        If ``im`` is not strictly increasing, ``H_im`` increases anywhere, or
        the lengths of ``im``, ``H_im`` and the last axis of ``fragility``
        differ.
    """
    im, H_im, fragility = (as_float(a)
                           for a in (im, H_im, fragility))
    if np.ndim(im) != 1 or np.size(im) != np.size(H_im):
        raise ValueError(f"im must be one-dimensional with the length of H_im "
                         f"({np.size(H_im)}), got shape {np.shape(im)}")
    if np.any(np.diff(im) <= 0):
        raise ValueError("im must be strictly increasing")
    _check_tabulated(H_im, fragility)

    # Trapezoid terms of the integral over the hazard, one per interval, then
    # the tail beyond the last level: the same terms numerical_mafe sums.
    intervals = 0.5 * (fragility[..., :-1] + fragility[..., 1:]) * (
        H_im[:-1] - H_im[1:])
    tail = H_im[-1] * fragility[..., -1:]
    contributions = np.concatenate([intervals, tail], axis=-1)

    if normalised:
        return contributions / contributions.sum(axis=-1, keepdims=True)
    return contributions


def risk_ratio_first_order(
        R_theta: scalar | vector, k1: scalar | vector,
        beta_1: scalar | vector, R_beta: scalar | vector = 1,
) -> scalar | vector:
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
    R_theta : scalar or vector
        Ratio of median capacities, :math:`\theta_2 / \theta_1`.
    k1 : scalar or vector
        Exponent of the first-order hazard curve.
    beta_1 : scalar or vector
        Dispersion of the reference (case 1) fragility.
    R_beta : scalar or vector, optional
        Ratio of dispersions, :math:`\beta_2 / \beta_1`. The default of one
        means the two fragilities have the same dispersion.

    Returns
    -------
    scalar or vector
        :math:`R_\lambda = \lambda_2 / \lambda_1`, with the broadcast shape
        of the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from hazrisk import risk
    >>> R_lambda = risk.risk_ratio_first_order(1.2, 2.5, 0.4)
    >>> round(float(R_lambda), 4)
    0.6339
    >>> grid = risk.risk_ratio_first_order(
    ...     *np.ix_([0.9, 1.0, 1.1], [2.0, 3.0], [0.2, 0.4], [1.0, 1.5]))
    >>> grid.shape
    (3, 2, 2, 2)
    """
    R_theta, k1, beta_1, R_beta = (as_float(a)
                                   for a in (R_theta, k1, beta_1, R_beta))
    return R_theta ** (-k1) * np.exp(
        0.5 * k1 ** 2 * (R_beta ** 2 - 1) * beta_1 ** 2)


def theta_ratio_first_order(
        R_lambda: scalar | vector, k1: scalar | vector,
        beta_1: scalar | vector, R_beta: scalar | vector = 1,
) -> scalar | vector:
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
    R_lambda : scalar or vector
        Target ratio of MAFEs, :math:`\lambda_2 / \lambda_1`.
    k1 : scalar or vector
        Exponent of the first-order hazard curve.
    beta_1 : scalar or vector
        Dispersion of the reference (case 1) fragility.
    R_beta : scalar or vector, optional
        Ratio of dispersions, :math:`\beta_2 / \beta_1`. The default of one
        means the two fragilities have the same dispersion.

    Returns
    -------
    scalar or vector
        :math:`R_\theta = \theta_2 / \theta_1`, with the broadcast shape of
        the inputs.

    Examples
    --------
    Halving the MAFE with the same dispersion, for ``k1 = 2.5``:

    >>> import numpy as np
    >>> from hazrisk import risk
    >>> round(float(risk.theta_ratio_first_order(0.5, 2.5, 0.4)), 4)
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
    R_lambda, k1, beta_1, R_beta = (as_float(a)
                                    for a in (R_lambda, k1, beta_1, R_beta))
    return np.exp(0.5 * k1 * (R_beta ** 2 - 1) * beta_1 ** 2
                  - np.log(R_lambda) / k1)
