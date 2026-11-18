import numpy as np
from hazrisk import hazard as haz
from hazrisk.fragility import lognormal_fragility_curve
from hazrisk._typing import scalar, vector


def required_capacity_first_order(target_mafe: scalar, k0: scalar, k1: scalar,
                                  beta: scalar) -> scalar:
    """Calculate theta_c from mean annual frequency of exceedance for first order 
    hazard."""
    return np.exp((0.5 * k1 * beta ** 2) - (np.log(target_mafe / k0) / k1))


def mafe_first_order(theta_c: scalar, k0: scalar, k1: scalar, beta: scalar
                     ) -> scalar:
    """Calculate mean annual frequency of collapse for first order hazard."""
    return k0 * theta_c ** (-k1) * np.exp(0.5 * (beta * k1) ** 2)


def p_second_order(k2: scalar, beta: scalar) -> scalar:
    """Calculate the p factor for second order risk approximation."""
    return 1 / (1 + 2 * k2 * beta ** 2)


def required_capacity_second_order(target_mafe: scalar, k0: scalar, k1: scalar, 
                                   k2: scalar, beta: scalar) -> scalar:
    """Calculate theta_c from mean annual frequency of exceedance for the 
    second order hazard."""
    p = p_second_order(k2, beta)
    return np.exp((1 / (2 * k2)) * (-k1 + np.sqrt((k1 ** 2 / p) - (4 * k2 / p) \
                * np.log(target_mafe / (k0 * np.sqrt(p))))))


def mafe_second_order(theta_c: scalar, k0: scalar, k1: scalar, k2: scalar, 
                      beta: scalar) -> scalar:
    """Calculate mean annual frequency of collapse for second order hazard."""
    p = p_second_order(k2, beta)
    H_theta = haz.second_order_approx(theta_c, k0, k1, k2)
    return np.sqrt(p) * k0 ** (1 - p) * H_theta ** p * \
               np.exp(0.5 * p * k1 ** 2 * beta **2)


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
    """
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


def risk_contribution(im: scalar|vector, H_im: vector, fragility: vector,  
                      normalised: bool=True) -> scalar|vector:
    """Calculate the risk contribution using the left Riemann Sum.
    -- fragility: fragility curve values
    -- dH_dx: derivative of the hazard curve
    -- step: constant im step assumed for im values corresponding to the 
             fragility curve and dH_dx
    -- mafe: mean annual frequency of exceedance of limit state
    """

    dim = np.diff(im)
    dH = np.diff(H_im)
    dH_dim = dH / dim

    contributions = fragility[:len(dH_dim)] * np.abs(dH_dim) * dim
    contributions = np.append(contributions, fragility[-1] * H_im[-1])
    
    if normalised:
        mafe = sum(contributions)
        return contributions / mafe
    return


def risk_ratio_first_order(
        R_theta: scalar|vector, k1:scalar|vector, 
        beta_1: scalar|vector|None, R_beta: scalar|vector=1
        ) -> np.ndarray:
    """ calculate the risk ratio between two median capacities for the first order hazard
    
    Any of them can be vector, in which case the output is a multidimensional array with the same shape as the broadcasted inputs.

    R_theta: scalar|vector (n) - ratio of median capacities (theta_2/theta_1)
    k1: scalar|vector (m) - first order hazard exponent
    beta_1: scalar|vector (p) - standard deviation of the first order hazard
    R_beta: scalar|vector (q) - ratio of median fragilities
    

    Returns:
    -------
    R_lambda: scalar|np.ndarray shape(n, m, p, q) - ratio of mean annual frequencies of exceedance (lambda_2/lambda_1)
    """

    R_theta = np.asarray(R_theta)
    k1 = np.asarray(k1)
    beta_1 = np.asarray(beta_1)
    R_beta = np.asarray(R_beta)

    R_theta_4d = np.expand_dims(R_theta, axis=(1, 2, 3))
    k1_4d = np.expand_dims(k1, axis=(0, 2, 3))
    beta_1_4d = np.expand_dims(beta_1, axis=(0, 1, 3))
    R_beta_4d = np.expand_dims(R_beta, axis=(0, 1, 2))

    R_lambda = R_theta_4d ** (-k1_4d) * np.exp(0.5 * k1_4d ** 2 * (R_beta_4d ** 2 - 1) * beta_1_4d ** 2)

    return R_lambda


def theta_ratio_first_order(
        R_lambda: scalar|vector, k1:scalar|vector, 
        beta_1: scalar|vector|None, R_beta: scalar|vector=1
        ) -> np.ndarray:
    """ calculate the ratio of median capacities to achieve a desired change in risk level
    using the first order hazard approximation.
    
    Any of them can be vector, in which case the output is a multidimensional array with the same shape as the broadcasted inputs.

    R_lambda: scalar|vector (n) - ratio of mean annual frequencies of exceedance (lambda_2/lambda_1)
    k1: scalar|vector (m) - first order hazard exponent
    beta_1: scalar|vector (p) - standard deviation of the first order hazard
    R_beta: scalar|vector (q) - ratio of median fragilities
    

    Returns:
    -------
    R_theta: scalar|np.ndarray shape(n, m, p, q) - ratio of median capacities (theta_2/theta_1)
    """

    R_lambda = np.array(R_lambda)
    k1 = np.array(k1)
    beta_1 = np.array(beta_1)
    R_beta = np.array(R_beta)

    R_lambda_4d = np.expand_dims(R_lambda, axis=(1, 2, 3))
    k1_4d = np.expand_dims(k1, axis=(0, 2, 3))
    beta_1_4d = np.expand_dims(beta_1, axis=(0, 1, 3))
    R_beta_4d = np.expand_dims(R_beta, axis=(0, 1, 2))

    R_theta = np.exp(0.5 * k1_4d * (R_beta_4d ** 2 - 1) * beta_1_4d ** 2 - np.log(R_lambda_4d) / k1_4d)

    return R_theta
