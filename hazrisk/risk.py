import numpy as np
from hazrisk import hazard as haz
from hazrisk.typedefs import scalar, vector


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
    measure levels ``im`` so that each point ``(H_im[i], fragility[i])`` is a
    matched pair; ``im`` is retained only to document this shared support and is
    not used in the calculation.

    im : vector
        Intensity measure levels (ordered increasing). Not used directly; the
        fragility and hazard must be evaluated at these same levels.

    Parameters
    ----------
    H_im : vector
        Hazard curve values, i.e. the mean annual frequency of exceeding each
        ``im`` (monotonically decreasing).
    fragility : vector
        Fragility curve, i.e. P(collapse | im) at each ``im``.

    Returns
    -------
    scalar
        The mean annual frequency of collapse.
    """
    # -trapz because H_im decreases with im; add the tail beyond the last level.
    return -np.trapezoid(fragility, H_im) + H_im[-1] * fragility[-1]


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

    # return fragility * np.abs(dH_dim) * dim / mafe