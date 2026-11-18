r"""Fragility curves: the probability of a limit state given the intensity.

The middle link of the hazard -> fragility -> risk chain. A fragility curve
:math:`P(LS \mid im)` gives the probability that a structure reaches a limit
state (for example collapse) when shaken at intensity measure level ``im``.
The risk functions in :mod:`hazrisk.risk` combine it with a hazard curve from
:mod:`hazrisk.hazard`.
"""

from numpy.typing import ArrayLike
from scipy.stats import lognorm

from hazrisk._typing import FloatOrArray, as_float

__all__ = ["lognormal_fragility_curve"]


def lognormal_fragility_curve(
    im: ArrayLike, median: ArrayLike, beta: ArrayLike
) -> FloatOrArray:
    r"""Lognormal fragility curve: probability of the limit state given ``im``.

    The capacity is taken as lognormally distributed with median
    :math:`\theta` and dispersion :math:`\beta` (the standard deviation of
    its natural logarithm), so the probability that intensity ``im`` reaches
    the limit state is

    .. math::

        P(LS \mid im) = \Phi\left(\frac{\ln(im / \theta)}{\beta}\right),

    with :math:`\Phi` the standard normal cumulative distribution function.
    This is the fragility the closed-form solutions in :mod:`hazrisk.risk`
    assume (Cornell et al. 2002; Vamvatsikos 2013).

    Parameters
    ----------
    im : array_like
        Intensity measure levels (any unit, > 0).
    median : array_like
        Median capacity :math:`\theta`, in the unit of ``im``.
    beta : array_like
        Lognormal dispersion :math:`\beta` (> 0).

    Returns
    -------
    float or ndarray
        Probability of reaching the limit state at each ``im``, with the
        broadcast shape of the inputs.

    References
    ----------
    Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
    Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
    moment frame guidelines. *Journal of Structural Engineering*, 128(4),
    526-533. https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)

    Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
    solutions with second-order hazard approximation. *Earthquake Engineering
    & Structural Dynamics*, 42(8), 1171-1188. https://doi.org/10.1002/eqe.2265

    Examples
    --------
    At the median the probability is one half:

    >>> from hazrisk import fragility
    >>> print(fragility.lognormal_fragility_curve([0.15, 0.3, 0.6], 0.3, 0.4))
    [0.04155957 0.5        0.95844043]
    """
    im, median, beta = (as_float(a) for a in (im, median, beta))
    # scipy's lognormal: shape s is the dispersion beta (the standard deviation
    # of ln im), scale is the median.
    result: FloatOrArray = lognorm.cdf(im, s=beta, scale=median)
    return result
