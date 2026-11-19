r"""Conversions between probability of exceedance, annual frequency and return period.

Events are taken to occur as a Poisson process in time. A mean annual
frequency :math:`\lambda` then fixes the probability of at least one event in
a time span of :math:`t` years,

.. math::

    P = 1 - e^{-\lambda t},

and the mean return period :math:`T_R = 1 / \lambda`. The six functions here
convert between these three quantities in every direction.

The conversions apply to any mean annual frequency, not only to the hazard.
:math:`\lambda` can be the hazard :math:`H(im)` at one intensity level (for
example the 10 % in 50 years level), or the mean annual frequency of a limit
state, :math:`\lambda_C`, from :mod:`hazrisk.risk` (for example to state a
collapse risk as a 50-year probability of collapse).

Units: frequencies per year, return periods and the time span ``t`` in years,
probabilities as fractions (0.1, not 10 %).
"""

import numpy as np
from numpy.typing import ArrayLike

from hazrisk._typing import FloatOrArray, as_float

# The public interface of the module, in the order the docs list it.
__all__ = [
    "poe_to_mafe",
    "mafe_to_poe",
    "poe_to_rtp",
    "rtp_to_poe",
    "mafe_to_rtp",
    "rtp_to_mafe",
]


def poe_to_mafe(poe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Mean annual frequency from a probability of exceedance in ``t`` years.

    Solving :math:`P = 1 - e^{-\lambda t}` for :math:`\lambda`
    (Baker et al. 2021):

    .. math::

        \lambda = -\frac{\ln(1 - P)}{t}.

    Parameters
    ----------
    poe : array_like
        Probability of at least one exceedance in ``t`` years, in [0, 1).
    t : array_like, optional
        Time span in years (> 0, default 1).

    Returns
    -------
    float or ndarray
        Mean annual frequency of exceedance (per year), with the broadcast
        shape of the inputs.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    10 % in 50 years:

    >>> from hazrisk import convert
    >>> print(f"{convert.poe_to_mafe(0.10, t=50):.4e}")
    2.1072e-03
    """
    # as_float lets lists and integer arrays through the arithmetic below.
    poe, t = as_float(poe), as_float(t)
    return -np.log(1 - poe) / t


def mafe_to_poe(mafe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Probability of exceedance in ``t`` years from a mean annual frequency.

    The inverse of :func:`poe_to_mafe`, for a Poisson process
    (Baker et al. 2021):

    .. math::

        P = 1 - e^{-\lambda t}.

    Parameters
    ----------
    mafe : array_like
        Mean annual frequency of exceedance (per year, >= 0).
    t : array_like, optional
        Time span in years (> 0, default 1).

    Returns
    -------
    float or ndarray
        Probability of at least one exceedance in ``t`` years, with the
        broadcast shape of the inputs.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    The 475-year return period corresponds to about 10 % in 50 years:

    >>> from hazrisk import convert
    >>> print(f"{convert.mafe_to_poe(1 / 475, t=50):.4f}")
    0.0999
    """
    mafe, t = as_float(mafe), as_float(t)
    return 1 - np.exp(-mafe * t)


def poe_to_rtp(poe: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Mean return period from a probability of exceedance in ``t`` years.

    The return period is the reciprocal of the mean annual frequency
    (:func:`poe_to_mafe`; Baker et al. 2021):

    .. math::

        T_R = \frac{1}{\lambda} = -\frac{t}{\ln(1 - P)}.

    Parameters
    ----------
    poe : array_like
        Probability of at least one exceedance in ``t`` years, in (0, 1).
    t : array_like, optional
        Time span in years (> 0, default 1).

    Returns
    -------
    float or ndarray
        Mean return period (years), with the broadcast shape of the inputs.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    The familiar design levels, 10 % and 2 % in 50 years:

    >>> from hazrisk import convert
    >>> print([round(float(T)) for T in convert.poe_to_rtp([0.10, 0.02], t=50)])
    [475, 2475]
    """
    # poe_to_mafe casts the inputs to float.
    return 1 / poe_to_mafe(poe, t)


def rtp_to_poe(rtp: ArrayLike, t: ArrayLike = 1) -> FloatOrArray:
    r"""Probability of exceedance in ``t`` years from a mean return period.

    The inverse of :func:`poe_to_rtp`: with :math:`\lambda = 1 / T_R` in
    :math:`P = 1 - e^{-\lambda t}` (Baker et al. 2021),

    .. math::

        P = 1 - e^{-t / T_R}.

    Parameters
    ----------
    rtp : array_like
        Mean return period (years, > 0).
    t : array_like, optional
        Time span in years (> 0, default 1).

    Returns
    -------
    float or ndarray
        Probability of at least one exceedance in ``t`` years, with the
        broadcast shape of the inputs.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    The 475-year and 2475-year return periods as probabilities in 50 years:

    >>> from hazrisk import convert
    >>> print([round(float(P), 3) for P in convert.rtp_to_poe([475, 2475], t=50)])
    [0.1, 0.02]
    """
    rtp, t = as_float(rtp), as_float(t)
    return 1 - np.exp(-t / rtp)


def mafe_to_rtp(mafe: ArrayLike) -> FloatOrArray:
    r"""Mean return period from a mean annual frequency.

    .. math::

        T_R = \frac{1}{\lambda}

    (Baker et al. 2021).

    Parameters
    ----------
    mafe : array_like
        Mean annual frequency of exceedance (per year, > 0).

    Returns
    -------
    float or ndarray
        Mean return period (years), with the shape of ``mafe``.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    A collapse frequency of 2e-4 per year:

    >>> from hazrisk import convert
    >>> print(f"{convert.mafe_to_rtp(2e-4):.0f}")
    5000
    """
    mafe = as_float(mafe)
    return 1 / mafe


def rtp_to_mafe(rtp: ArrayLike) -> FloatOrArray:
    r"""Mean annual frequency from a mean return period.

    The inverse of :func:`mafe_to_rtp` (Baker et al. 2021):

    .. math::

        \lambda = \frac{1}{T_R}.

    Parameters
    ----------
    rtp : array_like
        Mean return period (years, > 0).

    Returns
    -------
    float or ndarray
        Mean annual frequency of exceedance (per year), with the shape of
        ``rtp``.

    References
    ----------
    Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
    Risk Analysis*. Cambridge University Press.

    Examples
    --------
    >>> from hazrisk import convert
    >>> print(f"{convert.rtp_to_mafe(2475):.4e}")
    4.0404e-04
    """
    rtp = as_float(rtp)
    return 1 / rtp
