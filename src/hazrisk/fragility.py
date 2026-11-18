"""Fragility curves: the probability of a limit state given the intensity.

The middle link of the hazard -> fragility -> risk chain. A fragility curve
``P(LS | im)`` gives the probability that a structure reaches a limit state
(for example collapse) when shaken at intensity measure level ``im``. The risk
functions in :mod:`hazrisk.risk` combine it with a hazard curve from
:mod:`hazrisk.hazard`.
"""

import numpy as np
from scipy.stats import lognorm

from hazrisk._typing import as_float, scalar, vector

__all__ = ["lognormal_fragility_curve"]


def lognormal_fragility_curve(im: scalar|vector, median: scalar, beta: scalar):
    """Calculate the probability of limit state exceedance of an im or ims
    assuming the fragility curve follows a lognormal cdf.
    """
    im, median, beta = (as_float(a) for a in (im, median, beta))
    # scipy's lognormal: shape s is the dispersion beta (the standard deviation
    # of ln im), scale is the median.
    return lognorm.cdf(im, s=beta, scale=median)
