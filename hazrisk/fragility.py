from scipy.stats import lognorm
from .types import Scalar, Vector

def lognormal_fragility_curve(im: Scalar|Vector, median: Scalar, beta: Scalar):
    """Calculate the probability of limit state exceedance of an im or ims
    assuming the fragility curve follows a lognormal cdf.
    """
    return lognorm.cdf(im, beta, scale=median)