from scipy.stats import lognorm
from hazrisk.typedefs import scalar, vector

def lognormal_fragility_curve(im: scalar|vector, median: scalar, beta: scalar):
    """Calculate the probability of limit state exceedance of an im or ims
    assuming the fragility curve follows a lognormal cdf.
    """
    return lognorm.cdf(im, s=beta, scale=median)