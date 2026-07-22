import numpy as np
from hazrisk.typedefs import scalar, vector

def first_order_approx(im: scalar|vector, k0: scalar, k1: scalar
                       ) -> scalar|vector:
    """First order hazard function."""
    return k0 * im ** (-k1)


def first_order_dHdx(im: scalar|vector, k0: scalar, k1: scalar
                     ) -> scalar|vector:
    """Derivative of the first order hazard function."""
    return -k0 * k1 * im ** (-k1 - 1)


def second_order_approx(im: scalar|vector, k0: scalar, k1: scalar, k2: scalar
                        ) -> scalar|vector:
    """Second order hazard function."""
    return k0 * np.exp(-k2 * np.log(im) ** 2 - k1 * np.log(im))


def second_order_dHdx(im: scalar|vector, k0: scalar, k1: scalar, k2: scalar
                      ) -> scalar|vector:
    """Derivative of the second order hazard function.
    --> derived using the chain rule with: 
                    g(s) = -k2 * log(s) ** 2 - k1 * log(s)
    """
    return -(2 * k2 * np.log(im) + k1) / im * second_order_approx(im, k0, k1, k2)


def poe_to_mafe(poe: scalar|vector, t: scalar=1) -> float:
    """ converts probability of exceedance in time span `t` to mean annual 
    frequency of exceedance"""
    return -np.log(1-poe) / t


def mafe_to_poe(mafe: scalar|vector, t:scalar=1) -> float:
    """ converts mean annual frequency of exceedance to 
    probability of exceedance in time span `t` """
    return 1 - np.exp((-mafe * t))


def poe_to_rtp(poe: scalar|vector, t: scalar=1) -> float:
    """ converts probability of exceedance in time span `t` to mean return period"""
    return 1 / poe_to_mafe(poe, t)