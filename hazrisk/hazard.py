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


