import numpy as np
from typedefs import Scalar, Vector

def first_order_approx(im: Scalar|Vector, k0: Scalar, k1: Scalar
                       ) -> Scalar|Vector:
    """First order hazard function."""
    return k0 * im ** (-k1)


def first_order_dHdx(im: Scalar|Vector, k0: Scalar, k1: Scalar
                     ) -> Scalar|Vector:
    """Derivative of the first order hazard function."""
    return -k0 * k1 * im ** (-k1 - 1)


def second_order_approx(im: Scalar|Vector, k0: Scalar, k1: Scalar, k2: Scalar
                        ) -> Scalar|Vector:
    """Second order hazard function."""
    return k0 * np.exp(-k2 * np.log(im) ** 2 - k1 * np.log(im))


def second_order_dHdx(im: Scalar|Vector, k0: Scalar, k1: Scalar, k2: Scalar
                      ) -> Scalar|Vector:
    """Derivative of the second order hazard function.
    --> derived using the chain rule with: 
                    g(s) = -k2 * log(s) ** 2 - k1 * log(s)
    """
    return -(2 * k2 * np.log(im) + k1) / im * second_order_approx(im, k0, k1, k2)


