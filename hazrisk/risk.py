import numpy as np
from hazrisk import hazard as haz
from typedefs import Scalar, Vector


def required_capacity_first_order(target_mafe: Scalar, k0: Scalar, k1: Scalar,
                                  beta: Scalar) -> Scalar:
    """Calculate theta_c from mean annual frequency of exceedance for first order 
    hazard."""
    return np.exp((0.5 * k1 * beta ** 2) - (np.log(target_mafe / k0) / k1))


def mafe_first_order(theta_c: Scalar, k0: Scalar, k1: Scalar, beta: Scalar
                     ) -> Scalar:
    """Calculate mean annual frequency of collapse for first order hazard."""
    return k0 * theta_c ** (-k1) * np.exp(0.5 * (beta * k1) ** 2)


def p_second_order(k2: Scalar, beta: Scalar) -> Scalar:
    """Calculate the p factor for second order risk approximation."""
    return 1 / (1 + 2 * k2 * beta ** 2)


def required_capacity_second_order(target_mafe: Scalar, k0: Scalar, k1: Scalar, 
                                   k2: Scalar, beta: Scalar) -> Scalar:
    """Calculate theta_c from mean annual frequency of exceedance for the 
    second order hazard."""
    p = p_second_order(k2, beta)
    return np.exp((1 / (2 * k2)) * (-k1 + np.sqrt((k1 ** 2 / p) - (4 * k2 / p) \
                * np.log(target_mafe / (k0 * np.sqrt(p))))))


def mafe_second_order(theta_c: Scalar, k0: Scalar, k1: Scalar, k2: Scalar, 
                      beta: Scalar) -> Scalar:
    """Calculate mean annual frequency of collapse for second order hazard."""
    p = p_second_order(k2, beta)
    H_theta = haz.second_order_approx(theta_c, k0, k1, k2)
    return np.sqrt(p) * k0 ** (1 - p) * H_theta ** p * \
               np.exp(0.5 * p * k1 ** 2 * beta **2)


def numerical_mafe(im: Vector, H_im: Vector, fragility: Vector):
    """Calculate the mean annual frequency of collapse using numerical 
    integration.
    """
    dim = np.diff(im)
    dH = np.diff(H_im)
    dH_dim = dH / dim
    dH_dim = np.append(dH_dim, dH_dim[-1])
    y = fragility[:len(dH_dim)] * np.abs(dH_dim)
    # add last term to capture the contribution of all imls larger than the last im
    return np.trapezoid(y, im) + H_im[-1] * fragility[-1]


def risk_contribution(im: Scalar|Vector, H_im: Vector, fragility: Vector,  
                      normalised: bool=True) -> Scalar|Vector:
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