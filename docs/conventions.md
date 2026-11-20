# Conventions

## Units

hazrisk does not convert units. Intensity measure levels `im`, median
capacities `theta_c` and the bounds of an `iml_band` must all be in the same
unit (e.g. g). Frequencies (`H_im`, `mafe`, `target_mafe`, a `mafe_band`) are
per year, and the investigation time `t` of the probability conversions is in
years. Dispersions `beta` are the standard deviation of $\ln$ capacity, so they
have no unit.

## Ordering

Tabulated curves are ordered by **increasing `im`**, so `H_im` decreases
along the array. The integration and contribution functions rely on it (the
sign of the trapezoid sum depends on it) and raise a `ValueError` if `H_im`
increases anywhere. Equal neighbours are allowed. The fitting and PCHIP
functions require `im` to be strictly increasing.

## Logarithms

Every logarithm is natural ($\ln$). The fit coefficients `k0`, `k1`, `k2` are
those of the hazard written with $\ln im$, as in Vamvatsikos (2013); a fit made
with $\log_{10}$ has a different `k2`.

## Numbers and arrays

Every function takes Python numbers, lists or numpy arrays. The closed-form
functions follow numpy broadcasting: scalars give a scalar (a Python `float`),
arrays give an array of the broadcast shape. For a full grid of every
combination of several 1-D inputs, give each input its own axis with
{func}`numpy.ix_`:

```python
import numpy as np
from hazrisk import risk

R_lambda = np.array([0.5, 0.2, 0.1])
k1 = np.array([2.0, 3.0])
grid = risk.theta_ratio_first_order(*np.ix_(R_lambda, k1, [0.3, 0.5], [1.0]))
grid.shape  # (3, 2, 2, 1)
```

In typed code, the return annotation `float | ndarray` means a type checker asks
you to narrow the result (e.g. with `float(...)` or `np.asarray(...)`) before
using it as one or the other.

## Outside the tabulated range

The PCHIP functions ({func}`~hazrisk.hazard.pchip_approx`,
{func}`~hazrisk.hazard.pchip_dHdx`,
{func}`~hazrisk.risk.risk_contribution_pchip`) do not extrapolate: levels
below `im[0]` or above the last level with a positive hazard return `nan`. The
point-fitting functions raise a `ValueError` for points outside the curve. The
analytical approximations can be evaluated anywhere, but they are only
meaningful over the band they were fitted to.

## Trailing zeros from OpenQuake

OpenQuake writes a hazard of exactly zero at the highest levels, where the
probability of exceedance is below the precision it stores. Every function
accepts such a trailing run of zeros, and uses it as follows:

- **Integration** ({func}`~hazrisk.risk.numerical_mafe`,
  {func}`~hazrisk.risk.numerical_mafe_pchip`,
  {func}`~hazrisk.risk.risk_contribution`) uses the zeros: the integral runs
  down to the first level where the hazard is zero. Further zeros add nothing.
- **Fitting** (the `fit_*` functions in {mod}`hazrisk.hazard`) drops the
  zeros, because the fits work with the logarithm of the hazard.
- **PCHIP evaluation** ({func}`~hazrisk.hazard.pchip_approx`,
  {func}`~hazrisk.hazard.pchip_dHdx`,
  {func}`~hazrisk.risk.risk_contribution_pchip`) interpolates up to the last
  level with a positive hazard and returns `nan` above it.

A zero followed by a positive value is not a trailing run, and raises a
`ValueError`.

Keeping the zeros or trimming them off before calling
{func}`~hazrisk.risk.numerical_mafe_pchip` gives different results when the
fragility changes a lot between the last positive level and the first zero
level. {doc}`examples/trailing_zeros` shows where the difference comes from
and how to choose.

## Errors

hazrisk raises a `ValueError`, rather than returning a wrong number, when:

- a tabulated hazard curve increases somewhere, is not one-dimensional, or does
  not match the length of the fragility (integration and contribution
  functions), or is not strictly positive and strictly decreasing apart from a
  trailing run of zeros (fitting and PCHIP functions);
- a fitting band holds too few levels, or both an intensity band and a
  frequency band are given;
- $1 + 2 k_2 \beta^2 \le 0$ in a second-order closed form;
- a target frequency is out of reach of the fitted second-order curve.

The message says which check failed and, for the closed forms, for how many
elements.
