---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quick start

This page computes the mean annual frequency of collapse of a structure in two
ways: with the SAC/FEMA closed form on a fitted hazard curve, and by numerical
integration of the tabulated curve. The hazard curve is ten levels of site B of
the {doc}`example data <data>`, rounded and written out so the example runs on
its own.

```{code-cell} python
import numpy as np
from hazrisk import hazard, risk

# A tabulated hazard curve H(im) (e.g. from OpenQuake): im in g, mean annual
# frequency of exceedance per year, ordered by increasing im.
im = np.array([0.00213, 0.0044, 0.00909, 0.0188, 0.0387,
               0.08, 0.165, 0.341, 0.704, 1.45])
H_im = np.array([3.43e-2, 1.73e-2, 7.24e-3, 2.44e-3, 6.49e-4,
                 1.36e-4, 1.97e-5, 1.6e-6, 5.55e-8, 4.42e-10])
```

Fit the second-order approximation (Vamvatsikos 2013) by least squares over the
return periods from 10 to 10 000 years, i.e. frequencies from $10^{-4}$ to
$10^{-1}$ per year:

```{code-cell} python
k0, k1, k2, r2 = hazard.fit_second_order_regression(im, H_im, mafe_band=(1e-4, 1e-1))
print(f"k0 = {k0:.3e}, k1 = {k1:.3f}, k2 = {k2:.3f}, r2 = {r2:.4f}")
```

The fragility is lognormal with a median capacity of 0.3 g and a dispersion of
0.4:

```{code-cell} python
theta_c, beta = 0.3, 0.4
print(f"closed form: {risk.mafe_second_order(theta_c, beta, k0, k1, k2):.3e} /yr")
print(f"numerical:   {risk.numerical_mafe_pchip(theta_c, im, H_im, beta):.3e} /yr")
```

The two answers differ because the closed form is only as good as the hazard
fit: here most of the risk comes from hazard levels rarer than the fitted band,
where the fit overestimates the hazard. The numerical integration uses the
tabulated curve itself. The {doc}`examples/closed_form_vs_numerical` example
looks at this in more detail.

The inverse question, the median capacity needed for a target frequency of
collapse of $2 \times 10^{-4}$ per year (on the fitted curve):

```{code-cell} python
theta_req = risk.required_capacity_second_order(2e-4, beta, k0, k1, k2)
print(f"required median capacity: {theta_req:.3f} g")
```

Every function also takes arrays, so a range of capacities is one call:

```{code-cell} python
theta = np.array([0.2, 0.3, 0.5, 0.8])
print(risk.numerical_mafe_pchip(theta, im, H_im, beta))
```
