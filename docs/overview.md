---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Overview

## The hazard → fragility → risk chain

A seismic risk calculation combines two curves:

- the **hazard curve** $\lambda(im)$, the mean annual frequency of exceeding
  each level of an intensity measure $im$ at the site (from a PSHA, e.g. with
  OpenQuake);
- the **fragility curve** $P(C \mid im)$, the probability that the structure
  reaches a limit state $C$ (e.g. collapse) when it is shaken at intensity $im$.

The **risk** is the mean annual frequency of the limit state, $\lambda_C$
(MAFE), the integral of the fragility over the hazard (see {doc}`theory`). The
package has one module per step: {mod}`hazrisk.hazard`,
{mod}`hazrisk.fragility` and {mod}`hazrisk.risk`.

```{code-cell} python
:tags: [remove-input]

# Figure of the chain for site B of the example data: the hazard curve, a
# fragility, and the risk per unit ln(im), whose area is the MAFE.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import fragility, hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

data = np.loadtxt(Path("..") / "examples" / "data" / "site_B.csv",
                  delimiter=",", skiprows=1)
im, H_im = data[:, 0], data[:, 1]
theta_c, beta = 0.3, 0.4
iml = np.geomspace(im[0], im[H_im > 0][-1], 400)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
axes[0].loglog(im, H_im, "o-", ms=3, color="0.25")
axes[0].set(xlabel="AvgSA [g]", ylabel=r"$\lambda(im)$ [1/yr]",
            title="hazard")
axes[1].semilogx(iml, fragility.lognormal_fragility_curve(iml, theta_c, beta),
                 color="C3")
axes[1].set(xlabel="AvgSA [g]", ylabel=r"$P(C \mid im)$",
            title=rf"fragility ($\theta_c$ = {theta_c} g, $\beta$ = {beta})")
axes[2].semilogx(iml, risk.risk_contribution_pchip(iml, theta_c, im, H_im, beta),
                 color="C0")
axes[2].set(xlabel="AvgSA [g]", ylabel=r"$d\lambda_C / d\ln im$ (normalised)",
            title="risk")
for ax in axes:
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
fig.tight_layout()
```

## Which function do I need?

| I have … and want … | Use |
|---|---|
| a tabulated hazard curve and a lognormal fragility; the MAFE | {func}`~hazrisk.risk.numerical_mafe_pchip` |
| a tabulated hazard curve and a fragility tabulated at the same levels; the MAFE | {func}`~hazrisk.risk.numerical_mafe` |
| the SAC/FEMA closed form for the MAFE | fit the hazard with {func}`~hazrisk.hazard.fit_first_order_regression` or {func}`~hazrisk.hazard.fit_second_order_regression` (or the `fit_*_points` variants), then {func}`~hazrisk.risk.mafe_first_order` or {func}`~hazrisk.risk.mafe_second_order` |
| the median capacity that gives a target MAFE | fit the hazard as above, then {func}`~hazrisk.risk.required_capacity_first_order` or {func}`~hazrisk.risk.required_capacity_second_order` |
| to know which intensities the risk comes from | {func}`~hazrisk.risk.risk_contribution_pchip` (a curve) or {func}`~hazrisk.risk.risk_contribution` (one value per tabulated level) |
| how much the capacity has to change for a target change in risk | {func}`~hazrisk.risk.theta_ratio_first_order` (and {func}`~hazrisk.risk.risk_ratio_first_order` for the reverse) |
| a smooth hazard curve, or its slope, between the tabulated levels | {func}`~hazrisk.hazard.pchip_approx`, {func}`~hazrisk.hazard.pchip_dHdx` |
| to convert a probability of exceedance in $t$ years to a frequency or a return period, and back | {func}`~hazrisk.hazard.poe_to_mafe`, {func}`~hazrisk.hazard.mafe_to_poe`, {func}`~hazrisk.hazard.poe_to_rtp` |
| a lognormal fragility curve | {func}`~hazrisk.fragility.lognormal_fragility_curve` |

The {doc}`examples/index` show each of these on real hazard curves.
