---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Required capacity for a target MAFC

The inverse question: what median capacity $\theta_c$ gives a target mean
annual frequency of collapse (MAFC)? The closed forms answer it directly
({func}`~hazrisk.risk.required_capacity_first_order`,
{func}`~hazrisk.risk.required_capacity_second_order`). This page compares them
with the capacity that gives the target on the tabulated curve, found by
solving {func}`~hazrisk.risk.numerical_mafe_pchip` for $\theta_c$.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from hazrisk import hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site, as written by OpenQuake."""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def required_capacity_numerical(target, im, H_im, beta):
    """Median capacity at which the PCHIP-integrated MAFC equals the target.

    The MAFC decreases monotonically with theta_c, so a bracketing root
    finder on ln(theta_c) is enough. The bracket stays inside the tabulated
    curve.
    """
    def f(x):
        return np.log(risk.numerical_mafe_pchip(np.exp(x), im, H_im, beta) / target)

    pos = H_im > 0
    return float(np.exp(brentq(f, np.log(im[0]) + 1, np.log(im[pos][-1]) - 1)))


band = (1e-4, 1e-1)  # fitting band: return periods 10 to 10 000 yr
beta = 0.4
# Target MAFC, 1/yr. Lower targets need capacities so close to the end of
# site A's curve that the tail term would dominate the numerical result.
targets = np.geomspace(3e-5, 1e-3, 25)
```

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, label in zip(axes, "ABC"):
    im, H_im = load_site(label)
    pos = H_im > 0
    k0_1, k1_1, _ = hazard.fit_first_order_regression(im[pos], H_im[pos],
                                                      mafe_band=band)
    k0, k1, k2, _ = hazard.fit_second_order_regression(im[pos], H_im[pos],
                                                       mafe_band=band)
    first = risk.required_capacity_first_order(targets, k0_1, k1_1, beta)
    second = risk.required_capacity_second_order(targets, k0, k1, k2, beta)
    numerical = [required_capacity_numerical(t, im, H_im, beta) for t in targets]

    ax.loglog(targets, numerical, "k-", lw=2, label="numerical (PCHIP)")
    ax.loglog(targets, first, "--", color="C0", label="1st-order closed form")
    ax.loglog(targets, second, "-", color="C3", label="2nd-order closed form")
    ax.set(xlabel="target MAFC [1/yr]", ylabel=r"required $\theta_c$ [g]",
           title=rf"site {label}, $\beta$ = {beta}")
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.legend(fontsize=8)
fig.tight_layout()
```

The same pattern as for the forward calculation (see
{doc}`closed_form_vs_numerical`): the first-order closed form asks for more
capacity than the tabulated curve needs (by 30 to 90 % here), and the
second-order closed form is within a few per cent of the numerical answer. For a target of
$10^{-4}$ per year on each site:

```{code-cell} python
target = 1e-4
print(f"{'site':>4}  {'numerical':>9}  {'1st order':>9}  {'2nd order':>9}   [g]")
for label in "ABC":
    im, H_im = load_site(label)
    pos = H_im > 0
    k0_1, k1_1, _ = hazard.fit_first_order_regression(im[pos], H_im[pos],
                                                      mafe_band=band)
    k0, k1, k2, _ = hazard.fit_second_order_regression(im[pos], H_im[pos],
                                                       mafe_band=band)
    print(f"{label:>4}  {required_capacity_numerical(target, im, H_im, beta):9.3f}"
          f"  {risk.required_capacity_first_order(target, k0_1, k1_1, beta):9.3f}"
          f"  {risk.required_capacity_second_order(target, k0, k1, k2, beta):9.3f}")
```

Each closed-form capacity is the exact inverse of its forward calculation:
putting it back into {func}`~hazrisk.risk.mafe_second_order` returns the
target.

```{code-cell} python
theta = risk.required_capacity_second_order(target, k0, k1, k2, beta)
print(f"{risk.mafe_second_order(theta, k0, k1, k2, beta):.6e}")
```
