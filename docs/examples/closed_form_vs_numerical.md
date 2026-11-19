---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Closed form against numerical integration

The SAC/FEMA closed forms give the mean annual frequency of the limit state
(here called MAFC, for collapse) in one line, but only on a fitted hazard
curve. Numerical integration uses the tabulated curve itself. This page
compares the two on the three {doc}`example sites <../data>` across a range of
dispersions $\beta$, and then looks at how the numerical result depends on the
interpolation between the tabulated levels.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import fragility, hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site"""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


band = (1e-4, 1e-1)  # fitting band: return periods 10 to 10 000 yr
betas = np.linspace(0.1, 0.7, 25)
```

## Across the dispersion

For each site the median capacity is put where the curve has a 2 % in 50 year
frequency, so the three are comparable. The closed forms use the regression
fits over the 10 to 10 000 year band; the numerical result is
{func}`~hazrisk.risk.numerical_mafe_pchip`, which handles the trailing zeros
itself (the fits need them removed).

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
for ax, label in zip(axes, "ABC"):
    im, H_im = load_site(label)
    pos = H_im > 0
    k0_1, k1_1, _ = hazard.fit_first_order_regression(im[pos], H_im[pos],
                                                      mafe_band=band)
    k0, k1, k2, _ = hazard.fit_second_order_regression(im[pos], H_im[pos],
                                                       mafe_band=band)
    # Median capacity at the 2 % in 50 yr level of the numerical curve.
    mafe_2in50 = hazard.poe_to_mafe(0.02, t=50)
    theta_c = float(np.exp(np.interp(np.log(mafe_2in50), np.log(H_im[pos])[::-1],
                                     np.log(im[pos])[::-1])))

    numerical = risk.numerical_mafe_pchip(theta_c, im, H_im, betas)
    first = risk.mafe_first_order(theta_c, k0_1, k1_1, betas)
    second = risk.mafe_second_order(theta_c, k0, k1, k2, betas)

    ax.semilogy(betas, numerical, "k-", lw=2, label="numerical (PCHIP)")
    ax.semilogy(betas, first, "--", color="C0", label="1st-order closed form")
    ax.semilogy(betas, second, "-", color="C3", label="2nd-order closed form")
    ax.set(xlabel=r"dispersion $\beta$", ylabel="MAFC [1/yr]",
           title=rf"site {label}, $\theta_c$ = {theta_c:.3g} g")
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.legend(fontsize=8)
fig.tight_layout()
```

With $\theta_c$ at the 2 % in 50 year level, the risk comes from intensities
inside the fitted band, and the second-order closed form stays within a few
per cent of the numerical result on all three sites and for every $\beta$. The
first-order regression over such a wide fitting band does not pass through the 
actual hazard curve at $\theta_c$, so it is off even at small $\beta$ (by about 
a third on site B), and its error changes with $\beta$ as the risk moves to 
intensities where the power law and the curve part ways.

The closed forms are only as good as the fit over the intensities that carry
the risk. In the {doc}`../quickstart`, with $\theta_c$ = 0.3 g on site B, most
of the risk comes from hazard rarer than the fitted band (see
{doc}`risk_contribution`), and the second-order closed form overestimates the
risk by about a third.

## Interpolating the tabulated curve

{func}`~hazrisk.risk.numerical_mafe` integrates the fragility against the
hazard at whatever levels it is given. On the 25 levels of the provided PSHA 
curves that is
coarse. Two ways to refine the curve first are straight lines between the
levels in log-log space, and PCHIP in log-log space (what
{func}`~hazrisk.risk.numerical_mafe_pchip` does). The relative difference from
a very fine PCHIP reference shows how quickly each converges as the number of
refined levels grows (site B, $\beta = 0.4$):

```{code-cell} python
im, H_im = load_site("B")
pos = H_im > 0
im_p, H_p = im[pos], H_im[pos]
theta_c, beta = 0.3, 0.4


def mafe_straight_lines(n_im):
    """MAFC with the hazard interpolated by straight lines in log-log space."""
    iml = np.geomspace(im_p[0], im_p[-1], n_im)
    H_lin = np.exp(np.interp(np.log(iml), np.log(im_p), np.log(H_p)))
    P = fragility.lognormal_fragility_curve(iml, theta_c, beta)
    return risk.numerical_mafe(H_lin, P)


reference = risk.numerical_mafe_pchip(theta_c, im, H_im, beta, n_im=200_000)
tabulated = risk.numerical_mafe(
    H_im, fragility.lognormal_fragility_curve(im, theta_c, beta))
n_levels = np.unique(np.geomspace(25, 20_000, 30).astype(int))
err_lin = [mafe_straight_lines(n) / reference - 1 for n in n_levels]
err_pchip = [risk.numerical_mafe_pchip(theta_c, im, H_im, beta, n_im=n)
             / reference - 1 for n in n_levels]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.semilogx(n_levels, 100 * np.array(err_lin), "o-", ms=3, color="C0",
            label="straight lines in log-log")
ax.semilogx(n_levels, 100 * np.array(err_pchip), "o-", ms=3, color="C3",
            label="PCHIP in log-log")
ax.axhline(100 * (tabulated / reference - 1), color="0.4", ls=":",
           label="the 25 tabulated levels only")
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="number of refined levels", ylabel="difference from reference [%]",
       title=r"site B, $\theta_c$ = 0.3 g, $\beta$ = 0.4")
ax.grid(True, which="both", lw=0.4, alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
print(f"reference MAFC: {reference:.4e} /yr")
print(f"25 tabulated levels: {100 * (tabulated / reference - 1):+.1f} %")
print(f"straight lines, {n_levels[-1]} levels: {100 * err_lin[-1]:+.2f} %")
print(f"PCHIP, {n_levels[-1]} levels: {100 * err_pchip[-1]:+.4f} %")
```

Integrating the 25 tabulated levels directly overestimates the risk here by
about 20 %: between the levels the trapezoid rule draws straight lines in
linear space. Refining with straight lines in log-log space converges, but to a
different answer from PCHIP (about 1 % lower here), because it is a different
curve, whose slope jumps at every tabulated level (see
{doc}`risk_contribution`). PCHIP gives a smooth, monotone curve through the same
points and converges quickly: with the default of 2000 levels of
{func}`~hazrisk.risk.numerical_mafe_pchip`, the result is within 0.01 % of the
reference.
