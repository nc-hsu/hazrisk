---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Fitting hazard curves

The closed-form risk solutions need the hazard curve as a formula: the
first-order power law or the second-order log-quadratic (see {doc}`../theory`).
This page fits both to the three {doc}`example curves <../data>` and shows how
the result depends on the choices made: fitting through points or by
regression, where the points go, and how wide the band is.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import hazard

# White figure background, so the figures read well in the dark theme too.
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

# The example data, relative to this page (docs/examples/).
DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site, without the trailing zeros."""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    im, H_im = data[:, 0], data[:, 1]
    # The fitting functions need a strictly positive curve.
    keep = H_im > 0
    return im[keep], H_im[keep]


def hazard_axes(ax, im, H_im, title):
    """Plot the numerical curve on log-log axes, with labels."""
    ax.loglog(im, H_im, "o-", color="0.25", ms=3, lw=1.2, zorder=5,
              label="numerical hazard curve")
    ax.set(xlabel="AvgSA(0.3 s) [g]", ylabel="MAFE [1/yr]", title=title)
    ax.set_ylim(H_im[-1] / 3, H_im[0] * 3)
    ax.grid(True, which="both", lw=0.4, alpha=0.4)


im, H_im = load_site("B")
fine = np.geomspace(im[0], im[-1], 300)
```

## Through points or by regression

A fit through points passes exactly through two (first order) or three (second
order) points of the curve; here the 10 %, 5 % and 2 % in 50 year levels,
converted to frequencies with {func}`~hazrisk.hazard.poe_to_mafe`. A
regression fits all the tabulated levels in a band, here return periods of 10
to 10 000 years. Site B has only 25 levels, so the 475 to 2475 year band holds
too few of them for a regression: the band has to be wider.

```{code-cell} python
mafe_anchors = hazard.poe_to_mafe(np.array([0.10, 0.05, 0.02]), t=50)
band = (1e-4, 1e-1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

ax = axes[0]
hazard_axes(ax, im, H_im, "fits through chosen points")
k0, k1 = hazard.fit_first_order_points(im, H_im, mafe=mafe_anchors[[0, 2]])
ax.loglog(fine, hazard.first_order_approx(fine, k0, k1), "--", color="C0",
          label=f"1st order, 2 points: k1={k1:.3g}")
k0, k1, k2 = hazard.fit_second_order_points(im, H_im, mafe=mafe_anchors)
ax.loglog(fine, hazard.second_order_approx(fine, k0, k1, k2), "-", color="C3",
          label=f"2nd order, 3 points: k1={k1:.3g}, k2={k2:.3g}")
# Mark the anchors on the numerical curve (interpolated in log-log space).
iml_anchors = np.exp(np.interp(np.log(mafe_anchors), np.log(H_im)[::-1],
                               np.log(im)[::-1]))
ax.plot(iml_anchors, mafe_anchors, "*", color="gold", ms=14, mec="k", mew=0.5,
        zorder=10, label="10 %, 5 %, 2 % in 50 yr")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
hazard_axes(ax, im, H_im, "regression over the 10 to 10 000 yr band")
k0, k1, r2 = hazard.fit_first_order_regression(im, H_im, mafe_band=band)
ax.loglog(fine, hazard.first_order_approx(fine, k0, k1), "--", color="C0",
          label=f"1st order: k1={k1:.3g}, r2={r2:.4f}")
k0, k1, k2, r2 = hazard.fit_second_order_regression(im, H_im, mafe_band=band)
ax.loglog(fine, hazard.second_order_approx(fine, k0, k1, k2), "-", color="C3",
          label=f"2nd order: k1={k1:.3g}, k2={k2:.3g}, r2={r2:.4f}")
ax.axhspan(*band, color="gold", alpha=0.15, label="fitting band")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
```

The power law can follow the curve only over a short stretch, so both
first-order fits drift away from it on either side. The second-order fits
follow its curvature over a much wider range.

## Where the points go

A first-order fit through two points is only as good as the choice of points.
The same curve, fitted through three different pairs of return periods, gives
quite different exponents $k_1$:

```{code-cell} python
pairs = [(75, 475), (475, 2475), (2475, 10000)]

fig, ax = plt.subplots(figsize=(6.5, 5))
hazard_axes(ax, im, H_im, "first-order fits through different pairs")
for colour, (rtp_lo, rtp_hi) in zip(("C0", "C1", "C2"), pairs):
    mafe = 1 / np.array([rtp_lo, rtp_hi])
    k0, k1 = hazard.fit_first_order_points(im, H_im, mafe=mafe)
    ax.loglog(fine, hazard.first_order_approx(fine, k0, k1), "--", color=colour,
              label=f"{rtp_lo} to {rtp_hi} yr: k1={k1:.3g}")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
```

The exponent grows as the pair moves to rarer frequencies, because the curve
steepens. Since the closed-form risk grows with $\exp(\tfrac{1}{2} k_1^2
\beta^2)$, the choice matters: fit around the intensities that dominate the risk
of the structure in question.

## How wide the band is

Widening the regression band covers more of the curve, at the cost of a poorer
fit inside it. The $r^2$ (in $\ln H$) shows how much the power law suffers,
compared with the second order:

```{code-cell} python
bands = {
    "250 to 2475 yr": (1 / 2475, 1 / 250),
    "100 to 5000 yr": (1 / 5000, 1 / 100),
    "10 to 10 000 yr": (1e-4, 1e-1),
    "whole curve": (H_im[-1], H_im[0]),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, order in zip(axes, (1, 2)):
    hazard_axes(ax, im, H_im, f"order {order} regression, by band")
    for colour, (label, band_i) in zip(("C0", "C1", "C2", "C3"), bands.items()):
        if order == 1:
            k0, k1, r2 = hazard.fit_first_order_regression(im, H_im,
                                                           mafe_band=band_i)
            fitted = hazard.first_order_approx(fine, k0, k1)
        else:
            k0, k1, k2, r2 = hazard.fit_second_order_regression(im, H_im,
                                                                mafe_band=band_i)
            fitted = hazard.second_order_approx(fine, k0, k1, k2)
        ax.loglog(fine, fitted, "-", color=colour, lw=1.4,
                  label=f"{label}: r2={r2:.4f}")
    ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
```

## Three sites

The three example sites have increasing curvature, from A to C. Below, the
second-order regression over the 10 to 10 000 year band on each, with the
intensity at which it turns over (see {doc}`../theory`) marked where it falls
inside the plot:

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, label in zip(axes, "ABC"):
    im_s, H_s = load_site(label)
    fine_s = np.geomspace(im_s[0], im_s[-1], 300)
    hazard_axes(ax, im_s, H_s, f"site {label}")
    k0, k1, r2 = hazard.fit_first_order_regression(im_s, H_s, mafe_band=band)
    ax.loglog(fine_s, hazard.first_order_approx(fine_s, k0, k1), "--",
              color="C0", label=f"1st: k1={k1:.2f}, r2={r2:.4f}")
    k0, k1, k2, r2 = hazard.fit_second_order_regression(im_s, H_s,
                                                        mafe_band=band)
    ax.loglog(fine_s, hazard.second_order_approx(fine_s, k0, k1, k2), "-",
              color="C3", label=f"2nd: k1={k1:.2f}, k2={k2:.2f}, r2={r2:.4f}")
    im_turn = np.exp(-k1 / (2 * k2))
    if im_turn > im_s[0]:
        ax.axvline(im_turn, color="C3", ls=":", lw=1,
                   label=f"2nd-order turnover, {im_turn:.3g} g")
    ax.axhspan(*band, color="gold", alpha=0.15)
    ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
```

Over the band, the second-order fit is close on all three sites. Outside it,
and on site C especially, it departs from the curve: below the turnover it
even rises with intensity. Use the fit only over the band it was fitted to.
