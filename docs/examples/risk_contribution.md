---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Where the risk comes from

The risk is an integral over all intensities. Which of them contribute most
depends on both the hazard and the fragility, and it tells you where on the
hazard curve the calculation needs to be accurate: which band a closed-form fit
should cover, and whether the curve extends far enough. hazrisk gives the
contribution in two forms (see {doc}`../theory`):

- {func}`~hazrisk.risk.risk_contribution`: one value per tabulated level, the
  terms of the trapezoid sum of {func}`~hazrisk.risk.numerical_mafe`;
- {func}`~hazrisk.risk.risk_contribution_pchip`: a curve, the risk per unit
  $\ln im$, on the PCHIP-interpolated hazard.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import fragility, hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site, as written by OpenQuake."""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


im, H_im = load_site("B")
theta_c, beta = 0.3, 0.4
P = fragility.lognormal_fragility_curve(im, theta_c, beta)
```

## On the tabulated levels

{func}`~hazrisk.risk.risk_contribution` splits the risk into one value per
interval between tabulated levels, plus a last value for everything above the
last level (the tail term; zero here, because the curve ends in OpenQuake's
zeros). Normalised, they add up to one, and the raw contributions add up to
the result of `numerical_mafe`:

```{code-cell} python
shares = risk.risk_contribution(im, H_im, P)
print(f"sum of the shares: {shares.sum():.6f}")
print(f"MAFC: {risk.numerical_mafe(H_im, P):.4e} /yr "
      f"= sum of the raw contributions: "
      f"{risk.risk_contribution(im, H_im, P, normalised=False).sum():.4e} /yr")
```

## As a curve, and compared with the bars

To put the bars and the curve on one plot, each bar is divided by the width of
its interval in $\ln im$, so that it too is a contribution per unit
$\ln im$; then the area of each bar is its share. For comparison, the dashed
line is the same integrand with straight lines between the tabulated levels in
log-log space instead of PCHIP: its hazard slope is constant within each
interval, so it steps at every tabulated level.

```{code-cell} python
iml = np.geomspace(im[0], im[H_im > 0][-1], 2000)
curve = risk.risk_contribution_pchip(iml, theta_c, im, H_im, beta)

# Bars: the shares of the intervals (not the tail), per unit ln(im).
width_ln = np.diff(np.log(im))
density = shares[:-1] / width_ln

# Straight lines in log-log: dH/dim = s_i * H / im within interval i, with
# s_i the log-log slope of the interval, so the integrand per unit ln(im)
# is P * |s_i| * H. Normalised by the same total as the PCHIP curve.
pos = H_im > 0
im_p, H_p = im[pos], H_im[pos]
slopes = np.diff(np.log(H_p)) / np.diff(np.log(im_p))
interval = np.clip(np.searchsorted(im_p, iml, side="right") - 1, 0, len(slopes) - 1)
H_lin = np.exp(np.interp(np.log(iml), np.log(im_p), np.log(H_p)))
P_fine = fragility.lognormal_fragility_curve(iml, theta_c, beta)
straight = (P_fine * np.abs(slopes[interval]) * H_lin
            / risk.numerical_mafe_pchip(theta_c, im, H_im, beta))

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(im[:-1], density, width=np.diff(im), align="edge", color="C0",
       alpha=0.35, edgecolor="C0", label="risk_contribution (per interval)")
ax.semilogx(iml, curve, "-", color="C3", lw=2, label="risk_contribution_pchip")
ax.semilogx(iml, straight, "--", color="0.3", lw=1,
            label="straight lines in log-log")
ax.axvline(theta_c, color="k", ls=":", lw=1, label=rf"$\theta_c$ = {theta_c} g")
ax.set(xscale="log", xlim=(0.02, 3), xlabel="AvgSA(0.3 s) [g]",
       ylabel=r"$d\lambda_C / d \ln im$ (normalised)",
       title=rf"site B, $\theta_c$ = {theta_c} g, $\beta$ = {beta}")
ax.grid(True, which="both", lw=0.4, alpha=0.4)
ax.legend(fontsize=8)
fig.tight_layout()
```

The risk peaks below the median capacity: the fragility is still small there,
but those ground motions are much more frequent. The PCHIP curve and the bars
agree in shape; the straight-line version steps at every tabulated level.

## The band that matters

The cumulative contribution gives the intensities, and so the hazard
frequencies, between which a given share of the risk lies, which is the band a
closed-form fit should cover:

```{code-cell} python
cumulative = np.cumsum(shares)
lo = im[np.searchsorted(cumulative, 0.05)]
hi = im[np.searchsorted(cumulative, 0.95) + 1]
H_lo, H_hi = hazard.pchip_approx([lo, hi], im, H_im)
print(f"90 % of the risk lies between about {lo:.2g} g and {hi:.2g} g,")
print(f"i.e. hazard return periods of about {1 / H_lo:.2g} to {1 / H_hi:.2g} years.")
```

Here that is far rarer than the 10 to 10 000 year band fitted in the
{doc}`../quickstart`, which is why the closed form there overestimates the
risk: to use a closed form for this structure, fit the hazard over this band
instead (if the PSHA curve extends that far).
