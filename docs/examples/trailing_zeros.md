---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Trailing zeros: keep them or trim them?

Some hazard software writes a hazard of exactly zero at any intensity level
where the hazard cannot be computed (e.g. due to truncation of the gmm).
{func}`~hazrisk.risk.numerical_mafe_pchip` and
{func}`~hazrisk.risk.risk_contribution_pchip` accept such curves as they are,
but you can also trim the zeros off before the call. The two give different
answers, and this page shows where the difference comes from and how to choose:

- **Z: with zeros.** The curve is passed as written with trailing zeros. The
  integral runs down to the first level where the hazard is zero.
- **T: without zeros.** The trailing zeros are trimmed off first. The curve
  then ends at the last positive level, and the hazard above it is handled by
  the tail term of {func}`~hazrisk.risk.numerical_mafe`.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import fragility, hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site."""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


im, H_im = load_site("B")
theta_c, beta = 2.0, 0.4
```

The curve of `site_B.csv` with a median capacity of 2.0 g is chosen because
the difference is large there. Most cases differ far less (see
{ref}`the last section <trailing-zeros-how-much>`).

## The two calculations

T trims the curve at the last positive level. `n` is the number of levels up
to and including it:

```{code-cell} python
# Index of the last positive level; everything after it is a trailing zero.
n = np.flatnonzero(H_im > 0)[-1] + 1

mafc_Z = risk.numerical_mafe_pchip(theta_c, im, H_im, beta)
mafc_T = risk.numerical_mafe_pchip(theta_c, im[:n], H_im[:n], beta)
print(f"Z, with zeros:    {mafc_Z:.4e} /yr")
print(f"T, without zeros: {mafc_T:.4e} /yr")
print(f"Z / T = {mafc_Z / mafc_T:.4f}")
```

## Where the difference comes from

Call the last positive level $im_N$, its hazard $H_N$, and the first zero
level $im_z$. The fragility at the two levels is $P_N$ and $P_z$. The risk is
the area under the fragility plotted against the hazard (see
{doc}`../theory`),

$$
\lambda_C = -\int P \, dH .
$$

Both calculations integrate the same PCHIP curve from the first level up to
$im_N$, so that part of the area is identical. They differ only in the last
piece of area, from $H = H_N$ down to $H = 0$. The hazard falls to zero
somewhere between $im_N$ and $im_z$, and the tabulated curve does not say
where.

- **T** has no level above $im_N$, so it holds the fragility at $P_N$ for all
  of the remaining hazard. The last piece is a rectangle:
  $H_N P_N$.
- **Z** has the point $(H = 0, P_z)$ and joins it to $(H_N, P_N)$ with a
  straight line, as the trapezoid rule does between any two levels. The last
  piece is a trapezoid: $\tfrac{1}{2}(P_N + P_z) H_N$.

The difference between them is the triangle

$$
\lambda_C^Z - \lambda_C^T = \tfrac{1}{2} (P_z - P_N) H_N .
$$

```{code-cell} python
im_N, H_N, im_z = im[n - 1], H_im[n - 1], im[n]
P_N, P_z = fragility.lognormal_fragility_curve(np.array([im_N, im_z]), theta_c, beta)
last_T = H_N * P_N  # rectangle: fragility held at P_N
last_Z = 0.5 * (P_N + P_z) * H_N  # trapezoid: straight line to (0, P_z)
common = mafc_T - last_T  # the area up to im_N, the same in Z and T

print(f"im_N = {im_N:.3g} g, H_N = {H_N:.3e} /yr, im_z = {im_z:.3g} g")
print(f"P_N = {P_N:.3f}, P_z = {P_z:.3f}")
print(f"common part up to im_N:   {common:.4e} /yr")
print(f"last piece, T (rectangle): {last_T:.4e} /yr")
print(f"last piece, Z (trapezoid): {last_Z:.4e} /yr")
print(f"triangle Z - T:            {last_Z - last_T:.4e} /yr"
      f" = {(last_Z - last_T) / mafc_T:.1%} of T")
```

The figure shows the four steps. Panel (c) is the one to look at.

```{code-cell} python
:tags: [hide-input]

C_Z, C_T, C_TRI = "#e8833a", "#2a6fdb", "#c0392b"

# PCHIP curve up to the last positive level, as both calculations use it.
im_x = np.geomspace(im[0], im_N, 2000)
H_x = hazard.pchip_approx(im_x, im, H_im)
P_x = fragility.lognormal_fragility_curve(im_x, theta_c, beta)
# The unnormalised contribution is the same in Z and T; only the total differs.
density = risk.risk_contribution_pchip(im_x, theta_c, im, H_im, beta,
                                       normalised=False)

fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

# (a) The hazard curve, with the zero levels drawn at the bottom of the axis
# because zero has no place on a log scale.
ax = axs[0, 0]
pos = H_im > 0
y_floor = H_im[pos].min() / 30
ax.loglog(im_x, H_x, color="0.3", lw=1.5, label="PCHIP of H(im)")
ax.loglog(im[pos], H_im[pos], "o", ms=5, color="0.3", label="tabulated H(im) > 0")
ax.plot(im[~pos], np.full((~pos).sum(), y_floor), "v", ms=9, color=C_TRI,
        label="tabulated H(im) = 0 (drawn at the axis)")
for x, text, ha in [(im_N, "$im_N$: last H > 0 ", "right"),
                    (im_z, " $im_z$: first H = 0", "left")]:
    ax.axvline(x, color="0.6", ls=":", lw=1)
    ax.text(x, H_im[0], text, rotation=90, va="top", ha=ha, fontsize=9)
ax.annotate("The hazard falls to zero somewhere\nbetween $im_N$ and $im_z$.\n"
            "Z: the curve ends at $im_z$.\nT: the curve ends at $im_N$.",
            xy=(im_N, H_N), xytext=(0.004, 2e-8), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set(ylim=(y_floor / 2, H_im[0] * 2), xlabel="AvgSA[0,3] [g]",
       ylabel="H(im) [1/yr]", title="(a) Hazard curve")
ax.legend(loc="lower left", fontsize=8)

# (b) The fragility, and how much it rises between im_N and im_z.
ax = axs[0, 1]
im_f = np.geomspace(0.1, 4, 400)
ax.semilogx(im_f, fragility.lognormal_fragility_curve(im_f, theta_c, beta),
            color="0.3", lw=1.5)
ax.plot(im_N, P_N, "o", color=C_T, ms=8, label=f"$P_N$ = P($im_N$) = {P_N:.3f}")
ax.plot(im_z, P_z, "s", color=C_Z, ms=8, label=f"$P_z$ = P($im_z$) = {P_z:.3f}")
ax.axvspan(im_N, im_z, color=C_TRI, alpha=0.08)
ax.axvline(theta_c, color="0.6", ls="--", lw=1)
ax.text(theta_c, 0.95, r" $\theta_c$", fontsize=10)
ax.text(im_N * 0.95, 0.45, "fragility rises\nfrom $P_N$ to $P_z$\n"
        "between $im_N$ and $im_z$\n(shaded)", ha="right", fontsize=9)
ax.set(ylim=(0, 1.02), xlabel="AvgSA[0,3] [g]", ylabel="P(C | im)",
       title="(b) Fragility")
ax.legend(loc="upper left", fontsize=9)

# (c) The risk as the area under P against H, zoomed to the smallest H, where
# the two calculations differ.
ax = axs[1, 0]
sel = H_x <= 6 * H_N
ax.fill_between(H_x[sel], P_x[sel], color="0.85", lw=0, zorder=0,
                label="up to $im_N$: the same in Z and T")
ax.plot(H_x[sel], P_x[sel], color="0.3", lw=1.5, label="P against H")
ax.fill_between([0, H_N], [P_N, P_N], color=C_T, alpha=0.35, lw=0,
                label=f"T and Z: rectangle $H_N P_N$ = {last_T:.3e}")
ax.fill_between([0, H_N], [P_z, P_N], [P_N, P_N], color=C_TRI, alpha=0.45, lw=0,
                label=r"Z only: triangle $\frac{1}{2}(P_z-P_N)H_N$ = "
                      f"{last_Z - last_T:.3e}")
ax.plot([0, H_N], [P_z, P_N], color=C_Z, lw=2, label="Z: straight line to (0, $P_z$)")
ax.plot([0, H_N], [P_N, P_N], color=C_T, lw=2, ls="--", label="T: P held at $P_N$")
ax.plot(H_N, P_N, "o", color=C_T, ms=8)
ax.plot(0, P_z, "s", color=C_Z, ms=8, clip_on=False)
ax.annotate(f"$H_N$ = {H_N:.2e}\n(at $im_N$)", xy=(H_N, P_N),
            xytext=(1.3 * H_N, P_N * 0.35), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.annotate("H = 0 at $im_z$", xy=(0, P_z), xytext=(1.5 * H_N, P_z * 1.05),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set(xlim=(0, 6 * H_N), ylim=(0, max(P_z, P_x[sel].max()) * 1.15),
       xlabel="H [1/yr], linear, zoomed to the rare end", ylabel="P(C | im)",
       title=r"(c) $\lambda_C = -\int P\,dH$ = area under P against H")
ax.legend(loc="upper right", fontsize=8)

# (d) The normalised contribution: the same curve divided by two totals.
ax = axs[1, 1]
ax.semilogx(im_x, density / mafc_Z, color=C_Z, lw=2, label=r"Z: divided by $\lambda_C^Z$")
ax.semilogx(im_x, density / mafc_T, color=C_T, lw=2, ls="--",
            label=r"T: divided by $\lambda_C^T$")
ax.axvline(im_N, color="0.6", ls=":", lw=1)
ax.axvspan(im_N, im_z, color=C_TRI, alpha=0.08)
ax.text(np.sqrt(im_N * im_z), 0.05 * (density / mafc_T).max(),
        "not drawn\n(above $im_N$)", ha="center", fontsize=8)
ax.text(0.03, 0.97,
        r"The same d$\lambda_C$/d ln im up to $im_N$ in Z and T." "\n"
        "Only the total it is divided by differs, so\n"
        f"Z is T x {mafc_T / mafc_Z:.3f}.\n\n"
        "Area under the curve (share up to $im_N$):\n"
        f"  Z: {1 - last_Z / mafc_Z:.3f}\n"
        f"  T: {1 - last_T / mafc_T:.3f}\n"
        "The rest lies between $im_N$ and $im_z$.",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(facecolor="w", edgecolor="0.8"))
ax.set(xlim=(0.05, 3), xlabel="AvgSA[0,3] [g]",
       ylabel=r"$d\lambda_C / d \ln im$ (normalised)",
       title="(d) Risk contribution, normalised")
ax.legend(loc="center left", fontsize=8.5)

for a in axs.flat:
    a.grid(True, which="major", color="0.92", lw=0.6)
fig.suptitle(rf"site_B.csv, $\theta_c$ = {theta_c} g, $\beta$ = {beta}:  "
             rf"$\lambda_C^Z$ = {mafc_Z:.4e},  $\lambda_C^T$ = {mafc_T:.4e} /yr")
fig.tight_layout()
```

**(a) Hazard curve.** The curve is positive up to $im_N$ and zero at $im_z$
and above. All the tabulated curve says is that the hazard reaches zero
somewhere in between. Z ends the curve at $im_z$; T ends it at $im_N$.

**(b) Fragility.** Between $im_N$ and $im_z$ the fragility rises from $P_N$ to
$P_z$, because the median capacity lies in that gap. This is why the choice
matters here.

**(c) Where the difference comes from.** The grey area, up to $im_N$, is the
same in both. T adds the blue rectangle $H_N P_N$: every exceedance of $im_N$
is counted with the fragility at $im_N$. Z adds the blue rectangle and the red
triangle, because it runs a straight line from $(H_N, P_N)$ to $(0, P_z)$. The
triangle is the whole difference between Z and T.

**(d) Risk contribution.** {func}`~hazrisk.risk.risk_contribution_pchip`
is evaluated up to $im_N$ only, and up there it is the same in Z and T. The
normalised curve is that density divided by the total, so Z lies below T by the
ratio of the totals. Neither curve integrates to one: the rest of the risk is
the last piece above $im_N$, which is larger in Z.

## Which one to use

The last piece of area lies between two bounds. The fragility can only rise
between $im_N$ and $im_z$, and the remaining hazard $H_N$ is used up somewhere
in that gap, so

$$
H_N P_N \;\le\; \text{last piece} \;\le\; H_N P_z .
$$

T is the lower bound, so T can only underestimate the risk. Z takes the mean
of the two bounds. Neither is exact; the true value depends on where in the
gap the hazard falls to zero, which the tabulated curve does not tell.

- **Leave the zeros in (Z)** for a best estimate. It uses the information that
  the hazard is zero at $im_z$, and it does not bias the result low.
- **Trim the zeros (T)** when you want the lower bound, or a result that does
  not depend on where the next level of the intensity grid happens to lie: in
  Z, moving $im_z$ changes $P_z$ and so the result, while T does not use
  $im_z$ at all.
- **When Z and T differ by more than you can accept**, neither is reliable.
  Rerun the hazard with more intensity levels between $im_N$ and $im_z$, so
  that the gap, and with it the triangle, shrinks.

(trailing-zeros-how-much)=
## How much does it matter?

The triangle $\tfrac{1}{2}(P_z - P_N) H_N$ is only large when the fragility
changes a lot between $im_N$ and $im_z$, which happens when the median capacity
is close to the top of the hazard curve. For a median capacity well below
$im_N$, both $P_N$ and $P_z$ are close to one and the triangle vanishes. A
larger $\beta$ flattens the fragility across the gap and makes the triangle
smaller. The ratio Z / T on the three example curves:

```{code-cell} python
def ratio_Z_to_T(im, H_im, theta_c, beta):
    """Return lambda_C with zeros divided by lambda_C without zeros."""
    n = np.flatnonzero(H_im > 0)[-1] + 1
    return (risk.numerical_mafe_pchip(theta_c, im, H_im, beta)
            / risk.numerical_mafe_pchip(theta_c, im[:n], H_im[:n], beta))


print(f"{'site':>4} {'beta':>5} " + " ".join(f"{f'theta_c={t}':>12}" for t in (0.3, 1.0, 2.0)))
for label in "ABC":
    im_s, H_s = load_site(label)
    for b in (0.4, 0.6):
        row = [ratio_Z_to_T(im_s, H_s, t, b) for t in (0.3, 1.0, 2.0)]
        print(f"{label:>4} {b:>5} " + " ".join(f"{r:>12.5f}" for r in row))
```

To check your own curve, compare the two calls as above, or compute the two
bounds of the last piece directly:

```{code-cell} python
spread = (P_z - P_N) * H_N  # upper bound minus lower bound of the last piece
print(f"the two bounds of the last piece are {spread / mafc_T:.1%} of T apart")
```
