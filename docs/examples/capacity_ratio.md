---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Capacity ratio for a target risk ratio

How much stronger does a structure have to be to reduce its risk by a given
factor? On a first-order hazard curve the answer depends only on the slope
$k_1$ and the dispersions, not on $k_0$ or the starting capacity (see
{doc}`../theory`):

$$
R_\theta = \exp\left(\tfrac{1}{2} k_1 \beta_1^2 (R_\beta^2 - 1)
                     - \frac{\ln R_\lambda}{k_1}\right),
$$

with $R_\lambda = \lambda_2/\lambda_1$ the risk ratio, $R_\theta =
\theta_2/\theta_1$ the ratio of median capacities, and $R_\beta =
\beta_2/\beta_1$ the ratio of dispersions.
{func}`~hazrisk.risk.theta_ratio_first_order` evaluates it with numpy
broadcasting, so {func}`numpy.ix_` gives the whole grid of combinations in one
call.

```{code-cell} python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hazrisk import hazard, risk

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["savefig.facecolor"] = "w"

DATA = Path("..") / ".." / "examples" / "data"


def load_site(label):
    """Return (im, H_im) of an example site, without the trailing zeros."""
    data = np.loadtxt(DATA / f"site_{label}.csv", delimiter=",", skiprows=1)
    keep = data[:, 1] > 0
    return data[keep, 0], data[keep, 1]


# k1 of the first-order regression over the 10 to 10 000 yr band, per site.
k1_sites = {}
for label in "ABC":
    im, H_im = load_site(label)
    _, k1_sites[label], _ = hazard.fit_first_order_regression(
        im, H_im, mafe_band=(1e-4, 1e-1))

print("k1 values for sites:")
print({label: round(k1, 2) for label, k1 in k1_sites.items()})
```

## The grid

Risk ratios from 1 (no change) down to 0.05 (a twentieth of the risk), the
three site slopes, three dispersions, and two cases for the dispersion of the
new design: the same ($R_\beta = 1$) and 50 % smaller ($R_\beta = 0.5$):

```{code-cell} python
R_lambda = np.geomspace(1, 0.05, 50)
k1 = np.array(list(k1_sites.values()))
beta_1 = np.array([0.3, 0.4, 0.5])
R_beta = np.array([1.0, 0.5])

# One axis per input: the result has shape (50, 3, 3, 2).
R_theta = risk.theta_ratio_first_order(*np.ix_(R_lambda, k1, beta_1, R_beta))
print(R_theta.shape)
```

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
styles = {0.3: ":", 0.4: "-", 0.5: "--"}
for ax, j_beta in zip(axes, range(len(R_beta))):
    for i_k, (label, k1_i) in enumerate(k1_sites.items()):
        for i_b, b in enumerate(beta_1):
            ax.semilogx(R_lambda, R_theta[:, i_k, i_b, j_beta], styles[b],
                        color=f"C{i_k}",
                        label=rf"site {label} ($k_1$={k1_i:.2f}), $\beta_1$={b}")
    ax.set(xlabel=r"risk ratio $R_\lambda = \lambda_2 / \lambda_1$",
           ylabel=r"capacity ratio $R_\theta = \theta_2 / \theta_1$",
           title=rf"$R_\beta$ = {R_beta[j_beta]}")
    ax.invert_xaxis()
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
axes[0].legend(fontsize=7, ncol=1)
fig.tight_layout()
```

The flatter the hazard curve (the smaller $k_1$, here site B over this band),
the more capacity a given reduction in risk costs: halving the risk takes a
factor $2^{1/k_1}$ on the capacity when the dispersion stays the same. With equal dispersions the result
does not depend on $\beta_1$, so the three line styles coincide on the left. A
smaller dispersion for the new design (right) reduces the risk by itself, so
less extra capacity is needed, and at a risk ratio of one the capacity can
even be lowered.

The single-number case needs no grid:

```{code-cell} python
print(f"{risk.theta_ratio_first_order(0.5, 3.0, 0.4):.4f}")  # = 2 ** (1 / 3)
```
