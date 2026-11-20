# hazrisk

**hazrisk** computes seismic risk from a hazard curve and a fragility curve. It
fits the first-order (power law) and second-order (log-quadratic)
approximations to a numerical hazard curve, evaluates the SAC/FEMA closed-form
solutions for the mean annual frequency of a limit state (MAFE, e.g. of
collapse) and the median capacity needed to reach a target, integrates a
fragility numerically against a tabulated hazard curve, and shows which
intensities the risk comes from. It is written for engineers and researchers
who work with probabilistic seismic hazard analysis (PSHA) results, for example
from OpenQuake, in their own Python scripts and notebooks. Every function works
on numbers and on numpy arrays, so a whole hazard curve or a grid of capacities
is handled in one call.

> **Alpha software.** The interface may still change between releases. If you
> use hazrisk for published results, pin the exact version
> (`pip install hazrisk==0.1.0`) and cite it (see below).

## Installation

```bash
pip install hazrisk
```

hazrisk needs Python 3.12 or later, numpy and scipy. For plotting your results
you can install matplotlib alongside it:

```bash
pip install "hazrisk[plot]"
```

## Quick start

```python
import numpy as np
from hazrisk import hazard, risk

# A tabulated hazard curve H(im) (e.g. from OpenQuake): im in g, mean annual
# frequency of exceedance per year. Trailing zeros, as OpenQuake writes them,
# can be left in.
im = np.array([0.00213, 0.0044, 0.00909, 0.0188, 0.0387, 0.08, 0.165, 0.341, 0.704, 1.45])
H_im = np.array([3.43e-2, 1.73e-2, 7.24e-3, 2.44e-3, 6.49e-4, 1.36e-4, 1.97e-5, 1.6e-6, 5.55e-8, 4.42e-10])

# Second-order fit over the 10 to 10 000 year band (Vamvatsikos 2013).
k0, k1, k2, r2 = hazard.fit_second_order_regression(im, H_im, mafe_band=(1e-4, 1e-1))

# Mean annual frequency of collapse for a median capacity of 0.3 g, beta = 0.4:
theta_c, beta = 0.3, 0.4
print(f"closed form: {risk.mafe_second_order(theta_c, beta, k0, k1, k2):.3e} /yr")
print(f"numerical:   {risk.numerical_mafe_pchip(theta_c, im, H_im, beta):.3e} /yr")
```

```text
closed form: 8.184e-06 /yr
numerical:   6.143e-06 /yr
```

The two answers differ because the closed form is only as good as the hazard
fit: here most of the risk comes from hazard levels rarer than the fitted band,
where the fit overestimates the hazard. The numerical integration uses the
tabulated curve itself.

## What is in the package

The package follows the chain hazard → fragility → risk, one module per step,
with a fourth module for converting between probability, frequency and return
period.

| Module | Main functions |
|---|---|
| `hazrisk.hazard` | `first_order_approx`, `second_order_approx` (and `*_dHdx`): analytical hazard curves; `fit_first_order_points`, `fit_second_order_points`, `fit_first_order_regression`, `fit_second_order_regression`: fit them to a numerical curve; `pchip_interpolant`, `pchip_approx`, `pchip_dHdx`: smooth interpolation of a tabulated curve |
| `hazrisk.fragility` | `lognormal_fragility_curve`: probability of the limit state given the intensity |
| `hazrisk.risk` | `mafe_first_order`, `mafe_second_order`: closed-form MAFE; `required_capacity_first_order`, `required_capacity_second_order`: median capacity for a target MAFE; `numerical_mafe`, `numerical_mafe_pchip`: numerical integration; `risk_contribution`, `risk_contribution_pchip`: where the risk comes from; `risk_ratio_first_order`, `theta_ratio_first_order`: ratios between two designs |
| `hazrisk.convert` | `poe_to_mafe`, `mafe_to_poe`, `poe_to_rtp`, `rtp_to_poe`, `mafe_to_rtp`, `rtp_to_mafe`: probability of exceedance in `t` years, mean annual frequency and return period |

## Documentation

- Documentation: <https://hazrisk.readthedocs.io>
- Source code and issues: <https://github.com/nc-hsu/hazrisk>
- Changelog: <https://github.com/nc-hsu/hazrisk/blob/main/CHANGELOG.md>

<!-- Zenodo DOI badge: added after the first release. -->

## How to cite

If you use hazrisk in your work, please cite the software. The citation details
are in [`CITATION.cff`](https://github.com/nc-hsu/hazrisk/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button reads it). Please also cite the methods
you used:

- Baker, J. W., Bradley, B. A. & Stafford, P. J. (2021). *Seismic Hazard and
  Risk Analysis*. Cambridge University Press.
  <https://doi.org/10.1017/9781108425056>
- Cornell, C. A., Jalayer, F., Hamburger, R. O. & Foutch, D. A. (2002).
  Probabilistic basis for 2000 SAC Federal Emergency Management Agency steel
  moment frame guidelines. *Journal of Structural Engineering*, 128(4),
  526–533. <https://doi.org/10.1061/(ASCE)0733-9445(2002)128:4(526)>
- Fritsch, F. N. & Butland, J. (1984). A method for constructing local monotone
  piecewise cubic interpolants. *SIAM Journal on Scientific and Statistical
  Computing*, 5(2), 300–304. <https://doi.org/10.1137/0905021>
- Moler, C. (2004). *Numerical Computing with MATLAB*. Society for Industrial
  and Applied Mathematics. <https://doi.org/10.1137/1.9780898717952>
- Vamvatsikos, D. (2013). Derivation of new SAC/FEMA performance evaluation
  solutions with second-order hazard approximation. *Earthquake Engineering &
  Structural Dynamics*, 42(8), 1171–1188. <https://doi.org/10.1002/eqe.2265>

## Licence

Licensed under GPL-3.0-or-later.

Copyright (C) 2026 Nicholas Clemett

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
