# Example hazard curves

Three mean hazard curves used by the examples in the hazrisk documentation.
They are real PSHA results, chosen to show low (site A), medium (site B) and
high (site C) curvature in log-log space, i.e. low to high `k2` of the
second-order hazard approximation.

| File | Site |
|---|---|
| `site_A.csv` | low curvature |
| `site_B.csv` | medium curvature |
| `site_C.csv` | high curvature |

## Contents

- Intensity measure: average spectral acceleration AvgSA[0,3].
- Statistic: the mean hazard curve.
- Computed with OpenQuake on the 2020 European Seismic Hazard Model (ESHM20).
- The sites are anonymised: no names, coordinates or site identifiers are
  given.

Each file has a header line and two columns:

| Column | Meaning | Unit |
|---|---|---|
| `im` | intensity measure level, AvgSA[0,3] | g |
| `mafe` | mean annual frequency of exceeding `im` | 1/year |

The IM, AvgSA[0,3], is the average spectral acceleration over 10 linearly
spaced periods on the inclusive interval 0 to 3 seconds (so the PGA is
included). The investigation time is 1 year. The curves are given exactly as
OpenQuake wrote them, including the zeros at the highest levels, where the
probability of exceedance is below the precision OpenQuake stores. hazrisk
accepts the curves with these zeros; see the Conventions page of the
documentation for how each function uses them.

## Licence and source

These data are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>), separately from the hazrisk
code, which is licensed under GPL-3.0-or-later.

They were computed from the ESHM20 source and ground motion models. If you use
them, please cite ESHM20:

Danciu, L., Nandan, S., Reyes, C., Basili, R., Weatherill, G., Beauval, C.,
Rovida, A., Vilanova, S., Sesetyan, K., Bard, P.-Y., Cotton, F., Wiemer, S. &
Giardini, D. (2021). The 2020 update of the European Seismic Hazard Model:
Model Overview. EFEHR Technical Report 001, v1.0.0.
<https://doi.org/10.12686/a15>
