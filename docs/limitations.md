# Limitations

The numerical integration uses the tabulated hazard curve, and the closed forms
use a fitted one. Both rest on assumptions worth checking for each calculation.

- **Below the first level the fragility is neglected.** The integration starts
  at `im[0]`. If the fragility is not close to zero there, the risk is
  underestimated. Make sure the hazard curve starts well below the median
  capacity, a few dispersions below it.
- **Above the last level the fragility is held constant.** The tail term counts
  every ground motion above the last level (with a positive hazard) with the
  fragility at that level. If the fragility is still rising there, the risk is
  slightly underestimated; for a curve that extends well above the median
  capacity the tail term is small.
- **The closed forms assume a lognormal fragility and the fitted hazard
  shape.** They are exact only if the hazard follows the power law (first
  order) or the log-quadratic (second order) at every intensity that matters
  for the risk. A real curve does so only over a band, so the result depends
  on the fitting band. Compare with {func}`~hazrisk.risk.numerical_mafe_pchip`
  (see {doc}`examples/closed_form_vs_numerical`).
- **The second-order fit turns over.** Below the turnover intensity the fitted
  hazard increases with $im$ (see {doc}`theory`). hazrisk does not check for
  it.
- **The ratio functions are first order only.** They assume both designs see
  the same power-law hazard, with the same $k_1$.
- **One hazard curve at a time.** hazrisk takes one hazard curve per call
  (usually the mean). To look at the epistemic uncertainty in the hazard, call
  it for each curve or fractile separately.
