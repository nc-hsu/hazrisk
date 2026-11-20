# Changelog

All notable changes to hazrisk are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The first public release, 0.1.0. Earlier commits were never released; the
changes below matter to anyone who used them from git.

### Added

- Packaging metadata: dependencies (numpy >= 2.0, scipy >= 1.13), Python >= 3.12,
  licence, classifiers and project links, so `pip install` gives a working
  package.
- `hazrisk.__version__`; `import hazrisk` imports the `hazard`, `fragility`,
  `risk` and `convert` modules.
- `hazrisk.convert`, for probability of exceedance in `t` years, mean annual
  frequency and return period: `poe_to_mafe`, `mafe_to_poe`, `poe_to_rtp`,
  `rtp_to_poe`, `mafe_to_rtp` and `rtp_to_mafe`. They apply to any mean annual
  frequency, of the hazard or of a limit state.
- Hazard curves that end in a run of zeros, as OpenQuake writes them, are
  accepted by every function: the integration uses the zeros, the fitting
  functions drop them, and the PCHIP functions evaluate up to the last
  positive level.
- `hazard.pchip_interpolant`, the public PCHIP interpolant of a hazard curve in
  log-log space, which `pchip_approx`, `pchip_dHdx` and `numerical_mafe_pchip`
  are built on.
- Type hints throughout, checked with `mypy --strict`, and a `py.typed` marker.
- numpydoc docstrings for every public function, with formulas, references and
  examples that run as doctests.
- Example hazard curves for three anonymised sites in `examples/data/`.
- Example page *Trailing zeros: keep them or trim them?*: how the zeros at the
  end of a hazard curve change `numerical_mafe_pchip` and
  `risk_contribution_pchip`, and how to choose between keeping and trimming
  them.

### Changed

- The package uses a `src/` layout. If you installed it in editable mode, run
  `pip install -e .` again.
- Every public function accepts Python lists and integer arrays as well as
  float arrays.
- **Argument order of the closed forms:** the dispersion `beta` comes second,
  after the capacity or the target, followed by the hazard coefficients:
  `mafe_first_order(theta_c, beta, k0, k1)`,
  `required_capacity_first_order(target_mafe, beta, k0, k1)`,
  `mafe_second_order(theta_c, beta, k0, k1, k2)` and
  `required_capacity_second_order(target_mafe, beta, k0, k1, k2)`. Calls that
  pass the arguments by position in the order `k0, k1[, k2], beta` must be
  updated; calls by keyword are unaffected.
- `poe_to_mafe`, `mafe_to_poe` and `poe_to_rtp` are in `hazrisk.convert`, not
  `hazrisk.hazard`.
- The evaluation levels of `pchip_approx`, `pchip_dHdx` and
  `risk_contribution_pchip` are called `im_x` (was `iml`).
- `numerical_mafe_pchip` integrates a curve that ends in zeros down to the
  first zero level, instead of stopping at the last positive level and adding
  the tail term there. Results change for median capacities near or above the
  last positive level (by up to about 5 % on the example curves), and the
  normalised `risk_contribution_pchip` changes with them. To get the result
  without the zeros, trim them before the call; the example page *Trailing
  zeros* compares the two.
- `risk_ratio_first_order` and `theta_ratio_first_order` use ordinary numpy
  broadcasting: single numbers give a single number. They no longer build a
  4-D outer product from four 1-D vectors; for that grid, call
  `theta_ratio_first_order(*np.ix_(R_lambda, k1, beta_1, R_beta))`, which
  gives the same array as before.
- New `ValueError`s for input that used to give silently wrong results:
  - `numerical_mafe` and `risk_contribution` raise if `H_im` increases
    anywhere (levels in the wrong order gave a negative MAFE) or if the
    lengths of the arrays do not match;
  - `p_second_order`, `mafe_second_order` and
    `required_capacity_second_order` raise if `1 + 2*k2*beta**2 <= 0` (this
    gave `nan`);
  - `required_capacity_second_order` raises if the target MAFE is out of reach
    of the fitted curve.

### Fixed

- `risk_contribution` used a left Riemann sum, which undercounted the risk
  (on a 25-level curve its terms added up to about two thirds of
  `numerical_mafe`), and returned `None` for `normalised=False`. It now uses
  the trapezoid rule: its terms add up to `numerical_mafe`, and
  `normalised=False` returns them. Bar charts made with the old version will
  change.
- `risk_ratio_first_order` and `theta_ratio_first_order` failed on single
  numbers and on 2-D input; `risk_ratio_first_order` failed on integer input.
- `required_capacity_second_order` raised `ZeroDivisionError` at `k2 = 0` and
  lost accuracy for small `k2` (relative error 1e-4 at `k2 = 1e-12`). It now
  uses a form without division by `k2` and gives exactly
  `required_capacity_first_order` at `k2 = 0`.
- The analytical hazard functions failed on integer arrays.

### Removed

- `hazrisk.gmm` (the Boore, Joyner & Fumal 1997 ground motion model) and its
  coefficient file. It did not work when installed from a wheel. Use the
  implementation in OpenQuake.
- `hazrisk.imcorrelation` (the Baker & Jayaram 2008 correlation model).
  **It was wrong:** for period pairs with one period below 0.109 s and the
  other above, it returned correlations above 1 (e.g. 1.75 instead of 0.59 for
  0.05 s and 0.5 s). If you used `BJ08`, check your results. Use the
  implementation in OpenQuake.
- The empty `conditionalspectum` module and `recordselection` package.
- The private `hazard._pchip_log_log`; use `hazard.pchip_interpolant`.
