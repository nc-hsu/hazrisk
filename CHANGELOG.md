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
- `hazrisk.__version__`; `import hazrisk` imports the `hazard`, `fragility` and
  `risk` modules.
- `hazard.pchip_interpolant`, the public PCHIP interpolant of a hazard curve in
  log-log space, which `pchip_approx`, `pchip_dHdx` and `numerical_mafe_pchip`
  are built on.
- Type hints throughout, checked with `mypy --strict`, and a `py.typed` marker.
- numpydoc docstrings for every public function, with formulas, references and
  examples that run as doctests.
- Example hazard curves for three anonymised sites in `examples/data/`.

### Changed

- The package uses a `src/` layout. If you installed it in editable mode, run
  `pip install -e .` again.
- Every public function accepts Python lists and integer arrays as well as
  float arrays.
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
