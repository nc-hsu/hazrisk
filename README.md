# hazrisk
useful modules for hazard and risk analysis in earthquake engineering

## To do

- [ ] **Update docstrings** — bring all modules to a consistent numpydoc style: a short
  summary line, a citation to the source paper where one exists, and `Parameters` /
  `Returns` sections. `risk.py::numerical_mafe` is the standard to match. Currently
  `risk_contribution` uses `-- param:` bullets and `gmm.py` uses `#` comments.
- [ ] **Remove unnecessary modules** — ground motion models are handled by OpenQuake,
  and conditional spectrum / record selection by `pickagm`. Drop `conditionalspectum.py`
  and `recordselection/` (both empty stubs), and retire `gmm.py` and `imcorrelation.py`
  once any remaining callers are migrated.
- [ ] **Declare dependencies** — `pyproject.toml` lists none, despite the code importing
  numpy, pandas and scipy.
- [ ] **Add a test suite** — pytest, covering the forward/inverse consistency of the
  `mafe_*` and `required_capacity_*` pairs, and benchmarks against published values.
