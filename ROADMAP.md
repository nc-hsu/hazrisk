# hazrisk roadmap

Possible future enhancements. Nothing here is scheduled; it is a list of
ideas worth doing, kept with the code so it can be updated alongside it.
When an item is done, remove it here and describe it in `CHANGELOG.md`.

## Documentation

- **Theory in the docstrings.** Give each function a numpydoc "Theory"
  section, placed after References and before Examples, holding the
  derivation behind it, and retire `docs/theory.md`. Short defining equations
  stay at the top of the docstring (e.g. `second_order_dHdx`); long
  derivations (e.g. `required_capacity_second_order`) go in the Theory
  section, with links between functions whose theory is shared.
  - To decide first: where the background that belongs to no single function
    goes (the Poisson assumption, the risk integral and its tail term, why
    PCHIP is used, fitting through points vs by regression, what r² measures,
    the second-order turnover). Options: the closest function, the module
    docstrings, or a slimmed `theory.md`.
  - The docs build needs `napoleon_custom_sections` in `docs/conf.py` for a
    "Theory" section to render.
  - The long explanation in `risk.numerical_mafe` currently sits in its Notes
    section and would move to its Theory section.

## Functionality

- **Deaggregation of risk** by magnitude, distance and ε, building on
  `risk_contribution`.
- **Loss and vulnerability functions**, towards expected annual loss.
- **Second-order risk and capacity ratios**, counterparts of
  `risk_ratio_first_order` and `theta_ratio_first_order` on the
  second-order hazard.
- **Plotting helpers** (`hazrisk.plot`, behind the `plot` extra) for hazard
  curves, fits and risk contributions.

## Project

- A **JOSS paper**, reusing the README, the documentation and the Zenodo DOI.
