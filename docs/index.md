# hazrisk

**hazrisk** computes seismic risk from a hazard curve and a fragility curve. It
fits first-order (power law) and second-order (log-quadratic) approximations to
a numerical hazard curve, evaluates the SAC/FEMA closed-form solutions for the
mean annual frequency of a limit state (MAFE) and the median capacity needed to
reach a target, integrates a fragility numerically against a tabulated hazard
curve, and shows which intensities the risk comes from.

It is written for engineers and researchers who work with probabilistic seismic
hazard analysis (PSHA) results, for example from OpenQuake, in their own Python
scripts and notebooks. Every function is a plain function that works on numbers
and on numpy arrays.

```{admonition} Alpha software
:class: warning

The interface may still change between releases. If you use hazrisk for
published results, pin the exact version (`pip install hazrisk==0.1.0`) and
{doc}`cite it <citing>`.
```

Start with the {doc}`overview` to find the function you need, or the
{doc}`quickstart` for a first calculation.

```{toctree}
:maxdepth: 2
:caption: Getting started

overview
installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

examples/index
theory
conventions
data
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
references
```

```{toctree}
:maxdepth: 1
:caption: About

citing
changelog
licence
```
