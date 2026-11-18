"""hazrisk: seismic hazard-curve approximation and risk integration.

The package follows the modelling chain from a site's seismic hazard to the
annual frequency of a structural limit state (for example collapse):

1. :mod:`hazrisk.hazard` -- the hazard curve, i.e. the mean annual frequency
   of exceeding each level of an intensity measure (``im``). It fits the
   first-order (power law) and second-order (log-quadratic) approximations,
   interpolates tabulated curves, and converts between probability of
   exceedance, annual frequency and return period.
2. :mod:`hazrisk.fragility` -- the fragility curve, i.e. the probability of
   reaching the limit state given ``im`` (lognormal).
3. :mod:`hazrisk.risk` -- the risk: the mean annual frequency of the limit
   state, from the SAC/FEMA closed-form solutions or by numerical
   integration of the fragility against the hazard curve, plus the inverse
   problem (required capacity for a target frequency) and where the risk
   comes from (risk contribution).

The three modules are the public interface::

    from hazrisk import hazard, fragility, risk
"""

# importlib.metadata reads the version of the installed distribution, so the
# version is written only once, in pyproject.toml.
from importlib.metadata import version

# Importing the modules here lets ``import hazrisk`` followed by
# ``hazrisk.risk...`` work without a separate import of each module.
from hazrisk import fragility, hazard, risk

__version__ = version("hazrisk")

__all__ = ["hazard", "fragility", "risk", "__version__"]
