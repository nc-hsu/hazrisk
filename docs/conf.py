"""Sphinx configuration for the hazrisk documentation.

Build locally with ``sphinx-build -W -n -b html docs docs/_build/html``
(``-W`` turns warnings into errors, ``-n`` reports every unresolved
cross-reference). Read the Docs runs the same build (``.readthedocs.yaml``).
"""

from importlib.metadata import version as _version

# -- Project information ------------------------------------------------------

project = "hazrisk"
author = "Nicholas Clemett"
copyright = "2026, Nicholas Clemett"
# The version comes from the installed package metadata, so the docs always
# show the version that was built (the package must be installed first).
release = _version("hazrisk")
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_nb",  # Markdown pages and executed Markdown notebooks
    "sphinx.ext.autodoc",  # API pages from the docstrings
    "sphinx.ext.autosummary",  # one page per function, with summary tables
    "sphinx.ext.napoleon",  # numpydoc-style docstring sections
    "sphinx.ext.intersphinx",  # links to Python, numpy and scipy docs
    "sphinx.ext.viewcode",  # "[source]" links next to each function
    "sphinx_copybutton",  # copy button on code blocks
]

exclude_patterns = ["_build", "jupyter_execute", ".jupyter_cache"]

# -- MyST and MyST-NB ---------------------------------------------------------

# $...$ and $$...$$ maths, and LaTeX environments such as align.
myst_enable_extensions = ["dollarmath", "amsmath"]
# Give headings anchors so pages can link to sections.
myst_heading_anchors = 3

# Execute the example notebooks at build time and cache the results, so every
# figure in the docs is regenerated from the current code, and a failing cell
# fails the build.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
# Notebooks run with docs/examples/ as the working directory, from which the
# example data is reached as ../../examples/data/.
nb_execution_in_temp = False

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Show the short alias names in the signatures rather than expanding them.
autodoc_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "FloatOrArray": "hazrisk._typing.FloatOrArray",
    "FloatArray": "hazrisk._typing.FloatArray",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# The docstrings give numpydoc types ("array_like", "float or ndarray",
# "optional"). Napoleon splits them into words, drops "or" and "optional",
# and links each word; the aliases point the numpy words at the numpy docs.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array_like": ":term:`array_like <numpy:array_like>`",
    "ndarray": "numpy.ndarray",
}

# -- intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML ---------------------------------------------------------------------

html_theme = "furo"
html_title = f"hazrisk {release}"
html_static_path: list[str] = []
