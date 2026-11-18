# Contributing to hazrisk

Bug reports, questions and suggestions are welcome as
[GitHub issues](https://github.com/nc-hsu/hazrisk/issues). For a change to the
code, please open an issue first so we can agree on the approach.

## Setting up

Clone the repository and install it in editable mode with the development
tools (tests, docs, lint, type checks, packaging), in a virtual environment:

```bash
git clone https://github.com/nc-hsu/hazrisk.git
cd hazrisk
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

These are the checks CI runs; please run them before opening a pull request.

```bash
python -m pytest           # tests and the examples in the docstrings
ruff check .               # lint
ruff format --check .      # formatting (run `ruff format .` to fix)
mypy                       # type check of src/ (strict)
mypy tests                 # type check of the tests
```

## Documentation

```bash
sphinx-build -W -n -b html docs docs/_build/html
```

then open `docs/_build/html/index.html`. The example pages are executed during
the build, so a failing example fails the build.

## Conventions

- Flat, directly importable functions (no classes), named after the quantity
  an engineer asks for, using the symbols of the literature.
- Functions work on numbers and numpy arrays alike (broadcasting); tabulated
  curves are ordered by increasing intensity measure.
- numpydoc docstrings with the formula, the source paper, and an example that
  runs as a doctest.
- New or changed functions come with tests; a forward/inverse pair gets an
  inverse-consistency test.
