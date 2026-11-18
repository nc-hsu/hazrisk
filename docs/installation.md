# Installation

hazrisk needs Python 3.12 or later. Install it from PyPI:

```bash
pip install hazrisk
```

This also installs its two dependencies, numpy (2.0 or later) and scipy (1.13 or
later). hazrisk does not plot anything itself, but the examples in this
documentation use matplotlib; to install it alongside:

```bash
pip install "hazrisk[plot]"
```

To check the installation:

```bash
python -c "import hazrisk; print(hazrisk.__version__)"
```

## Pinning the version

hazrisk is alpha software, so the interface may change between releases. For
work you will publish, pin the version you used, e.g. in a `requirements.txt`:

```text
hazrisk==0.1.0
```

## Development version

To work on hazrisk itself, see
[CONTRIBUTING.md](https://github.com/nc-hsu/hazrisk/blob/main/CONTRIBUTING.md):
clone the repository and install it in editable mode with
`pip install -e ".[dev]"`.
