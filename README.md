# ofdtd

Octree-based finite-difference time-domain tools in JAX.

## Development install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Run tests

```bash
pytest
```

## Run linting and type checks

```bash
ruff check .
mypy
```

## Build package artifacts

```bash
python -m build
```

## Build docs locally

```bash
mkdocs serve
```

The API reference pages are generated automatically from public functions,
classes, and docstrings under `src/ofdtd/`.
