# Development Environment Setup

This project targets Python `3.14`.

The recommended workflow uses `uv` since it provides a more modern and streamlined experience for managing Python environments and dependencies.
However, we still keep `requirements.txt` and `requirements-dev.txt` for `pip` + `venv` compatibility.

## Recommended Setup

First, make sure `uv` can provide Python `3.14` locally:

```bash
uv python install 3.14
```

Then create the development environment and install both runtime and dev
dependencies:

```bash
uv sync --dev
```

`.venv` will be created automatically in the project root. Activate it with:

```bash
source .venv/bin/activate
```

## Traditional Setup

If you prefer the traditional workflow, create a virtual environment with
Python `3.14` and install from the exported requirements files:

```bash
# create a virtual environment with Python 3.14
python3.14 -m venv .venv
# activate the virtual environment
source .venv/bin/activate
# upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Use `requirements.txt` if you only need runtime dependencies, or
`requirements-dev.txt` if you want the full local development toolchain.

## Dependency Source Of Truth

The dependency source of truth is `pyproject.toml` plus `uv.lock`.

The `requirements*.txt` files are generated compatibility snapshots:

```bash
uv export --no-hashes -o requirements.txt
uv export --no-hashes --dev -o requirements-dev.txt
```

This is automated via git hooks, so you don't need to worry about it.
