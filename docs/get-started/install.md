# Installation

## Requirements

splax needs an NVIDIA GPU and a CUDA-enabled JAX. The `jax[cuda12]` wheel is pulled in as a dependency, so no system CUDA toolchain is required. Python 3.12 is required.

## Install with pip

```bash
uv pip install "git+https://github.com/amacati/splax"
```

Verify the import:

```bash
python -c "import splax; print(splax.__all__)"
```

## Developer setup

[pixi](https://pixi.sh/) installs splax editable with the developer tooling into a managed environment.

=== "pixi"

    ```bash
    git clone https://github.com/amacati/splax.git
    cd splax
    pixi shell
    ```

=== "pixi + docs"

    ```bash
    git clone https://github.com/amacati/splax.git
    cd splax
    pixi shell -e docs
    ```

The dependency groups in `pyproject.toml` (`dev`, `docs`) mirror the pixi features, so [uv](https://docs.astral.sh/uv/) works as an alternative:

```bash
uv sync --group dev     # tests and ruff
uv sync --group docs    # documentation tooling
```

## Building the documentation

The docs are built with [ProperDocs](https://properdocs.org/) and the Material theme. Set `JAX_PLATFORMS=cpu` so the API reference imports splax without a GPU:

```bash
JAX_PLATFORMS=cpu pixi run -e docs docs-build
```
