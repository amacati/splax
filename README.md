<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_emerges_dark.gif">
    <img src="docs/assets/logo_emerges_light.gif" alt="splax" width="560">
  </picture>
</div>

--------------------------------------------------------------------------------

<div align="center">

# splax

**Differentiable 3D gaussian splatting for JAX, with rasterization kernels written in [NVIDIA Warp](https://github.com/NVIDIA/warp).**

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://github.com/amacati/splax/actions/workflows/ruff.yml/badge.svg)](https://github.com/amacati/splax/actions/workflows/ruff.yml)
[![Ty](https://github.com/amacati/splax/actions/workflows/ty.yml/badge.svg)](https://github.com/amacati/splax/actions/workflows/ty.yml)
[![Docs](https://github.com/amacati/splax/actions/workflows/docs.yml/badge.svg)](https://amacati.github.io/splax)

</div>

splax renders and trains 3D gaussian splats inside JAX. Batched rendering and training run through `jax.vmap`, `jax.grad`, and `jax.jit`, with no system CUDA toolchain needed.

## Examples

Render a `.ply` scene from one camera.

```python
import jax.numpy as jnp
import splax

means, scales, quats, colors, opacities = splax.load_ply("scene.ply")
img = splax.inference.render(
    means, scales, quats, colors, opacities,
    viewmat=viewmat, background=jnp.ones(3),
    img_shape=(H, W), f=(fx, fy), c=(W // 2, H // 2),
    glob_scale=1.0, clip_thresh=0.01,
)  # (H, W, 3)
```

Batch over a stack of camera poses with `jax.vmap`. One batched kernel launch, not a Python loop.

```python
import jax

frames = jax.vmap(lambda vm: splax.inference.render(
    means, scales, quats, colors, opacities,
    viewmat=vm, background=jnp.ones(3), img_shape=(H, W),
    f=(fx, fy), c=(W // 2, H // 2), glob_scale=1.0, clip_thresh=0.01,
))(viewmats)  # (B, H, W, 3)
```

Take gradients through the differentiable renderer with `jax.grad`. `splax.render` is `splax.training.render` and differentiates w.r.t. means, scales, quats, colors, opacities.

```python
import jax

def loss(means, scales, quats, colors, opacities):
    img, _ = splax.render(
        means, scales, quats, colors, opacities,
        viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W),
        f=(fx, fy), c=(W // 2, H // 2), glob_scale=1.0, clip_thresh=0.01,
    )
    return jnp.mean((img - target) ** 2)

grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(means, scales, quats, colors, opacities)
```

To edit ``.ply`` scenes, we recommend [superspl.at](https://superspl.at/editor).

## Documentation

Full documentation lives at [amacati.github.io/splax](https://amacati.github.io/splax): installation, a quickstart, a user guide for rendering, training, batching, and IO, and the API reference.

## Why

Gaussian splatting lives mostly in PyTorch and hand-written CUDA. splax puts it inside JAX so splat rendering composes with `jax.vmap`, `jax.grad`, and `jax.jit` and drops into research pipelines that already run on JAX, without leaving the ecosystem for the render step.

## Architecture

The renderer is not pure JAX because the core of splatting does not map to XLA primitives. Rasterization is tile-binned with a data-dependent sort of gaussian-tile intersections, per-pixel early termination once transmittance saturates, and a memory-frugal backward that recomputes the blend instead of storing per-pixel state. splax implements the projection, rasterization, and their backward passes as Warp kernels and wires them into JAX through FFI custom calls under `jax.custom_vjp`. The kernels are batch-native: `jax.vmap` maps to a single batched launch (camera id folded into the sort key) rather than a sequential per-sample loop.

## Relation to jaxsplat

splax started from [jaxsplat](https://github.com/yklcs/jaxsplat) as the reference and parity baseline. The rasterizer was rewritten from CUDA to Warp to drop the system toolchain, then extended with the feature and performance work below.

## Improvements

Ported from [gsplat](https://github.com/nerfstudio-project/gsplat) and the papers behind it, which inspired most of the performance work (credit per item):

- Native multi-camera batched rendering, one launch with the camera id folded into the sort key (gsplat)
- Opacity-aware tight tile intersection (StopThePop, Speedy-Splat, gsplat #927)
- Packed 32-bit sort keys with quantized depth
- Persistent sort and bin scratch across frames (gsplat caching allocator design)
- Cooperative shared-memory tile blending with block-vote early exit (3DGS, gsplat)
- Fixed-budget MCMC training with static shapes, relocation plus covariance noise (gsplat MCMCStrategy, Kheradmand et al. 2024)
- Opacity and scale regularizers (gsplat mcmc preset)
- Progressive resolution fine-tuning (coarse-to-fine, 3DGS)
- Per-parameter Adam learning-rate schedules (gsplat, 3DGS)
- L1 plus D-SSIM photometric loss (3DGS, gsplat, via dm-pix)
- Camera pose gradients via `render(..., diff_wrt=("viewmat",))` (gsplat projection backward)
- Batch-native backward passes, `jax.vmap(jax.grad(render))` runs as one batched launch (gsplat)
- Batched training steps with sqrt-batch learning-rate scaling (gsplat `batch_size` and `steps_scaler`)
- Anti-aliased opacity compensation (Mip-Splatting, gsplat), depth regularization from COLMAP points (gsplat `depth_loss`), per-image exposure correction (gsplat appearance optimization), all opt-in

## Installation

Requires an NVIDIA GPU and a CUDA-enabled JAX (`jax[cuda12]`, pulled in as a dependency).

```sh
uv pip install "git+https://github.com/amacati/splax"
```

Developer setup with [pixi](https://pixi.sh/), which installs splax editable with the dev tooling:

```sh
git clone https://github.com/amacati/splax.git
cd splax
pixi shell
```

## License

MIT (see [LICENSE](LICENSE)). gsplat-derived portions are under Apache-2.0 ([licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)).
