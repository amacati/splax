# splax

<div align="center">
  <img src="assets/logo.svg" alt="splax" width="360"/>
</div>

**Differentiable 3D gaussian splatting for JAX, with rasterization kernels written in [NVIDIA Warp](https://github.com/NVIDIA/warp).**

splax renders and trains 3D gaussian splats inside JAX. Projection, rasterization, and their backward passes run as Warp kernels wired into JAX through FFI custom calls under `jax.custom_vjp`, so rendering composes with `jax.vmap`, `jax.grad`, and `jax.jit`. No system CUDA toolchain is required.

## Hero example

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

## Two render entry points

splax exposes two renderers that share every Warp kernel and differ only in their JAX-level wrapping.

- [`splax.inference.render`](user-guide/rendering.md) is the pure, grad-free forward path. Use it to serve a baked scene.
- [`splax.training.render`](user-guide/training.md), aliased as `splax.render`, is the differentiable path. Use it with `jax.grad` to fit gaussians.

## Where to go next

- [Installation](get-started/install.md) covers the pip install, GPU requirements, and the pixi developer setup.
- [Quickstart](get-started/quickstart.md) walks through rendering a scene, batching with `jax.vmap`, and taking a gradient.
- [User Guide](user-guide/rendering.md) documents rendering, training, batching, and PLY IO.
- [API Reference](api/index.md) is generated from the source.
