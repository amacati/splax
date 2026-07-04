# Quickstart

This page renders a scene, batches over cameras, and takes a gradient. All three
share the same five gaussian arrays: `means` `(N, 3)`, `scales` `(N, 3)`, `quats`
`(N, 4)` unit wxyz, `colors` `(N, 3)` in `[0, 1]`, and `opacities` `(N, 1)` in
`[0, 1]`.

## Render a scene

`splax.load_ply` reads a 3DGS `.ply` into the five render-space arrays.
`splax.inference.render` is the pure, grad-free forward path.

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

`viewmat` is a `(4, 4)` world-to-camera matrix in the OpenCV convention (+z
forward). `f` is the focal length `(fx, fy)` and `c` is the principal point
`(cx, cy)`.

## Batch over cameras

`jax.vmap` maps a stack of view matrices to a single batched kernel launch, not a
Python loop.

```python
import jax

frames = jax.vmap(lambda vm: splax.inference.render(
    means, scales, quats, colors, opacities,
    viewmat=vm, background=jnp.ones(3), img_shape=(H, W),
    f=(fx, fy), c=(W // 2, H // 2), glob_scale=1.0, clip_thresh=0.01,
))(viewmats)  # (B, H, W, 3)
```

## Take a gradient

`splax.render` is the differentiable `splax.training.render`. It differentiates
w.r.t. means, scales, quats, colors, and opacities, and returns an
`(image, depths)` pair whose depth slot is `None` unless `render_depth=True`.

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

## Next steps

- [Rendering](../user-guide/rendering.md) covers camera conventions, backgrounds, and the antialiased flag.
- [Training](../user-guide/training.md) covers `diff_wrt`, the depth channel, and the trainer scripts.
- [Batching](../user-guide/batching.md) covers vmap semantics for inference and gradients.
