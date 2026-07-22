# Batching

The Warp kernels are batch-native. `jax.vmap` maps to a single batched launch,
with the camera id folded into the sort key, rather than a sequential per-sample
Python loop.

## Batched inference

Wrap `splax.render` in `jax.vmap` over any batched argument. Mapping
over a stack of view matrices renders one image per camera.

```python
frames = jax.vmap(lambda vm: splax.render(
    means, scales, quats, colors, opacities,
    viewmat=vm, background=jnp.ones(3), img_shape=(H, W),
    f=(fx, fy),
)[0])(viewmats)  # (B, H, W, 3)
```

Both underlying FFIs carry `vmap_method="expand_dims"`, so the batch axis is
handled inside one launch.

## Batched gradients

`jax.vmap(jax.grad(render))` over `splax.render` runs a single batched
backward launch for every gradient selection, matching per-sample sequential
gradients. The reduction depends on how an input is batched.

- Broadcast inputs, shared across the batch, get their gradients summed over the batch axis.
- Per-image inputs, for example a batch of camera poses differentiated with `jax.grad(loss, argnums=viewmat)`, get per-image gradients.

## Memory trade at large batch

A batched launch renders all `B` cameras together, so the sort and blend scratch
scale with the batch size. At large `B` this raises the peak memory footprint
relative to looping one camera at a time. `splax.clear_cache` releases the
cached scratch buffers when switching between very different batch sizes.
