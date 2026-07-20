# Rendering

`splax.render` is the rendering entry point. The call returns an `(image, depths)` pair
whose depth slot is `None` unless `render_depth=True`; gradients are covered under [Training](training.md).

```python
img, _ = splax.render(
    means, scales, quats, colors, opacities,
    viewmat=viewmat, background=jnp.ones(3),
    img_shape=(H, W), f=(fx, fy),
)  # (H, W, 3)
```

## Inputs

| Argument | Shape | Meaning |
|---|---|---|
| `means` | `(N, 3)` | World positions |
| `scales` | `(N, 3)` | Positive per-axis scales |
| `quats` | `(N, 4)` | Unit wxyz quaternions |
| `colors` | `(N, 3)` | RGB in `[0, 1]` |
| `opacities` | `(N, 1)` | Opacity in `[0, 1]` |

## Camera conventions

`viewmat` is a `(4, 4)` world-to-camera matrix in the OpenCV convention (+z
forward, +y down, +x right). This is what COLMAP stores directly. NeRF and
OpenGL poses (-z forward) must be converted first, as `scripts/train_lego.py`
does by multiplying the camera-to-world matrix by `diag(1, -1, -1, 1)` before
inverting.

`f` is the focal length `(fx, fy)` in pixels and `c` is the principal point
`(cx, cy)` in pixels, where the optical axis meets the image plane. It defaults
to the image center `(W / 2, H / 2)`, which is exact for synthetic cameras.
Calibrated real cameras (COLMAP intrinsics) provide their own off-center values.
`img_shape` is `(H, W)`. `glob_scale` multiplies every gaussian scale, and
`clip_thresh` is the near-plane depth cutoff.

## Backgrounds

`background` is a length-3 RGB color composited behind the splat where
transmittance remains. It is a constant and is not differentiated.

## Antialiased mode

`antialiased=True` applies the Mip-Splatting opacity compensation. A per-gaussian
factor from `splax.opacity_compensation` is multiplied into the blend opacity,
cancelling the area inflation that thin gaussians gain from the projection's
screen-space dilation. The tile intersection still counts with the raw opacity.
Default `False` is byte-identical to the plain path. Use the same setting at
inference that a model was trained with.

## Dynamic scene composition

Composed scenes can move whole sections of gaussians with rigid transforms, for
example a drone splat concatenated onto a room splat. `gaussian_transforms` is a
`(K, 4, 4)` stack of world-space transforms and `gaussian_slices` the `K`
matching non-overlapping `(start, stop)` index ranges. The gaussians in slice `k`
move by `gaussian_transforms[k]` and everything outside the slices stays static.
The transform is applied on the fly inside the projection kernel, so the splat
is never copied.

```python
img, _ = splax.render(
    means, scales, quats, colors, opacities,
    viewmat=viewmat, background=jnp.ones(3),
    img_shape=(H, W), f=(fx, fy),
    gaussian_transforms=poses,             # (K, 4, 4)
    gaussian_slices=((100, 1000), (1000, 1500)),
)
```

Batched dynamics work through `jax.vmap` over the transform stack. Every batch
element renders the same shared splat with its objects at different poses, and
one launch covers the whole batch.

```python
render_at = lambda poses: splax.render(
    means, scales, quats, colors, opacities,
    viewmat=viewmat, background=jnp.ones(3),
    img_shape=(H, W), f=(fx, fy),
    gaussian_transforms=poses, gaussian_slices=slices,
)[0]
imgs = jax.vmap(render_at)(pose_batch)  # (B, K, 4, 4) -> (B, H, W, 3)
```

Omitting the arguments is the plain path with identical output and performance.
The slices are static Python values, so changing them retraces a jitted render.
The transforms are differentiable, see
[object pose gradients](training.md#camera-pose-and-object-pose-gradients).

## Low-level primitives

`splax.render` composes two `jax.custom_vjp` primitives that are also
public.

- `splax.project` maps gaussians to screen-space `(xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit)`.
- `splax.rasterize` blends the projected gaussians into the `(H, W, 3)` image.

The Warp backend caches grow-only sort and bin scratch across renders.
`splax.clear_scratch` releases it, for example before switching to a very
different workload size.
