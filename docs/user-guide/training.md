# Training

`splax.render` composes the `jax.custom_vjp` projection and rasterization
primitives, so `jax.grad` and `jax.value_and_grad` flow through it with respect
to means, scales, quats, colors, and opacities. The viewmat, background, and
rigid transforms are constants by default. The call always returns an
`(image, depths)` pair. The depth slot is `None` unless `render_depth=True`.

```python
def loss(means, scales, quats, colors, opacities):
    img, _ = splax.render(
        means, scales, quats, colors, opacities,
        viewmat=viewmat, background=jnp.ones(3), img_shape=(H, W),
        f=(fx, fy),
    )
    return jnp.mean((img - target) ** 2)

grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(means, scales, quats, colors, opacities)
```

## Camera pose and object pose gradients

Gradient selection happens purely through `jax.grad` and its `argnums`. The
projection backward is a single `jax.custom_vjp` with `symbolic_zeros=True`, so it
reads which inputs are actually differentiated and launches only the kernels those
gradients need.

- Differentiating with respect to means, scales, quats (and colors, opacities through the rasterizer) runs the gaussian-grad kernels. The viewmat is treated as a constant.
- Differentiating with respect to the `viewmat` runs the camera-pose accumulator only. The gaussian projection chains and their atomics are skipped, so post-training pose optimization pays only for the camera gradient.
- Differentiating with respect to both runs the joint kernel. The gaussian gradients are bit-identical to the gaussian-only path.
- With [rigid transforms](rendering.md#dynamic-scene-composition) active, a transform-aware kernel runs instead for every gradient selection, applying the same transforms during the geometry recompute and additionally providing gradients with respect to the `(K, 4, 4)` transforms themselves, so object poses can be optimized directly.

Because `viewmat` is a keyword argument of `render`, take its gradient by closing
over it in the differentiated position, for example:

```python
def loss(viewmat):
    img, _ = splax.render(means, scales, quats, colors, opacities,
                          viewmat=viewmat, background=bg, **cam)
    return photometric(img, target)

pose_grad = jax.grad(loss)(viewmat)  # runs the camera-pose accumulator only
```

## Depth channel

`render_depth=True` fills the depth slot of the returned `(image, depths)` pair
with an alpha-blended expected-depth map `D = Σ wᵢ dᵢ`. The depth channel is
differentiable and routes a cotangent through the gaussian geometry and camera
pose. It uses a separate Warp kernel, so the plain render (`render_depth=False`,
whose depth slot is `None`) never pays for it. This feeds COLMAP sparse-point
depth regularization.

## Antialiased mode

`antialiased=True` enables the Mip-Splatting opacity compensation, the same factor
described under [Rendering](rendering.md#antialiased-mode). Its gradient chains
back to scales, quats, and means through the existing conic-to-covariance vjp with
no Warp-kernel change. Default `False` is byte-identical to the plain path.

## MCMC training utilities

`splax.mcmc` ports the fixed-budget MCMC strategy (Kheradmand et al. 2024) as
static-shape JAX ops, so a pipeline that needs fixed array shapes still gets
MCMC-style training without densification that grows `N`.

- `relocate` teleports dead low-opacity gaussians onto alive ones and corrects opacity and scale for the resulting multiplicity. It returns a reset mask marking rows whose optimizer moments to zero.
- `inject_noise` adds covariance- and opacity-weighted Gaussian noise to the means every step, so low-opacity gaussians random-walk to explore while high-opacity ones stay put.

## Trainer scripts

Two scripts under `scripts/` are reference training recipes.

- `scripts/train_lego.py` fits the synthetic NeRF lego scene. It uses per-parameter Adam schedules, relocation and noise, an L1 plus D-SSIM loss, opacity and scale regularizers, and progressive resolution fine-tuning.
- `scripts/train_colmap.py` fits any COLMAP sparse reconstruction. It reads the intrinsics and extrinsics directly, initializes gaussians from the sparse point cloud, normalizes the scene by a similarity transform, and reuses the same MCMC recipe. It also exposes opt-in depth regularization, per-image affine exposure correction, and batched training steps with sqrt-batch learning-rate scaling.
