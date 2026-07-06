"""Grad-free rendering entry point.

splax.inference.render shares every Warp kernel with splax.training.render and
differs only in the JAX wrapping. It calls the projection and rasterization FFI
primals directly, so there is no custom_vjp on the trace and jax.grad through it
raises. The blend residuals final_Ts and final_idx are discarded, and because
nothing downstream reads them XLA dead-code-eliminates them from the compiled
program. The training forward must keep them alive as backward residuals.

This path also supports rigid transforms tied to gaussians for composed dynamic
scenes. Non-overlapping slices of the gaussians each follow their own 4x4
transform, applied on the fly inside the projection kernel, so the base splat is
never duplicated. Under jax.vmap over the transform stack every batch element
moves the same gaussians differently while the splat itself stays broadcast.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from splax._project import _project_call, opacity_compensation
from splax._rasterize import _rasterize_call


def _transform_ids(n: int, slices: Sequence[tuple[int, int]]) -> jax.Array:
    """Map each gaussian to the transform it follows, -1 means static.

    Slices must be non-overlapping and inside [0, n). Violations raise
    immediately because a bad slice map silently corrupts the render. The checks
    are pure Python on the static slice values, so they work under jit.
    """
    for k, (start, stop) in enumerate(slices):
        if not (0 <= start < stop <= n):
            raise ValueError(f"gaussian slice {k} = [{start}, {stop}) outside [0, {n})")
    ordered = sorted(slices)
    for (_, prev_stop), (start, _) in zip(ordered, ordered[1:]):
        if start < prev_stop:
            raise ValueError(f"gaussian slices overlap: {list(slices)}")
    ids = jnp.full((n,), -1, jnp.int32)
    for k, (start, stop) in enumerate(slices):
        ids = ids.at[start:stop].set(k)
    return ids


def render(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    *,
    viewmat: jax.Array,
    background: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    antialiased: bool = False,
    gaussian_transforms: jax.Array | None = None,
    gaussian_slices: Sequence[tuple[int, int]] | None = None,
) -> jax.Array:
    """Render a splat in inference mode without autodiff support.

    Same forward result as splax.training.render, but guaranteed grad-free. With
    antialiased=True the Mip-Splatting opacity compensation is applied to the
    blend opacity, matching a model trained with the antialiased training render.

    gaussian_transforms is an optional (K, 4, 4) stack of rigid world-space
    transforms and gaussian_slices the K matching non-overlapping (start, stop)
    index ranges. The gaussians in slice k move by gaussian_transforms[k], all
    others stay static, and the transform happens on the fly in the projection
    kernel without copying the splat. Batching works through jax.vmap over the
    transform stack (and over the viewmat if desired), so each batch element
    renders the same scene with its gaussians at different poses. The slices are
    static Python values. Omitting both arguments is the plain path with
    identical output and performance.
    """
    n = means3d.shape[0]
    H, W = img_shape

    if (gaussian_transforms is None) != (gaussian_slices is None):
        raise ValueError(
            "gaussian_transforms and gaussian_slices must be passed together"
        )
    transform_ids = None
    if gaussian_transforms is not None and gaussian_slices is not None:
        if gaussian_transforms.shape[-3:] != (len(gaussian_slices), 4, 4):
            raise ValueError(
                f"gaussian_transforms shape {gaussian_transforms.shape} does not "
                f"match {len(gaussian_slices)} slices, expected (K, 4, 4)"
            )
        transform_ids = _transform_ids(n, gaussian_slices)

    opac = opacities.reshape(n)
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = _project_call(
        means3d,
        scales,
        quats,
        viewmat,
        opac,
        n,
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
        gaussian_transforms,
        transform_ids,
    )

    # In antialiased mode the blend opacity is compensated while the key emission
    # stays on the raw opacity, so the sort offsets remain valid.
    blend_opac = opacities
    map_opac = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opac = opacities * rho.reshape(opacities.shape)
        map_opac = opacities
    _final_Ts, _final_idx, out_img = _rasterize_call(
        colors,
        blend_opac,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        map_opac,
    )
    return out_img


__all__ = ["render"]
