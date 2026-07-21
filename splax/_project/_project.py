"""Differentiable projection stage.

``project`` composes the Warp projection kernels from ``splax._project._kernels`` with a
``jax.custom_vjp``, so ``jax.grad`` flows through it with respect to the gaussian parameters, the
camera pose, and the rigid transforms. One backward kernel computes every gradient in a single pass,
so gradient selection is left to ``jax.grad``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from splax._project._kernels import _project_bwd_ffi, _project_ffi

if TYPE_CHECKING:
    from collections.abc import Sequence

# region public API


def project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    opacities: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float] | None = None,
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    gaussian_transforms: jax.Array | None = None,
    transform_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project gaussians to screen space with the opacity-aware tile intersection.

    The projection is differentiable through ``jax.custom_vjp`` with respect to the means, scales,
    quaternions, viewmat, and rigid transforms. A single backward kernel computes every gradient in
    one pass, and ``jax.grad`` keeps the ones its argnums select. Opacities determine the integer
    tile counts and carry no gradient through projection. Their gradient flows through rasterization
    instead.

    Args:
        mean3ds: Gaussian centers, shape ``(N, 3)``.
        scales: Per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        viewmat: World-to-camera matrix, shape ``(4, 4)``.
        opacities: Gaussian opacities, one entry per gaussian.
        img_shape: Image size as ``(height, width)`` in pixels.
        f: Focal lengths ``(fx, fy)`` in pixels.
        c: Principal point ``(cx, cy)`` in pixels, defaulting to the image center.
        glob_scale: Global factor applied to all scales.
        clip_thresh: Near-plane clipping threshold.
        gaussian_transforms: Rigid world-space transforms, shape ``(K, 4, 4)``.
        transform_ids: Per-gaussian transform index from ``_transform_ids``, shape ``(N,)``. Passed
            together with ``gaussian_transforms``.

    Returns:
        Tuple of the screen-space centers, depths, radii, conics, per-gaussian tile counts, and
        their inclusive prefix sum ``cum_tiles_hit``.
    """
    n = mean3ds.shape[0]
    if c is None:
        c = (img_shape[1] / 2, img_shape[0] / 2)
    has_transforms = gaussian_transforms is not None
    if not has_transforms:
        gaussian_transforms = jnp.zeros((1, 4, 4), jnp.float32)
        transform_ids = jnp.full((1,), -1, jnp.int32)
    # Warp works scalar-last (xyzw). Roll the wxyz inputs once here under JIT. jax.grad rolls the
    # returned quat gradient back to wxyz through this op, so the kernels stay purely xyzw.
    quats = jnp.roll(quats, -1, axis=-1)
    return _project(
        mean3ds,
        scales,
        quats,
        viewmat,
        opacities.reshape(n),
        gaussian_transforms,
        transform_ids,
        int(n),
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
        has_transforms,
    )


def opacity_compensation(conics: jax.Array, radii: jax.Array, eps: float = 0.3) -> jax.Array:
    """Compute the Mip-Splatting anti-aliased opacity compensation factor per gaussian.

    It cancels the artificial area inflation from the low-pass guard in the projection kernel by
    scaling back the opacities.

    The factor is computed in closed form from the projection's conics, so its gradient flows back
    to the scales, quaternions, and means through the existing projection vjp without any Warp
    kernel change. Culled gaussians with non-positive radii get a factor of 1.

    Args:
        conics: Inverse 2d covariances from projection, shape ``(N, 3)``.
        radii: Screen-space radii, non-positive for culled gaussians.
        eps: Screen-space dilation the projection applies, in squared pixels.

    Returns:
        Per-gaussian compensation factor, shape ``(N,)``.
    """
    conics = conics.reshape(-1, 3)
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    rho2 = 1.0 - eps * (a + c) + (eps * eps) * (a * c - b * b)
    rho = jnp.sqrt(jnp.clip(rho2, 0.0, 1.0))
    return jnp.where(radii.reshape(-1) > 0, rho, 1.0)


def transform_ids(n: int, slices: Sequence[tuple[int, int]]) -> jax.Array:
    """Map each gaussian to the transform it follows.

    Slices must be static, non-overlapping and inside [0, n). Violations raise to prevent corrupted
    renderings.
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


# region custom vjp


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    gaussian_transforms: jax.Array,
    transform_ids: jax.Array,
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    has_transforms: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Custom vjp for the projection.

    JAX requires a rigid array signature for custom_vjps. Since we allow for default None values in
    ``project``, the custom_vjp cannot be applied to ``project`` directly.
    """
    # gaussian_transforms and transform_ids are concrete arrays, the (1, 4, 4) / (1,) dummy
    # sentinels when has_transforms is False. The kernel skips the transform block in that case, so
    # the dummies never enter the math.
    H, W = img_shape
    num_transforms = gaussian_transforms.shape[-3]
    xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit = _project_ffi(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        gaussian_transforms,
        transform_ids,
        int(n),
        int(num_transforms),
        bool(has_transforms),
        int(H),
        int(W),
        float(f[0]),
        float(f[1]),
        float(c[0]),
        float(c[1]),
        float(glob_scale),
        float(clip_thresh),
        output_dims=n,
    )
    depths = depths.reshape(n, 1)
    radii = radii.reshape(n, 1)
    num_tiles_hit = num_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    cum_tiles_hit = cum_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    return xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit


def _project_fwd(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    gaussian_transforms: jax.Array,
    transform_ids: jax.Array,
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    has_transforms: bool,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple]:
    """Forward pass of _project, the custom_vjp of project."""
    # opacities carry no gradient through projection, they only drive integer tile counts. The
    # opacity gradient flows through rasterize.
    out = _project(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        gaussian_transforms,
        transform_ids,
        n,
        img_shape,
        f,
        c,
        glob_scale,
        clip_thresh,
        has_transforms,
    )
    _, _, radii, conics, _, _ = out
    return out, (mean3ds, scales, quats, viewmat, gaussian_transforms, transform_ids, radii, conics)


def _project_bwd(
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    has_transforms: bool,
    residuals: tuple,
    cotangents: tuple,
) -> tuple[jax.Array | None, ...]:
    """Backward pass of _project, the custom_vjp of project."""
    m, s, q, vm, tf, ids, radii, conics = residuals
    v_xys, v_depths, _v_radii, v_conics, _v_nth, _v_cum = cotangents
    r = radii.reshape(n).astype(jnp.int32)
    k = tf.shape[-3]
    dims = {"v_mean3d": n, "v_scale": n, "v_quat": n}
    dims["v_viewmat"] = (4, 4)
    dims["v_transforms"] = (k, 4, 4)
    v_mean, v_scale, v_quat, v_viewmat, v_transforms = _project_bwd_ffi(
        m,
        s,
        q,
        vm,
        r,
        conics,
        v_xys,
        v_depths.reshape(-1),
        v_conics,
        tf,
        ids,
        int(n),
        int(k),
        has_transforms,
        float(f[0]),
        float(f[1]),
        float(glob_scale),
        output_dims=dims,
    )
    return (v_mean, v_scale, v_quat, v_viewmat, None, v_transforms, None)


_project.defvjp(_project_fwd, _project_bwd)
