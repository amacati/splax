"""Differentiable rasterization stage.

``rasterize`` and ``rasterize_depth`` blend the projected gaussians into an image. The depth variant
additionally renders an expected depth map.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from splax._rasterize._kernels import (
    _rasterize_bwd_depth_ffi,
    _rasterize_bwd_ffi,
    _rasterize_depth_ffi,
    _rasterize_ffi,
)

# region public API


def rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    map_opacities: jax.Array | None = None,
) -> jax.Array:
    """Blend projected gaussians into an (H, W, 3) image.

    Differentiable with respect to colors, opacities, xys, and conics. background, depths, radii,
    and cum_tiles_hit are non-differentiable.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    out_img, _, _ = _rasterize(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
    )
    return out_img


def rasterize_depth(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Blend gaussians into an image (H, W, 3) and expected depth map (H, W, 1).

    Identical to rasterize but additionally renders the alpha-blended expected depth map with the
    same visibility weights as the color blend.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    out_img, out_depth, _, _ = _rasterize_depth(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
    )
    return out_img, out_depth


# region custom vjp


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Custom vjp for the blend, returning (out_img, final_Ts, final_idx).

    final_Ts and final_idx are the backward residuals, the per-pixel final transmittance and last
    contributing gaussian. The public rasterize discards them, the fwd rule keeps them. JAX requires
    a rigid array signature for custom_vjps, so the None default of map_opacities is resolved in the
    public rasterize before this is called.
    """
    final_Ts, final_idx, out_img = _rasterize_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        output_dims=(H, W),
    )
    return out_img, final_Ts, final_idx


def _rasterize_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Forward pass of _rasterize, reusing the primitive and keeping its residuals."""
    out = _rasterize(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    _, final_Ts, final_idx = out
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return out, residuals


def _rasterize_bwd(
    n: int, H: int, W: int, residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
) -> tuple[jax.Array | None, ...]:
    """Backward pass of _rasterize."""
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    v_img, _, _ = cotangents  # only the image cotangent is nonzero
    v_colors, v_opacity, v_xy, v_conic = _rasterize_bwd_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        int(n),
        int(H),
        int(W),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    # Cotangents for (colors, opacities, map_opacities, background, xys, depths, radii, conics,
    # cum_tiles_hit). map_opacities feeds only the integer key emission, so it is non-diff like
    # background, depths, radii, and the cumsum.
    return v_colors, v_opacity, None, None, v_xy, None, None, v_conic, None


_rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11))
def _rasterize_depth(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Custom vjp for the depth blend, returning (out_img, out_depth, final_Ts, final_idx)."""
    final_Ts, final_idx, out_img, out_depth = _rasterize_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        output_dims=(H, W),
    )
    return out_img, out_depth, final_Ts, final_idx


def _rasterize_depth_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Forward pass of _rasterize_depth, reusing the primitive and keeping its residuals."""
    out = _rasterize_depth(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
    )
    _, _, final_Ts, final_idx = out
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return out, residuals


def _rasterize_depth_bwd(
    n: int, H: int, W: int, residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...]
) -> tuple[jax.Array | None, ...]:
    """Backward pass of _rasterize_depth."""
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    v_img, v_depth_img, _, _ = cotangents
    v_colors, v_opacity, v_xy, v_conic, v_depths = _rasterize_bwd_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        v_depth_img,
        int(n),
        int(H),
        int(W),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    v_depths = v_depths.reshape(depths.shape)
    # Unlike the plain rasterize, depths carries a nonzero cotangent
    return v_colors, v_opacity, None, None, v_xy, v_depths, None, v_conic, None


_rasterize_depth.defvjp(_rasterize_depth_fwd, _rasterize_depth_bwd)
