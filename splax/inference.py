"""splax.inference: the pure, guaranteed grad-free rendering entry point.

``splax.inference.render`` is the inference-only twin of ``splax.training.render``.
Both share every Warp kernel; they differ only in the JAX-level wrapping:

- **No custom_vjp.** This path calls the projection / rasterization FFI *primals*
  (``_project_call`` / ``_rasterize_call``) directly, so there is no
  ``jax.custom_vjp`` interception and no residual-saving forward rule on the trace.
  Calling ``jax.grad`` through it raises (the FFIs have no autodiff rule) -- that is
  intentional: this is the documented grad-free contract.
- **No residual outputs.** The rasterize FFI always computes ``final_Ts`` /
  ``final_idx`` (they are cheap forward by-products), but this path discards them.
  Because nothing downstream reads them, XLA dead-code-eliminates them from the
  compiled program -- unlike the training forward, which must keep them live as
  custom_vjp residuals for the backward.
- **Tight O6 intersection** (opacity-aware SNUGBOX + AccuTile) is always used.
- **Batching.** vmap over viewmats/splats works exactly as in the training path
  (both underlying FFIs carry ``vmap_method="expand_dims"``).

The math is identical to ``splax.render`` on the forward; this module adds no new
kernels, only a grad-free wiring of the existing ones.
"""

from __future__ import annotations

import jax

from splax._project import _project_call, opacity_compensation
from splax import _rasterize as _R
from splax._rasterize import _rasterize_call, _rasterize_split_call


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
    glob_scale: float,
    clip_thresh: float,
    block_size: int = 16,
    antialiased: bool = False,
) -> jax.Array:
    """Pure-inference renderer: Warp projection + rasterization, no autodiff.

    Same signature and forward result as ``splax.render``, but guaranteed grad-free:
    it calls the FFI primals directly (no custom_vjp), uses the tight O6 tile
    intersection, and discards the ``final_Ts`` / ``final_idx`` blend residuals.
    For gradients use ``splax.training.render``.

    ``antialiased=True`` applies the Mip-Splatting opacity compensation (ρ from
    ``opacity_compensation``) to the blend opacity, matching a model trained with
    ``splax.training.render(..., antialiased=True)``. Default ``False`` is
    byte-identical to the plain grad-free path.
    """
    n = means3d.shape[0]
    H, W = img_shape

    # Opacity-aware tight (O6) projection: pass opacities so the projection emits
    # per-axis radii + an AccuTile ellipse-walk tile count (has_opac=1).
    opac = opacities.reshape(n)
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = _project_call(
        means3d,
        scales,
        quats,
        viewmat,
        opac,
        n,
        1,
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
        int(block_size),
    )

    # _rasterize_call returns (final_Ts, final_idx, out_img); keep only the image.
    # tight=True matches the tight projection above (required for valid sort offsets).
    # Antialiased: ρ-compensate the blend opacity; the tile count stays on raw opac.
    blend_opac = opacities
    map_opac = None
    if antialiased:
        rho = opacity_compensation(conics, radii)
        blend_opac = opacities * rho.reshape(opacities.shape)
        map_opac = opacities
    # Split-heavy-tile load balancing (phase 8t): opt-in, inference-only. Merges the
    # blend of heavy tile bins across blocks via associative segment compositing, then
    # discards nothing (returns just the image). Default off -> the plain byte-identical
    # blend. The training/differentiable path never takes this route.
    if _R._TILE_SPLIT:
        out_img = _rasterize_split_call(
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
            int(block_size),
            True,
            map_opac,
        )
        return out_img
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
        int(block_size),
        True,
        map_opac,
    )
    return out_img


__all__ = ["render"]
