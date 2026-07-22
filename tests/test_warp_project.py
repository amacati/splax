"""Parity tests: Warp splax.project vs gsplat.fully_fused_projection (forward).

gsplat is a different CUDA kernel from splax's Warp port, so the two cannot agree
bit-for-bit the way a faithful port would. They DO share the projection math
(EWA covariance, the same 0.3 px eps2d dilation, the same pinhole intrinsics), so
for every gaussian visible in both the projected quantities match to a tight
numeric tolerance:

  - xys (means2d): pixel coordinates, close to sub-pixel.
  - depths: camera-space z, essentially identical.
  - conics: inverse projected-2D-covariance (a, b, c), close.

Integer tile counts (radii / num_tiles_hit) are NOT compared. gsplat returns a
per-axis pixel radius under a different visibility/tiling convention than splax's
scalar 3-sigma radius, so only the visibility they induce is cross-checked (the
gaussians each backend keeps agree on the overwhelming majority). See
tests/_gsplat_ref.py for the full list of convention conversions.

Everything here needs the gsplat reference, so the module guard fails the
whole file loudly when gsplat cannot run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import _gsplat_ref as gref
import jax
import jax.numpy as jnp
import numpy as np

import splax

if TYPE_CHECKING:
    from pathlib import Path

# Every test here needs the gsplat reference, fail the whole module without it.
gref.require_working(allow_module_level=True)


class _ProjArgs(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


PROJ_ARGS: _ProjArgs = {
    "img_shape": (256, 256),
    "f": (256.0, 256.0),
    "c": (128, 128),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}
REF_ARGS = PROJ_ARGS


def _random_inputs(n: int, seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(seed)
    k = jax.random.split(key, 3)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    viewmat = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return means, scales, quats, viewmat


def _assert_parity(
    splax_out: tuple[jax.Array, ...],
    ref: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    atol: float = 2e-3,
) -> None:
    xys_s, depths_s, radii_s, conics_s = (np.asarray(x) for x in splax_out[:4])
    radii_g, xys_g, depths_g, conics_g = ref

    vis_s = radii_s.ravel() > 0
    vis_g = (radii_g > 0).all(-1) if radii_g.ndim == 2 else radii_g.ravel() > 0

    # The two visibility sets agree on the vast majority of gaussians (the few that
    # differ sit exactly on a cull boundary). Cross-check on the intersection.
    mask = vis_s & vis_g
    assert mask.sum() > 0
    agree = np.mean(vis_s == vis_g)
    assert agree > 0.98, f"visibility agreement only {agree:.3%}"

    np.testing.assert_allclose(xys_s[mask], xys_g[mask], atol=atol)
    np.testing.assert_allclose(depths_s.ravel()[mask], depths_g.ravel()[mask], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(conics_s[mask], conics_g[mask], atol=atol, rtol=1e-3)


def test_parity_random_10k() -> None:
    means, scales, quats, viewmat = _random_inputs(10_000, seed=1)
    opac = jnp.full((means.shape[0],), 0.99)
    a = splax.project(means, scales, quats, viewmat, opacities=opac, **PROJ_ARGS)
    b = gref.project(means, scales, quats, viewmat, **REF_ARGS)
    _assert_parity(a, b)


def test_parity_under_jit() -> None:
    means, scales, quats, viewmat = _random_inputs(10_000, seed=2)
    opac = jnp.full((means.shape[0],), 0.99)
    a = jax.jit(lambda m, s, q, v: splax.project(m, s, q, v, opacities=opac, **PROJ_ARGS))(
        means, scales, quats, viewmat
    )
    b = gref.project(means, scales, quats, viewmat, **REF_ARGS)
    _assert_parity(a, b)


def test_parity_lego_slice(lego_meta: dict, lego_ply: Path) -> None:
    means, scales, quats, _colors, _opac = splax.io.load_ply(lego_ply)
    means, scales, quats = means[:50_000], scales[:50_000], quats[:50_000]
    frame = lego_meta["frames"][0]
    W = H = 800
    ff = 0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"])
    viewmat = jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"]))
    args: _ProjArgs = {
        **PROJ_ARGS,
        "img_shape": (H, W),
        "f": (float(ff), float(ff)),
        "c": (W // 2, H // 2),
    }
    opac = jnp.full((means.shape[0],), 0.99)
    a = splax.project(means, scales, quats, viewmat, opacities=opac, **args)
    b = gref.project(means, scales, quats, viewmat, **args)
    _assert_parity(a, b)
