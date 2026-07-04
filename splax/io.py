"""PLY import/export for splats in render space.

``load_ply`` reads a 3DGS ``.ply`` and maps the stored (activation-space) fields
to render-space inputs:

    scales  = exp(scale_i)                     # log-scale     -> scale
    quats   = normalize(rot_i)                 # raw quat       -> unit wxyz
    colors  = clip(f_dc_i * C0 + 0.5, 0, 1)    # SH deg-0 coeff -> rgb
    opac    = sigmoid(opacity)                 # logit          -> [0, 1]

``write_ply`` takes the *render-space* splats (the same tensors ``render``
consumes) and writes the inverse activation-space fields so a subsequent
``load_ply`` reproduces them. Normals are zeroed and ``f_rest`` (higher-order SH)
is omitted -- ``load_ply`` reads neither, so a written file round-trips exactly
through it (SH degree 0 only, which is all splax renders).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

# SH degree-0 basis constant, shared with load_ply (colors = f_dc * C0 + 0.5).
_C0 = 0.28209479177387814


def load_ply(
    path: str | Path,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Read a 3DGS ``.ply`` into the five render-space arrays ``render`` consumes.

    Returns (means, scales, quats, colors, opacities) as float32 jax arrays,
    shapes (N, 3), (N, 3), (N, 4), (N, 3), (N, 1). SH degree 0 only (f_rest
    ignored). Exact inverse of ``write_ply``.
    """
    v = PlyData.read(path)["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1))
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    sh0 = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)
    colors = np.clip(sh0 * _C0 + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"]))[..., None]
    return (
        jnp.asarray(means, jnp.float32),
        jnp.asarray(scales, jnp.float32),
        jnp.asarray(quats, jnp.float32),
        jnp.asarray(colors, jnp.float32),
        jnp.asarray(opacities, jnp.float32),
    )


def write_ply(
    path: str | Path,
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
) -> None:
    """Write render-space splats to a 3DGS ``.ply`` (inverse of ``load_ply``).

    Args mirror ``render``'s first five positional arguments:
      means      (N, 3) world positions
      scales     (N, 3) positive per-axis scales (stored as log)
      quats      (N, 4) wxyz quaternions (normalized on write)
      colors     (N, 3) rgb in [0, 1] (stored as SH deg-0 coeff (rgb-0.5)/C0)
      opacities  (N, 1) or (N,) opacity in [0, 1] (stored as logit)

    Fields written: x,y,z, nx,ny,nz (zero), f_dc_0..2, opacity, scale_0..2,
    rot_0..3 -- exactly the set ``load_ply`` consumes.
    """
    means = np.asarray(means, np.float32)
    scales = np.asarray(scales, np.float32)
    quats = np.asarray(quats, np.float32)
    colors = np.asarray(colors, np.float32)
    opacities = np.asarray(opacities, np.float32).reshape(-1)
    n = means.shape[0]

    log_scales = np.log(scales)
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    f_dc = (colors - 0.5) / _C0
    # logit: inverse of sigmoid. Clip away from {0,1} to keep it finite.
    opac = np.clip(opacities, 1e-7, 1.0 - 1e-7)
    opacity = np.log(opac / (1.0 - opac))

    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    verts = np.empty(n, dtype=[(f, "f4") for f in fields])
    verts["x"], verts["y"], verts["z"] = means[:, 0], means[:, 1], means[:, 2]
    verts["nx"] = verts["ny"] = verts["nz"] = 0.0
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = (
        f_dc[:, 0],
        f_dc[:, 1],
        f_dc[:, 2],
    )
    verts["opacity"] = opacity
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = (
        log_scales[:, 0],
        log_scales[:, 1],
        log_scales[:, 2],
    )
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = (
        quats[:, 0],
        quats[:, 1],
        quats[:, 2],
        quats[:, 3],
    )

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(str(path))
