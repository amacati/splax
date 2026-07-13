"""PLY import/export for splats in render space.

``load_ply`` reads a 3DGS ``.ply`` and maps the stored fields to render-space inputs.

``write_ply`` takes the *render-space* splats and writes the inverse activation-space fields so a
subsequent ``load_ply`` reproduces them. Normals are zeroed and ``f_rest`` is omitted.

``fetch`` downloads remote assets into a local cache and returns the cached path, so examples and
tests can pull scenes on demand. It validates a cached file against the remote content token so a
re-uploaded asset is picked up without clearing the cache by hand.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from plyfile import PlyData, PlyElement

# SH degree-0 basis constant, shared with load_ply (colors = f_dc * C0 + 0.5).
_C0 = 0.28209479177387814


def fetch(url: str, *, cache: Path | None = None, force: bool = False) -> Path:
    """Download ``url`` into a local cache and return the path to the cached file.

    A cached file is reused only while its stored ETag still matches the remote, so a re-uploaded
    asset at the same URL is re-downloaded without clearing the cache. When the remote sends no
    ETag there is nothing to compare, so the asset is downloaded afresh on every call. Fetching with
    the ``force`` parameter ensures a fresh download. The cache directory defaults to
    ``$SPLAX_CACHE`` if set, else ``$XDG_CACHE_HOME/splax`` if set, else ``~/.cache/splax``.

    Args:
        url: URL to download.
        cache: Cache directory, overriding the environment-based default.
        force: Re-download and overwrite the cached copy even if it exists.

    Returns:
        Path to the cached file.
    """
    if cache is None:
        loc = os.environ.get("SPLAX_CACHE", os.environ.get("XDG_CACHE_HOME"))
        cache = Path(loc) if loc is not None else Path.home() / ".cache/splax"
    assert isinstance(cache, Path), f"cache must be a Path, got {type(cache)}"
    name = Path(urllib.parse.urlparse(url).path).name or "download"
    path = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + name)
    token_path = cache / (path.name + ".etag")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        etag = resp.headers.get("ETag")
    if not force and path.exists() and token_path.exists() and token_path.read_text() == etag:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    # Download to a temp file and atomically swap it in, so path is never left half-written.
    tmp = tempfile.NamedTemporaryFile(dir=cache, delete=False)
    try:
        with tmp, urllib.request.urlopen(url) as src:
            shutil.copyfileobj(src, tmp)
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    if etag is not None:
        token_path.write_text(etag)
    return path


def load_ply(path: Path) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Read a 3DGS ``.ply`` into the five render-space arrays ``render`` consumes.

    Args:
        path: Path to a 3DGS ``.ply`` file containing the fields ``x``, ``y``, ``z``,
            ``scale_0..2``, ``rot_0..3``, ``f_dc_0..2``, and ``opacity``.

    Returns:
        means, scales (N, 3), quats (N, 4), colors (N, 3), opacities (N, 1) as float32 jax arrays.
    """
    v = PlyData.read(str(path))["vertex"]
    means = jnp.asarray(np.stack([v["x"], v["y"], v["z"]], axis=-1), jnp.float32)
    scales = jnp.asarray(
        np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1)), jnp.float32
    )
    quats = jnp.asarray(np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1), jnp.float32)
    quats /= jnp.linalg.norm(quats, axis=-1, keepdims=True)
    sh0 = jnp.asarray(np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1), jnp.float32)
    colors = jnp.clip(sh0 * _C0 + 0.5, 0.0, 1.0)
    opacities = 1.0 / (1.0 + jnp.exp(-v["opacity"]))[..., None]
    return means, scales, quats, colors, opacities


def write_ply(
    path: Path,
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
) -> None:
    """Write render-space splats to a 3DGS ``.ply``.

    Args:
        path: Path to the output ``.ply`` file
        means: (N, 3) world positions
        scales: (N, 3) positive per-axis scales
        quats: (N, 4) wxyz quaternions
        colors: (N, 3) RGB in [0, 1]
        opacities: (N, 1) or (N,) opacity in [0, 1]
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
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
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
