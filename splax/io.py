"""PLY import/export for splats in render space.

``load_ply`` reads a 3DGS ``.ply`` and maps the stored fields to render-space inputs.

``write_ply`` takes the *render-space* splats and writes the inverse activation-space fields so a
subsequent ``load_ply`` reproduces them.

``fetch`` downloads remote assets into a local cache and returns the cached path, so examples and
tests can pull scenes on demand.
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
from scipy.special import logit

# SH degree-0 basis constant, shared with load_ply (colors = f_dc * C0 + 0.5).
_C0 = 0.28209479177387814


def fetch(
    url: str, *, cache: Path | None = None, force: bool = False, allow_unchecked: bool = False
) -> Path:
    """Download ``url`` into a local cache and return the path to the cached file.

    A cached file is reused only while its stored ETag still matches the remote. When the remote
    sends no ETag, the asset is downloaded on every call. Fetching with the ``force`` parameter
    ensures a fresh download. The cache directory defaults to ``$SPLAX_CACHE`` if set, else
    ``$XDG_CACHE_HOME/splax`` if set, else ``~/.cache/splax``.

    Args:
        url: URL to download.
        cache: Cache directory, overriding the environment-based default.
        force: Re-download and overwrite the cached copy even if it exists.
        allow_unchecked: Serve a cached file as-is instead of revalidating it against the remote.

    Returns:
        Path to the cached file.
    """
    if cache is None:
        xdg = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = Path(os.environ["SPLAX_CACHE"]) if "SPLAX_CACHE" in os.environ else xdg / "splax"
    assert isinstance(cache, Path), f"cache must be a Path, got {type(cache)}"
    name = Path(urllib.parse.urlparse(url).path).name
    path = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + name)
    token_path = cache / (path.name + ".etag")
    if not force and allow_unchecked and path.exists():
        return path
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
    colors = jnp.clip(sh0 * _C0 + 0.5, 0.0, 1.0)  # files may store out-of-range SH coefficients
    opacities = jax.nn.sigmoid(jnp.asarray(v["opacity"], jnp.float32))[:, None]
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
        path: Path to the output ``.ply`` file.
        means: World positions, shape ``(N, 3)``.
        scales: Positive per-axis scales, shape ``(N, 3)``.
        quats: wxyz quaternions, shape ``(N, 4)``.
        colors: RGB in ``[0, 1]``, shape ``(N, 3)``.
        opacities: Opacities in ``[0, 1]``, shape ``(N, 1)`` or ``(N,)``.
    """
    splats = [np.asarray(x, np.float32) for x in (means, scales, quats, colors, opacities)]
    means, scales, quats, colors, opacities = splats
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    f_dc = (colors - 0.5) / _C0
    # logit is infinite at exactly 0 and 1, clip to keep the stored values finite
    opac_logit = logit(np.clip(opacities.reshape(-1, 1), 1e-7, 1.0 - 1e-7))
    n = means.shape[0]
    data = np.hstack([means, np.zeros((n, 3)), f_dc, opac_logit, np.log(scales), quats])
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    fields += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    verts = np.empty(n, dtype=[(f, "f4") for f in fields])
    for field, column in zip(fields, data.T, strict=True):
        verts[field] = column
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))
