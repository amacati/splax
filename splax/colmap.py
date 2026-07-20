"""COLMAP sparse reconstruction ingestion.

Binary readers for ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` and the fixed-N splat
initialization from the sparse point cloud.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, BinaryIO

import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree
from scipy.special import logit

if TYPE_CHECKING:
    from pathlib import Path

    import jax

_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _r(f: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path: str | Path) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    """Read ``cameras.bin``.

    Args:
        path: Path to the ``cameras.bin`` file.

    Returns:
        Mapping of camera id to ``(model_name, width, height, params)``.
    """
    cams = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            cid, model, w, h = _r(f, "iiQQ")
            name, n_params = _CAMERA_MODELS[model]
            cams[cid] = (name, w, h, _r(f, "d" * n_params))
    return cams


def read_images(path: str | Path) -> list[dict]:
    """Read ``images.bin`` into per-image pose and observation records.

    Args:
        path: Path to the ``images.bin`` file.

    Returns:
        List of dicts with keys ``id``, ``qvec`` (wxyz), ``tvec``, ``camera_id``, ``name``,
        ``obs_xy`` (K, 2 float64), and ``obs_pid`` (K, int64), sorted by image name. The
        observations are the 2D keypoints with a valid triangulated 3D point, used for depth
        regularization. Views with no depth loss simply ignore them.
    """
    imgs = []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, camid = _r(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (np2d,) = _r(f, "Q")
            # per point2D: x, y (double) + point3D_id (int64)
            rec = np.frombuffer(f.read(np2d * 24), np.uint8).reshape(np2d, 24)
            xy = rec[:, :16].copy().view(np.float64)
            pid = rec[:, 16:].copy().view(np.int64).ravel()
            keep = pid >= 0
            imgs.append(
                {
                    "id": iid,
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camid,
                    "name": name.decode(),
                    "obs_xy": xy[keep],
                    "obs_pid": pid[keep],
                }
            )
    imgs.sort(key=lambda d: d["name"])
    return imgs


def read_points3D(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ``points3D.bin``.

    Args:
        path: Path to the ``points3D.bin`` file.

    Returns:
        Positions (M, 3) float64, colors (M, 3) uint8, point ids (M,) int64, and track lengths
        (M,) int64.
    """
    xyz, rgb, ids, track_lens = [], [], [], []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            pid, x, y, z, rr, gg, bb, err = _r(f, "QdddBBBd")
            (tl,) = _r(f, "Q")
            f.read(tl * 8)  # track: (image_id int32, point2D_idx int32) * tl
            xyz.append((x, y, z))
            rgb.append((rr, gg, bb))
            ids.append(pid)
            track_lens.append(tl)
    return (
        np.asarray(xyz, np.float64),
        np.asarray(rgb, np.uint8),
        np.asarray(ids, np.int64),
        np.asarray(track_lens, np.int64),
    )


def knn_scales(xyz: np.ndarray, cap: float, k: int = 3) -> np.ndarray:
    """Log-scale init from the mean distance to the k nearest neighbours.

    Args:
        xyz: Point positions, shape ``(M, 3)``.
        cap: Upper bound on the distance before the log.
        k: Number of neighbours to average over.

    Returns:
        Log scales, shape ``(M,)``, as float32.
    """
    d, _ = KDTree(xyz).query(xyz, k=k + 1)  # includes self at dist 0
    dist = np.clip(d[:, 1:].mean(axis=1), 1e-4, cap)  # floor: duplicate points would log to -inf
    return np.log(dist).astype(np.float32)


def init_from_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    n: int,
    opacity: float,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> dict[str, jax.Array]:
    """Initialize a fixed-N splat from the sparse cloud.

    Subsamples when the cloud has more than ``n`` points and pads by jittered duplication when it
    has fewer.

    Args:
        xyz: Sparse point positions, shape ``(M, 3)``.
        rgb: Point colors as uint8, shape ``(M, 3)``.
        n: Number of gaussians to initialize.
        opacity: Initial opacity of every gaussian.
        seed: Seed for the subsample and padding draws.
        weights: Positive per-point sampling weights such as track lengths, shape ``(M,)``.

    Returns:
        Parameter dict with the ``render_log`` arrays ``means``, ``log_scales``, ``quats``,
        ``colors_logit``, and ``opac_logit``.
    """
    rng = np.random.default_rng(seed)
    m = xyz.shape[0]
    prob = None
    if weights is not None:
        prob = np.log1p(weights)
        prob = prob / prob.sum()
    # cap init gaussian size (normalized units, cameras sit at dist ~1) so a few isolated outlier
    # points don't seed giant gaussians.
    cap = 0.3
    if m >= n:
        sel = rng.choice(m, n, replace=False, p=prob)
        xyz_n, rgb_n = xyz[sel], rgb[sel]
        log_scales = knn_scales(xyz_n, cap)
    else:
        pad = n - m
        src = rng.choice(m, pad, replace=True, p=prob)
        base_ls = knn_scales(xyz, cap)  # (m,) at the SPARSE m-point density
        # N-aware scale correction. knn_scales is the mean nearest-neighbour distance at the
        # *sparse* density (m points spread through the scene volume V). Padding to n>m gaussians
        # raises the density to n/V, and for a roughly uniform cloud the mean NN spacing scales as
        # density^(-1/3). The per-gaussian scale at the target density is thus smaller by a factor
        # cbrt(n/m). We correct in log space by subtracting (1/3)ln(n/m) from every knn log-scale.
        # The jitter that spreads the padded copies uses the corrected (smaller) scale too, so the
        # seeded points sit at the target spacing. Only fires when padding (n>m).
        base_ls = base_ls - np.log(n / m) / 3.0
        jitter = rng.normal(size=(pad, 3)).astype(np.float32) * np.exp(base_ls[src])[:, None]
        xyz_n = np.concatenate([xyz, xyz[src] + jitter], 0)
        rgb_n = np.concatenate([rgb, rgb[src]], 0)
        log_scales = np.concatenate([base_ls, base_ls[src]], 0)
    # logit is infinite at exactly 0 and 1, which pure black/white uint8 colors hit
    colors_logit = logit(np.clip(rgb_n / 255.0, 1e-4, 1.0 - 1e-4))
    return {
        "means": jnp.asarray(xyz_n, jnp.float32),
        "log_scales": jnp.asarray(log_scales[:, None].repeat(3, 1)),
        "quats": jnp.asarray(rng.normal(size=(n, 4)), jnp.float32),
        "colors_logit": jnp.asarray(colors_logit, jnp.float32),
        "opac_logit": jnp.full((n, 1), logit(opacity), jnp.float32),
    }
