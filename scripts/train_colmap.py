"""Fit a fixed set of Gaussians to a COLMAP scene with the splax Warp backend.

Generalized trainer for any COLMAP sparse reconstruction (``sparse/0`` with
``cameras.bin``, ``images.bin``, ``points3D.bin`` + an ``images/`` folder). It
reuses the Phase 6d MCMC quality recipe (fixed-N relocation + noise via
``splax.mcmc``, per-parameter LR schedules, L1 + 0.2 D-SSIM, opacity/scale
regularizers) that reached 33 dB on lego -- see ``scripts/train_lego.py`` and
``reports/phase6d.md``.

Key differences from the synthetic-lego trainer:

* **Real camera model.** COLMAP intrinsics are read straight from ``cameras.bin``.
  The renderer is a pinhole (``f=(fx,fy)``, ``c=(cx,cy)``); if the COLMAP camera
  carries distortion (SIMPLE_RADIAL / RADIAL / OPENCV ...), it is *ignored* and
  the pinhole approximation is used. For the drone scene the OPENCV distortion
  coefficients are tiny (k1~0.012, k2~-0.028, p1/p2~1e-3), so the approximation
  is acceptable; documented in ``reports/phase8f_drone_fit.md``.
* **Real extrinsics.** COLMAP stores world-to-camera ``(qvec, tvec)`` in the
  OpenCV convention (+z forward, +y down, +x right) -- exactly what
  ``splax.render``'s ``viewmat`` expects -- so ``viewmat = [[R, t], [0, 1]]``
  with ``R = quat2mat(qvec)`` directly, no axis flip (unlike the NeRF/OpenGL
  loader in ``train_lego.py`` which multiplies by ``diag(1,-1,-1,1)``).
* **Point-cloud initialization.** Gaussians are initialized *from* the sparse
  point cloud: means = 3D points, colors = their RGB, scales = log of the
  nearest-neighbour distance, moderate opacity, random quats. This is what makes
  real-scene fixed-N training converge. Padded (by jittered duplication) or
  subsampled to a fixed ``N`` for static shapes.
* **Scene normalization.** The scene is normalized by a similarity transform so
  the mean camera distance to the camera centroid is 1 (points + camera poses
  transformed consistently -- rendering is invariant to a world similarity), so
  the lego 6d LR/noise defaults transfer directly. The exported ``.ply`` is in
  this normalized frame.

Usage:
  python scripts/train_colmap.py --data data/drone --out-ply data/scenes/drone.ply
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import BinaryIO, cast

import dm_pix
import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.spatial import KDTree

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import splax

# --------------------------------------------------------------------------- #
# COLMAP binary readers (hand-parsed; format per colmap/scripts/read_write_model)
# --------------------------------------------------------------------------- #
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


def read_cameras(
    path: str | Path,
) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    """Return {camera_id: (model_name, w, h, params-tuple)}."""
    cams = {}
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            cid, model, w, h = _r(f, "iiQQ")
            name, npar = _CAMERA_MODELS[model]
            params = _r(f, "d" * npar)
            cams[cid] = (name, w, h, params)
    return cams


def read_images(path: str | Path) -> list[dict]:
    """Return list of dicts {id, qvec, tvec, camera_id, name, obs_xy, obs_pid}.

    ``obs_xy`` (K,2 float64) / ``obs_pid`` (K, int64) are the per-image 2D keypoint
    observations that have a valid triangulated 3D point (point3D_id != -1) -- these
    are the COLMAP sparse points visible in this view, used for depth regularization
    (survey T2). Views with no depth loss simply ignore them.
    """
    imgs = []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, camid = _r(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (np2d,) = _r(f, "Q")
            buf = f.read(np2d * 24)  # per point2D: x,y (double) + point3D_id (int64)
            if np2d:
                rec = np.frombuffer(buf, dtype=np.uint8).reshape(np2d, 24)
                xy = rec[:, :16].copy().view(np.float64).reshape(np2d, 2)
                pid = rec[:, 16:].copy().view(np.int64).reshape(np2d)
                keep = pid >= 0
                obs_xy, obs_pid = xy[keep], pid[keep]
            else:
                obs_xy = np.zeros((0, 2), np.float64)
                obs_pid = np.zeros((0,), np.int64)
            imgs.append(
                {
                    "id": iid,
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camid,
                    "name": name.decode(),
                    "obs_xy": obs_xy,
                    "obs_pid": obs_pid,
                }
            )
    imgs.sort(key=lambda d: d["name"])
    return imgs


def read_points3D(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz (M,3) float64, rgb (M,3) uint8, ids (M,) int64)."""
    xyz, rgb, ids = [], [], []
    with open(path, "rb") as f:
        (n,) = _r(f, "Q")
        for _ in range(n):
            pid, x, y, z, rr, gg, bb, err = _r(f, "QdddBBBd")
            (tl,) = _r(f, "Q")
            f.read(tl * 8)  # track: (image_id int32, point2D_idx int32) * tl
            xyz.append((x, y, z))
            rgb.append((rr, gg, bb))
            ids.append(pid)
    return (
        np.asarray(xyz, np.float64),
        np.asarray(rgb, np.uint8),
        np.asarray(ids, np.int64),
    )


def quat2mat(q: np.ndarray) -> np.ndarray:
    """COLMAP wxyz quaternion -> 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------- #
# Scene loading
# --------------------------------------------------------------------------- #
def _view_depth_targets(
    im: dict,
    vm: np.ndarray,
    id2row: dict[int, int],
    pts_xyz_norm: np.ndarray,
    r: float,
    W: int,
    H: int,
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-reg targets for one view (survey T2): pixel coords (downscaled) + the
    COLMAP sparse-point camera-space depths, to a fixed ``max_pts`` (mask the rest).

    Returns (uv (max_pts,2) f32, depth (max_pts,) f32, mask (max_pts,) f32). The
    target depth is the sparse point's z in this camera (same normalized units as the
    rendered expected-depth channel). Off-image / behind-camera observations are
    dropped; if more than max_pts survive, a random subset is kept."""
    uv = np.zeros((max_pts, 2), np.float32)
    depth = np.zeros((max_pts,), np.float32)
    mask = np.zeros((max_pts,), np.float32)
    rows = np.array([id2row.get(int(p), -1) for p in im["obs_pid"]], np.int64)
    ok = rows >= 0
    if not ok.any():
        return uv, depth, mask
    X = pts_xyz_norm[rows[ok]]  # (K,3) normalized world points
    z = X @ vm[:3, :3].T + vm[:3, 3]  # camera-space coords
    cam_z = z[:, 2]
    px = im["obs_xy"][ok] * r  # downscaled pixel coords (x, y)
    valid = (
        (cam_z > 1e-3)
        & (px[:, 0] >= 0)
        & (px[:, 0] < W)
        & (px[:, 1] >= 0)
        & (px[:, 1] < H)
    )
    px, cam_z = px[valid], cam_z[valid]
    k = px.shape[0]
    if k == 0:
        return uv, depth, mask
    if k > max_pts:
        sel = rng.choice(k, max_pts, replace=False)
        px, cam_z, k = px[sel], cam_z[sel], max_pts
    uv[:k] = px.astype(np.float32)
    depth[:k] = cam_z.astype(np.float32)
    mask[:k] = 1.0
    return uv, depth, mask


def load_scene(
    data_dir: str | Path,
    downscale: int,
    eval_every: int,
    max_depth_pts: int = 2048,
    seed: int = 0,
    sparse_model: int = 0,
) -> dict:
    """Load a COLMAP scene, normalized, downscaled. Returns a dict of arrays.

    Views carry (image float32 HxWx3 in [0,1], viewmat 4x4). Intrinsics are a
    single shared pinhole (fx, fy, cx, cy) at the downscaled resolution. The
    world is similarity-normalized (mean camera distance -> 1); the same
    transform is applied to the sparse points used for initialization.
    """
    data_dir = Path(data_dir)
    # COLMAP can emit several disconnected sub-models (sparse/0, 1, ...); the largest
    # is not always 0 (data/room's usable model is 2), so the index is selectable.
    sparse = data_dir / "sparse" / str(sparse_model)
    cams = read_cameras(sparse / "cameras.bin")
    images = read_images(sparse / "images.bin")
    pts_xyz, pts_rgb, pts_ids = read_points3D(sparse / "points3D.bin")
    id2row = {int(pid): i for i, pid in enumerate(pts_ids)}

    # --- camera centers + similarity normalization (mean cam dist -> 1) ------
    centers = np.array([-quat2mat(im["qvec"]).T @ im["tvec"] for im in images])
    ctr = np.median(centers, axis=0)
    s = 1.0 / np.mean(np.linalg.norm(centers - ctr, axis=1))

    def normalize_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Similarity-transform a w2c pose: X' = s (X - ctr). R stays, t' = s(t + R ctr)."""
        R = quat2mat(qvec)
        t_new = s * (tvec + R @ ctr)
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t_new
        return vm

    pts_xyz = (s * (pts_xyz - ctr)).astype(np.float32)

    # --- intrinsics (pinhole; distortion ignored) ---------------------------
    cam_name, W0, H0, params = cams[images[0]["camera_id"]]
    if cam_name in (
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "FOV",
    ):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:  # PINHOLE, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, THIN_PRISM_FISHEYE
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    W, H = W0 // downscale, H0 // downscale
    r = W / W0  # exact ratio (W0 divisible by downscale for the drone: 3840/4)
    intr = (fx * r, fy * r, cx * r, cy * r)

    # --- load + downscale images (box filter) -------------------------------
    train_imgs, train_vms, eval_imgs, eval_vms, eval_names = [], [], [], [], []
    tp_uv, tp_depth, tp_mask = [], [], []  # per-train-view depth-reg targets (T2)
    tgt_rng = np.random.default_rng(seed)
    print(f"loading {len(images)} images at {W}x{H} (downscale {downscale}) ...")
    for i, im in enumerate(images):
        fp = data_dir / "images" / im["name"]
        arr = iio.imread(fp)
        Hi, Wi = arr.shape[:2]
        fh, fw = Hi // H, Wi // W
        arr = arr[: H * fh, : W * fw].astype(np.float32) / 255.0
        arr = arr.reshape(H, fh, W, fw, 3).mean((1, 3))  # box downsample
        vm = normalize_pose(im["qvec"], im["tvec"])
        if i % eval_every == 0:
            eval_imgs.append(arr)
            eval_vms.append(vm)
            eval_names.append(im["name"])
        else:
            train_imgs.append(arr)
            train_vms.append(vm)
            uv, dep, msk = _view_depth_targets(
                im, vm, id2row, pts_xyz, r, W, H, max_depth_pts, tgt_rng
            )
            tp_uv.append(uv)
            tp_depth.append(dep)
            tp_mask.append(msk)
    return {
        "train_imgs": np.stack(train_imgs),
        "train_vms": np.stack(train_vms),
        "eval_imgs": np.stack(eval_imgs),
        "eval_vms": np.stack(eval_vms),
        "eval_names": eval_names,
        "H": H,
        "W": W,
        "intr": intr,
        "pts_xyz": pts_xyz,
        "pts_rgb": pts_rgb,
        "cam_name": cam_name,
        "cam_params": params,
        "norm_scale": float(s),
        "norm_center": ctr,
        "train_pts_uv": np.stack(tp_uv),
        "train_pts_depth": np.stack(tp_depth),
        "train_pts_mask": np.stack(tp_mask),
    }


# --------------------------------------------------------------------------- #
# Point-cloud initialization
# --------------------------------------------------------------------------- #
def knn_scales(xyz: np.ndarray, k: int = 3, cap: float | None = None) -> np.ndarray:
    """Log-scale init = log(mean distance to k nearest neighbours)."""
    tree = KDTree(xyz)
    d, _ = tree.query(xyz, k=k + 1)  # includes self at dist 0
    dist = d[:, 1:].mean(axis=1)
    dist = np.clip(dist, 1e-4, cap if cap else np.inf)
    return np.log(dist).astype(np.float32)


def init_from_points(
    xyz: np.ndarray, rgb: np.ndarray, n: int, opa: float, seed: int = 0
) -> dict[str, jax.Array]:
    """Fixed-N init from the sparse cloud (pad by jittered duplication / subsample)."""
    rng = np.random.default_rng(seed)
    m = xyz.shape[0]
    # cap init gaussian size (normalized units; cameras sit at dist ~1) so a few
    # isolated outlier points don't seed giant gaussians.
    cap = 0.3
    if m >= n:
        sel = rng.choice(m, n, replace=False)
        xyz_n, rgb_n = xyz[sel], rgb[sel]
        log_scales = knn_scales(xyz_n, cap=cap)
    else:
        pad = n - m
        src = rng.integers(0, m, pad)
        base_ls = knn_scales(xyz, cap=cap)  # (m,) at the SPARSE m-point density
        # N-aware scale correction. knn_scales is the mean nearest-neighbour distance at
        # the *sparse* density (m points spread through the scene volume V). Padding to
        # n>m gaussians raises the density to n/V, and for a roughly uniform cloud the
        # mean NN spacing scales as density^(-1/3); so the honest per-gaussian scale at
        # the target density is smaller by a factor cbrt(n/m). Without this every gaussian
        # is instantiated ~cbrt(n/m)x too large -- the diagnosed 1.5M-drone init ceiling
        # (reports/phase8i: 49,471 pts padded ~30x carried 49k-density scales ~3x too
        # large, stalling at ~12.4 dB). We correct in log space by subtracting (1/3)ln(n/m)
        # from every knn log-scale (originals and padded copies alike -- after padding the
        # whole cloud lives at n/V density). The jitter that spreads the padded copies uses
        # the corrected (smaller) scale too, so the seeded points sit at the target spacing.
        # Only fires when padding (n>m); the subsample branch above is untouched.
        base_ls = base_ls - np.log(n / m) / 3.0
        jitter = (
            rng.normal(size=(pad, 3)).astype(np.float32) * np.exp(base_ls[src])[:, None]
        )
        xyz_n = np.concatenate([xyz, xyz[src] + jitter], 0)
        rgb_n = np.concatenate([rgb, rgb[src]], 0)
        log_scales = np.concatenate([base_ls, base_ls[src]], 0)
    colors = np.clip(rgb_n.astype(np.float32) / 255.0, 1e-4, 1 - 1e-4)
    colors_logit = np.log(colors / (1 - colors))
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    opac_logit = np.full((n, 1), float(np.log(opa / (1 - opa))), np.float32)
    return {
        "means": jnp.asarray(xyz_n.astype(np.float32)),
        "log_scales": jnp.asarray(log_scales[:, None].repeat(3, 1)),
        "quats": jnp.asarray(quats),
        "colors_logit": jnp.asarray(colors_logit),
        "opac_logit": jnp.asarray(opac_logit),
    }


# --------------------------------------------------------------------------- #
# Rendering / metrics
# --------------------------------------------------------------------------- #
def render_params(
    p: dict[str, jax.Array],
    viewmat: jax.Array,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
    background: jax.Array | None = None,
    antialiased: bool = False,
    render_depth: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    fx, fy, cx, cy = intr
    means = p["means"]
    scales = jnp.exp(p["log_scales"])
    quats = p["quats"] / (jnp.linalg.norm(p["quats"], axis=-1, keepdims=True) + 1e-8)
    colors = jax.nn.sigmoid(p["colors_logit"])
    opac = jax.nn.sigmoid(p["opac_logit"])
    if background is None:
        background = jnp.ones(3)
    return splax.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        background=background,
        img_shape=(H, W),
        f=(fx, fy),
        c=(cx, cy),
        glob_scale=1.0,
        clip_thresh=0.01,
        antialiased=antialiased,
        render_depth=render_depth,
    )


def _bilinear_sample(D: jax.Array, uv: jax.Array) -> jax.Array:
    """Bilinearly sample the (H, W) depth map at pixel coords ``uv`` (K, 2) = (x, y).

    Pixel (j, i) center is at (j+0.5, i+0.5), so we shift by -0.5 before interpolating.
    Out-of-range coords are clamped (masked points contribute nothing to the loss)."""
    H, W = D.shape
    x = jnp.clip(uv[:, 0] - 0.5, 0.0, W - 1.0)
    y = jnp.clip(uv[:, 1] - 0.5, 0.0, H - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)
    wx = x - x0
    wy = y - y0
    d00 = D[y0, x0]
    d01 = D[y0, x1]
    d10 = D[y1, x0]
    d11 = D[y1, x1]
    top = d00 * (1.0 - wx) + d01 * wx
    bot = d10 * (1.0 - wx) + d11 * wx
    return top * (1.0 - wy) + bot * wy


# --------------------------------------------------------------------------- #
# Per-image exposure correction (affine variant of survey item T5)
# --------------------------------------------------------------------------- #
# Real captures (phone/drone) drift in exposure / white-balance across frames;
# without correction the splat absorbs that per-view color error as spurious
# view-dependent color. The affine fix learns one 3x4 color transform per
# *training* image -- [M (3x3) | b (3)], applied as ``M @ rgb + b`` per pixel to
# the *rendered* RGB before the loss only -- so the shared 3D color no longer has
# to explain per-image ISP variation. This is the cheap first step the T5 survey
# recommends ("start with the affine-exposure variant; full bilateral grid only
# if real captures show ISP drift"); the analogue of gsplat's per-image
# AppearanceOptModule / bilateral-grid post-processing, minus the grid + MLP.
#
# Honesty at eval: held-out views have NO learned transform (they were never in
# the optimizer), and letting eval fit its own transform would let it cheat by
# regressing the render onto the GT. So eval always scores the RAW render vs GT
# (``render_params`` unchanged) -- the exposure params touch the training loss
# only and are excluded from every reported metric and the exported .ply.
def init_exposure(ntr: int) -> jax.Array:
    """Per-training-image affine color transforms, identity-initialized.

    Returns (ntr, 3, 4): each 3x4 block is [M (3x3) | b (3)]. Init M=I, b=0 so the
    transform is the identity at step 0 -- an unoptimized transform leaves the
    render bit-identical, which is what keeps the ``--exposure-opt`` off-path
    (never constructed) and the pre-optimization state faithful.
    """
    eye = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (ntr, 3, 3))
    off = jnp.zeros((ntr, 3, 1), jnp.float32)
    return jnp.concatenate([eye, off], axis=2)


def apply_exposure(img: jax.Array, affine: jax.Array) -> jax.Array:
    """Apply one image's 3x4 affine color transform to an (H, W, 3) render.

    ``affine`` is (3, 4) = [M (3x3) | b (3)]; returns ``M @ rgb + b`` per pixel.
    Pure JAX, no kernels -- used on the rendered RGB before the training loss only.
    """
    M, b = affine[:, :3], affine[:, 3]
    return jnp.einsum("ij,hwj->hwi", M, img) + b


def psnr(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
    mse = float(np.mean((np.clip(np.asarray(a), 0, 1) - np.asarray(b)) ** 2))
    return -10 * np.log10(mse) if mse > 0 else float("inf")


def save_ply(path: str | Path, params: dict[str, jax.Array]) -> None:
    scales = jnp.exp(params["log_scales"])
    quats = params["quats"] / (
        jnp.linalg.norm(params["quats"], axis=-1, keepdims=True) + 1e-8
    )
    colors = jax.nn.sigmoid(params["colors_logit"])
    opac = jax.nn.sigmoid(params["opac_logit"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    splax.write_ply(path, params["means"], scales, quats, colors, opac)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# MCMC helpers (generic; mirror scripts/train_lego.py)
# --------------------------------------------------------------------------- #
def _reset_opt_state(
    opt_state: optax.OptState, reset_mask: jax.Array
) -> optax.OptState:
    n = reset_mask.shape[0]
    keep = (~reset_mask).astype(jnp.float32)

    def z(x: jax.Array) -> jax.Array:
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == n:
            return x * keep.reshape((-1,) + (1,) * (x.ndim - 1))
        return x

    return jax.tree.map(z, opt_state)


def _make_step(
    opt: optax.GradientTransformation,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
    ssim_lambda: float,
    opacity_reg: float,
    scale_reg: float,
    antialiased: bool = False,
    depth_loss: bool = False,
    depth_lambda: float = 1e-2,
    exp_tx: optax.GradientTransformation | None = None,
    batch: int = 1,
) -> Callable:
    """Build a jitted train step. ``bg`` is a per-step render-side background color
    (gsplat ``random_bkgd``; see the ``--random-bkgd`` caveat in ``main()``): unlike
    the lego trainer, COLMAP photos carry no alpha channel, so only the *render* is
    composited over ``bg`` -- the GT photo is fixed real content and is never
    recomposited. When disabled the caller passes a fixed white ``bg`` (bit-identical
    to the pre-T1 code path).

    ``depth_loss`` (survey T2, gsplat ``depth_loss``) adds a scale-normalized masked L1
    between the rendered expected-depth channel (bilinearly sampled at the COLMAP
    sparse-point pixels of the view) and those points' camera-space depths, weighted by
    ``depth_lambda`` (gsplat default 1e-2). When OFF, the depth branch is never traced,
    so the loss is bit-identical to the pre-T2 path (the ``pts_*`` step args are unused).

    ``exp_tx`` (survey T5 affine variant, ``--exposure-opt``): when given an optax
    optimizer, the step also takes the per-image exposure table ``exp_p`` + its opt
    state + the view index ``vi``, applies that view's 3x4 affine to the render before
    the photometric terms (see ``apply_exposure``), and returns the updated table.
    When None the exposure branch is never traced — off-path bit-identical.

    ``batch`` (survey T6, gsplat ``batch_size``): views processed per step. Every
    per-view step arg (``gt``, ``vm``, ``bg``, ``vi``, ``pts_*``) carries a leading axis
    of size ``batch``; ``per_view`` computes one view's photometric + depth terms and is
    ``jax.vmap``-ed over that axis (the shared scene ``p``/``exp_p`` are broadcast, so the
    grad is ONE batch-native backward launch — Phase 8a). The batch's per-view L1 and
    D-SSIM are *mean-reduced* (gsplat averages the loss over the batch); the opacity/scale
    regularizers are added once per step (they depend only on ``p``). Interactions:
    ``random_bkgd`` draws ``batch`` independent backgrounds; ``depth_loss`` stacks
    ``batch`` target sets; ``exposure-opt`` indexes ``batch`` affine rows (their grads land
    in their own rows — the vjp of the broadcast ``exp_p`` sums per-view, and each view
    touches a distinct row). At ``batch==1`` every per-view arg arrives with a leading
    1-axis holding exactly the arrays the pre-T6 single-view step received, and the
    size-1 vmap+mean is the identity, so the default path is numerically identical.
    """

    def per_view(
        p: dict[str, jax.Array],
        exp_p: jax.Array | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Photometric + depth terms for ONE view (vmapped over the batch axis)."""
        if depth_loss:
            img, depth = render_params(
                p,
                vm,
                H,
                W,
                intr,
                background=bg,
                antialiased=antialiased,
                render_depth=True,
            )
            assert depth is not None
            dpred = _bilinear_sample(depth, pts_uv)
            npts = jnp.sum(pts_mask) + 1e-8
            # per-view scale normalization: divide the L1 residual by the mean target
            # depth so the term is dimensionless / scale-invariant.
            scale = jnp.sum(pts_mask * pts_depth) / npts + 1e-8
            dl = jnp.sum(pts_mask * jnp.abs(dpred - pts_depth)) / npts / scale
        else:
            img, _ = render_params(
                p, vm, H, W, intr, background=bg, antialiased=antialiased
            )
            dl = jnp.array(0.0, jnp.float32)
        if exp_tx is not None:
            affine = jax.lax.dynamic_index_in_dim(exp_p, vi, axis=0, keepdims=False)
            img = apply_exposure(img, affine)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = jnp.asarray(1.0 - dm_pix.ssim(img, gt))
        return l1, dssim, dl

    def loss_fn(
        p: dict[str, jax.Array],
        exp_p: jax.Array | None,
        gt: jax.Array,
        vm: jax.Array,
        bg: jax.Array,
        vi: jax.Array,
        pts_uv: jax.Array,
        pts_depth: jax.Array,
        pts_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        l1s, dssims, dls = jax.vmap(
            per_view, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0)
        )(p, exp_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask)
        l1 = jnp.mean(l1s)  # batch-mean photometric (gsplat)
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * jnp.mean(dssims)
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(p["log_scales"]))
        if depth_loss:
            loss = loss + depth_lambda * jnp.mean(dls)
        return loss, l1

    if exp_tx is None:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
            vi = jnp.zeros((batch,), jnp.int32)  # unused when exp_tx is None
            (loss, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                p, None, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask
            )
            updates, opt_state = opt.update(grads, opt_state, p)
            # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
            return (
                cast(dict[str, jax.Array], optax.apply_updates(p, updates)),
                opt_state,
                l1,
            )
    else:

        @jax.jit
        def step(
            p: dict[str, jax.Array],
            opt_state: optax.OptState,
            exp_p: jax.Array,
            exp_state: optax.OptState,
            gt: jax.Array,
            vm: jax.Array,
            bg: jax.Array,
            vi: jax.Array,
            pts_uv: jax.Array,
            pts_depth: jax.Array,
            pts_mask: jax.Array,
        ) -> tuple[
            dict[str, jax.Array], optax.OptState, jax.Array, optax.OptState, jax.Array
        ]:
            (loss, l1), (grads, exp_grads) = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(p, exp_p, gt, vm, bg, vi, pts_uv, pts_depth, pts_mask)
            updates, opt_state = opt.update(grads, opt_state, p)
            exp_updates, exp_state = exp_tx.update(exp_grads, exp_state, exp_p)
            # apply_updates is typed as the broad optax ArrayTree; the pytrees keep their types.
            return (
                cast(dict[str, jax.Array], optax.apply_updates(p, updates)),
                opt_state,
                cast(jax.Array, optax.apply_updates(exp_p, exp_updates)),
                exp_state,
                l1,
            )

    return step


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> dict:
    scene = load_scene(
        args.data,
        args.downscale,
        args.eval_every,
        max_depth_pts=args.max_depth_pts,
        seed=args.seed,
        sparse_model=args.sparse_model,
    )
    H, W, intr = scene["H"], scene["W"], scene["intr"]
    ntr = scene["train_imgs"].shape[0]
    print(
        f"camera model {scene['cam_name']} -> pinhole intr(fx,fy,cx,cy)="
        f"{tuple(round(v, 2) for v in intr)}  norm_scale={scene['norm_scale']:.4f}"
    )
    print(
        f"{ntr} train / {len(scene['eval_names'])} eval views; "
        f"{scene['pts_xyz'].shape[0]} sparse points -> {args.n} gaussians"
    )

    params = init_from_points(
        scene["pts_xyz"], scene["pts_rgb"], args.n, args.init_opa, args.seed
    )

    # host-side image stacks; move one view per step (keeps GPU memory modest)
    train_imgs = scene["train_imgs"]
    train_vms = jnp.asarray(scene["train_vms"])
    # depth-reg targets (survey T2); kept on host, one view moved per step.
    tp_uv = scene["train_pts_uv"]
    tp_depth = scene["train_pts_depth"]
    tp_mask = scene["train_pts_mask"]
    if args.depth_loss:
        vis = float(tp_mask.sum(1).mean())
        print(
            f"depth-reg ON (lambda {args.depth_lambda}): "
            f"~{vis:.0f}/{args.max_depth_pts} sparse points per train view"
        )
    eval_imgs = [scene["eval_imgs"][i] for i in range(len(scene["eval_names"]))]
    eval_vms = [jnp.asarray(scene["eval_vms"][i]) for i in range(len(eval_imgs))]

    def eval_psnr(idxs: list[int]) -> list[float]:
        return [
            psnr(
                render_params(
                    params, eval_vms[i], H, W, intr, antialiased=args.antialiased
                )[0],
                eval_imgs[i],
            )
            for i in idxs
        ]

    eval_idxs = list(range(min(args.n_eval, len(eval_imgs))))

    # --- Batched training steps + sqrt(batch) LR scaling (survey T6, gsplat) -----
    # gsplat trains ``batch_size`` views/step (loss averaged over the batch) and scales
    # every LR by sqrt(batch·world_size); its ``steps_scaler`` compresses the schedules
    # so a batch-B run does 1/B the steps. We follow the same contract: the caller sets
    # ``--steps`` to the reduced count (steps ≈ total_view_visits / B) and we (a) scale
    # all Adam LRs -- incl. the means-schedule base -- by sqrt(B), and (b) divide the
    # per-STEP cadence knobs (relocate_every, refine_start) by B so relocation/refine
    # fire at the same *view-visit* as B=1. ``refine_stop`` and the means-decay length
    # both derive from ``--steps`` (already reduced), so they need no extra scaling.
    # At B=1 sqrt=1 and //1 are no-ops -> the default recipe is unchanged.
    B = args.batch_size
    lr_scale = float(np.sqrt(B))
    relocate_every = (
        max(1, round(args.relocate_every / B)) if args.relocate_every else 0
    )
    refine_start = round(args.refine_start / B)
    refine_stop = args.refine_stop  # already 0.9*steps in reduced-step units
    noise_stop_iter = (
        round(args.noise_stop_iter / B)
        if args.noise_stop_iter > 0
        else args.noise_stop_iter
    )
    if B > 1:
        print(
            f"batched training: B={B} views/step, {args.steps} steps "
            f"({args.steps * B} view-visits), LRs x{lr_scale:.3f} (sqrt B); "
            f"relocate_every {relocate_every}, refine [{refine_start}, {refine_stop}]"
        )

    means_sched = optax.exponential_decay(args.means_lr * lr_scale, args.steps, 0.01)
    txs: dict[Hashable, optax.GradientTransformation] = {
        "means": optax.adam(means_sched),
        "log_scales": optax.adam(args.scales_lr * lr_scale),
        "quats": optax.adam(args.quats_lr * lr_scale),
        "colors_logit": optax.adam(args.colors_lr * lr_scale),
        "opac_logit": optax.adam(args.opac_lr * lr_scale),
    }
    opt = optax.multi_transform(txs, {k: k for k in params})
    opt_state = opt.init(params)
    binoms = splax.mcmc.make_binoms(51)

    @jax.jit
    def relocate(
        p: dict[str, jax.Array], opt_state: optax.OptState, key: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        new, reset = splax.mcmc.relocate(
            key,
            p["means"],
            p["log_scales"],
            p["quats"],
            p["colors_logit"],
            p["opac_logit"],
            binoms,
            min_opacity=args.min_opacity,
        )
        return new, _reset_opt_state(opt_state, reset)

    @jax.jit
    def add_noise(
        p: dict[str, jax.Array], key: jax.Array, scaler: float
    ) -> dict[str, jax.Array]:
        m = splax.mcmc.inject_noise(
            key, p["means"], p["log_scales"], p["quats"], p["opac_logit"], scaler
        )
        return {**p, "means": m}

    # Per-image affine exposure correction (off by default; see init_exposure).
    # Own optax param group with its own LR (default 1e-3, matching gsplat's
    # AppearanceOptModule appearance_opt_lr -- the survey gives no affine-specific
    # figure, so we mirror gsplat's per-image appearance LR).
    exp_tx = None
    exp_params: jax.Array | None = None
    exp_state: optax.OptState | None = None
    if args.exposure_opt:
        exp_params = init_exposure(ntr)
        exp_tx = optax.adam(args.exposure_lr * lr_scale)  # sqrt(B) scaled too (T6)
        exp_state = exp_tx.init(exp_params)
        print(
            f"exposure-opt ON: {ntr} per-image affine transforms "
            f"(identity-init, lr {args.exposure_lr * lr_scale:g}); eval uses raw render"
        )
    step_fn = _make_step(
        opt,
        H,
        W,
        intr,
        args.ssim_lambda,
        args.opacity_reg,
        args.scale_reg,
        antialiased=args.antialiased,
        depth_loss=args.depth_loss,
        depth_lambda=args.depth_lambda,
        exp_tx=exp_tx,
        batch=B,
    )

    p0 = float(np.mean(eval_psnr(eval_idxs)))
    print(f"point-init eval PSNR: {p0:.2f} dB")
    curve = [{"step": 0, "visits": 0, "eval_psnr": round(p0, 3)}]

    key = jax.random.key(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(ntr)
    white = jnp.ones(3)
    t0 = time.perf_counter()
    for it in range(1, args.steps + 1):
        # B consecutive view-visits per step. At B=1, pos = it -> identical to the
        # pre-T6 ``order[it % ntr]`` sequence (default path unchanged).
        vis = [int(order[((it - 1) * B + 1 + j) % ntr]) for j in range(B)]
        vidx = np.asarray(vis)
        gt = jnp.asarray(train_imgs[vidx])  # (B, H, W, 3)
        vm = train_vms[jnp.asarray(vidx)]  # (B, 4, 4)
        if args.random_bkgd:
            keys = jax.random.split(key, B + 1)  # B independent bg draws (T6)
            key = keys[0]
            bg = jax.vmap(lambda k: jax.random.uniform(k, (3,)))(keys[1:])
        else:
            bg = jnp.broadcast_to(white, (B, 3))
        pt_args = (
            jnp.asarray(tp_uv[vidx]),
            jnp.asarray(tp_depth[vidx]),
            jnp.asarray(tp_mask[vidx]),
        )
        if exp_tx is not None:
            params, opt_state, exp_params, exp_state, l1 = step_fn(
                params,
                opt_state,
                exp_params,
                exp_state,
                gt,
                vm,
                bg,
                jnp.asarray(vidx, jnp.int32),
                *pt_args,
            )
        else:
            params, opt_state, l1 = step_fn(params, opt_state, gt, vm, bg, *pt_args)

        if (
            relocate_every
            and refine_start < it < refine_stop
            and it % relocate_every == 0
        ):
            key, sk = jax.random.split(key)
            params, opt_state = relocate(params, opt_state, sk)
        if (
            args.noise_lr > 0
            and it < args.steps
            and (noise_stop_iter < 0 or it < noise_stop_iter)
        ):
            scaler = float(jnp.asarray(means_sched(it))) * args.noise_lr
            key, sk = jax.random.split(key)
            params = add_noise(params, sk, scaler)

        if it % args.log_every == 0 or it == args.steps:
            l1.block_until_ready()
            ep = float(np.mean(eval_psnr(eval_idxs)))
            curve.append(
                {
                    "step": it,
                    "visits": it * B,
                    "eval_psnr": round(ep, 3),
                    "train_l1": round(float(l1), 5),
                }
            )
            print(f"step {it:5d}  train L1 {float(l1):.4f}  eval PSNR {ep:5.2f} dB")
    wall = time.perf_counter() - t0

    per_frame = eval_psnr(eval_idxs)
    ep_final = float(np.mean(per_frame))
    print(
        f"\nfinal held-out PSNR: {ep_final:.2f} dB  {[round(x, 2) for x in per_frame]}"
    )
    print(
        f"{args.steps} steps / {args.n} gaussians in {wall:.1f}s "
        f"({wall / args.steps * 1000:.1f} ms/step)"
    )

    # side-by-side render|GT for a few eval views
    Path("results").mkdir(exist_ok=True)
    for j, i in enumerate(eval_idxs):
        r = np.clip(
            np.asarray(
                render_params(
                    params, eval_vms[i], H, W, intr, antialiased=args.antialiased
                )[0]
            ),
            0,
            1,
        )
        iio.imwrite(
            f"results/drone_view{j}.png",
            (np.concatenate([r, eval_imgs[i]], 1) * 255).astype(np.uint8),
        )
        print(
            f"  results/drone_view{j}.png  ({scene['eval_names'][i]})  {per_frame[j]:.2f} dB"
        )

    if args.out_ply:
        save_ply(args.out_ply, params)
    if args.plot:
        _plot_curve(curve, wall, ep_final)
    result = {
        "per_frame": per_frame,
        "names": [scene["eval_names"][i] for i in eval_idxs],
        "final": ep_final,
        "wall": wall,
        "curve": curve,
        "batch": B,
        "steps": args.steps,
        "n": args.n,
        "view_visits": args.steps * B,
        "views_per_s": round(args.steps * B / wall, 1),
        "ms_per_step": round(wall / args.steps * 1000, 3),
        "depth_loss": bool(args.depth_loss),
        "depth_lambda": args.depth_lambda,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
    return result


def _plot_curve(curve: list[dict], wall: float, final: float) -> None:
    steps = [c["step"] for c in curve]
    ps = [c["eval_psnr"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, ps, "-o", ms=3, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out PSNR (dB)")
    ax.set_title(f"drone MCMC fit: {final:.2f} dB in {wall:.0f}s")
    ax.grid(alpha=0.3)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig("reports/figures/drone_fit_curve.png", dpi=130)
    print("wrote reports/figures/drone_fit_curve.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/drone",
        help="COLMAP scene dir (has sparse/<i>, images/)",
    )
    ap.add_argument(
        "--sparse-model",
        type=int,
        default=0,
        help="COLMAP sub-model index under sparse/ (largest is not always 0)",
    )
    ap.add_argument("--out-ply", default="data/scenes/drone.ply")
    ap.add_argument("--downscale", type=int, default=4, help="image downscale factor")
    ap.add_argument(
        "--eval-every", type=int, default=8, help="hold out every Nth image"
    )
    ap.add_argument(
        "--n-eval", type=int, default=3, help="held-out views scored/rendered"
    )
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="views per training step (gsplat batch_size, survey T6). "
        "Loss is averaged over the batch; all LRs are scaled by "
        "sqrt(batch) and the per-step MCMC cadence by 1/batch "
        "(steps_scaler). Set --steps to total_view_visits/batch. "
        "B=1 (default) is numerically identical to the pre-T6 path.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--antialiased",
        action="store_true",
        help="Mip-Splatting opacity compensation (gsplat rasterize_mode=antialiased)",
    )
    ap.add_argument("--log-every", type=int, default=200)
    # 6d MCMC recipe defaults (transferred via scene normalization)
    ap.add_argument("--means-lr", type=float, default=1.5e-3)
    ap.add_argument("--scales-lr", type=float, default=5e-3)
    ap.add_argument("--quats-lr", type=float, default=1e-3)
    ap.add_argument("--colors-lr", type=float, default=1e-2)
    ap.add_argument("--opac-lr", type=float, default=5e-2)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--noise-lr", type=float, default=5e5)
    ap.add_argument(
        "--noise-stop-iter",
        type=int,
        default=-1,
        help="stop MCMC noise injection after this step (-1=never, gsplat default)",
    )
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--relocate-every", type=int, default=100)
    ap.add_argument("--refine-start", type=int, default=200)
    ap.add_argument("--refine-stop", type=int, default=None, help="default 0.9*steps")
    ap.add_argument("--init-opa", type=float, default=0.1)
    ap.add_argument(
        "--random-bkgd",
        action="store_true",
        help="random per-step render-side bg color (gsplat random_bkgd). "
        "CAVEAT: COLMAP photos carry no alpha, so only the render is "
        "recomposited -- the fixed real GT photo is not. Off by default; "
        "see reports/phase8d_random_bkgd.md.",
    )
    ap.add_argument(
        "--depth-loss",
        action="store_true",
        help="COLMAP sparse-point depth regularization (gsplat depth_loss, "
        "survey T2): scale-normalized masked L1 between the rendered "
        "expected-depth channel and the sparse points' camera depths. "
        "Off by default; off-path is bit-identical. See "
        "reports/phase8g_depth_reg.md.",
    )
    ap.add_argument(
        "--depth-lambda",
        type=float,
        default=1e-2,
        help="depth-loss weight (gsplat default 1e-2)",
    )
    ap.add_argument(
        "--max-depth-pts",
        type=int,
        default=2048,
        help="fixed max COLMAP sparse points per view for depth reg "
        "(static shape; the rest are masked)",
    )
    ap.add_argument("--out-json", default=None, help="dump the result dict as JSON")
    # T5 affine variant: per-training-image 3x4 color transform on the render
    # before the loss (identity-init, own LR). OFF by default -> off-path is
    # bit-identical to the pre-T5 trainer. Eval always scores the raw render.
    ap.add_argument(
        "--exposure-opt",
        action="store_true",
        help="learn a per-training-image affine color correction (T5)",
    )
    ap.add_argument(
        "--exposure-lr",
        type=float,
        default=1e-3,
        help="LR for the exposure affine params (gsplat appearance_opt_lr)",
    )
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()
    if args.refine_stop is None:
        args.refine_stop = int(0.9 * args.steps)
    train(args)


if __name__ == "__main__":
    main()
