"""splax.distillation: teacher-student splat compression by synthetic re-rendering.

Radiance-field knowledge distillation (Distilled-3DGS et al.): a large trained
splat (the *teacher*) is rendered from many synthetic in-scene poses, and a much
smaller *student* is retrained on that dense synthetic dataset. Dense synthetic
coverage suppresses the view-dependent floaters/spikes that sparse real-view
training produces, so a heavily capped student can fit cleanly.

Three public entry points, all built on the existing splax machinery (no new Warp
kernels):

- ``sample_poses`` synthesizes world-to-camera ``viewmat`` s *inside* the scene
  volume (a shrunk opacity-weighted bounding box) that look at opacity-weighted
  gaussian positions, optionally biased toward a supplied camera trajectory (the
  original COLMAP training poses, whose regions the teacher supports best).
- ``render_views`` renders the teacher from those poses with the grad-free
  inference path (``splax.inference.render``), returning host-side ``uint8``
  frames (+ optional expected-depth maps via ``splax.render(render_depth=True)``)
  so ~500-1000 views never sit on device as float32.
- ``distill`` re-fits an ``n_student``-gaussian student on the synthetic frames
  with the phase-6d MCMC recipe (L1 + D-SSIM, per-group Adam, MCMC relocation +
  noise, opacity/scale regs -- the same recipe ``scripts/train_colmap.py`` fits
  with), plus optional dense depth distillation against the teacher depth maps.

The student parameters are in *render space* (the tensors ``splax.render`` and
``splax.write_ply`` consume) both on input (``teacher``) and output.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Mapping
from typing import cast

import dm_pix
import numpy as np
import jax
import jax.numpy as jnp
import optax

import splax


# --------------------------------------------------------------------------- #
# teacher/params plumbing
# --------------------------------------------------------------------------- #
_KEYS = ("means", "scales", "quats", "colors", "opacities")


def _as_teacher(
    teacher: Mapping[str, jax.Array | np.ndarray] | tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a teacher (dict or 5-tuple) to (means, scales, quats, colors, opac)."""
    if isinstance(teacher, dict):
        t = tuple(np.asarray(teacher[k]) for k in _KEYS)
    else:
        t = tuple(np.asarray(x) for x in teacher)
    means, scales, quats, colors, opac = t
    return means, scales, quats, colors, opac.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# pose sampling
# --------------------------------------------------------------------------- #
def _weighted_pct(vals: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted percentile ``q`` (0-100) of 1-D ``vals`` with weights ``w``."""
    idx = np.argsort(vals)
    v = vals[idx]
    cw = np.cumsum(w[idx])
    cw = cw / cw[-1]
    return float(np.interp(q / 100.0, cw, v))


def _opacity_bbox(
    means: np.ndarray, w: np.ndarray, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis (lo, hi) weighted-percentile bounds of the gaussian positions."""
    los = np.array([_weighted_pct(means[:, a], w, lo) for a in range(3)])
    his = np.array([_weighted_pct(means[:, a], w, hi) for a in range(3)])
    return los, his


def _lookat_viewmats(
    eyes: np.ndarray, targets: np.ndarray, up: tuple[float, float, float] | np.ndarray
) -> np.ndarray:
    """Build (n,4,4) world-to-camera matrices (OpenCV +z-forward) from eye/target.

    Rows of R are the camera axes in world coords (x right, y down, z forward);
    t = -R @ eye. Roll is arbitrary (an ``up`` hint that flips to an alternate
    when parallel to the view direction) -- it does not affect distillation, which
    renders teacher and student through the *same* matrices.
    """
    up = np.asarray(up, np.float64)
    n = eyes.shape[0]
    z = targets - eyes
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    upv = np.broadcast_to(up, (n, 3)).astype(np.float64).copy()
    # swap in an alternate up where it is (nearly) parallel to the view direction
    par = np.abs(z @ up) > 0.99
    upv[par] = (
        np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    x = np.cross(upv, z)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # (n,3,3), rows = cam axes
    t = -np.einsum("nij,nj->ni", R, eyes)
    vm = np.broadcast_to(np.eye(4, dtype=np.float32), (n, 4, 4)).copy()
    vm[:, :3, :3] = R.astype(np.float32)
    vm[:, :3, 3] = t.astype(np.float32)
    return vm


def _cam_centers(viewmats: np.ndarray) -> np.ndarray:
    """Camera centers C = -R^T t from world-to-camera matrices (n,4,4)."""
    vms = np.asarray(viewmats)
    R = vms[:, :3, :3]
    t = vms[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def sample_poses(
    means: np.ndarray,
    opacities: np.ndarray,
    n_views: int,
    seed: int = 0,
    viewmats: np.ndarray | None = None,
    bbox_lo: float = 20.0,
    bbox_hi: float = 80.0,
    min_dist: float = 0.2,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    traj_frac: float = 0.5,
) -> np.ndarray:
    """Synthesize ``n_views`` in-scene world-to-camera matrices (n,4,4).

    Camera positions are drawn uniformly inside the *inner* opacity-weighted
    bounding box of the gaussians (``bbox_lo``..``bbox_hi`` weighted percentile per
    axis), so every view stays within the splat volume and never sees its warped
    outside. Look-at targets are opacity-weighted samples of gaussian positions,
    pushed out to at least ``min_dist`` from the camera.

    If ``viewmats`` (e.g. the original COLMAP training cameras) is given, a
    ``traj_frac`` fraction of the eye positions are drawn by jittered interpolation
    between random pairs of those cameras' centers instead of from the bbox, biasing
    coverage toward the trajectory the teacher supports best (the rest still come
    from the bbox for broad in-volume coverage).
    """
    means = np.asarray(means, np.float64)
    opac = np.asarray(opacities, np.float64).reshape(-1)
    w = opac / (opac.sum() + 1e-12)
    rng = np.random.default_rng(seed)

    los, his = _opacity_bbox(means, w, bbox_lo, bbox_hi)
    eyes = rng.uniform(los, his, size=(n_views, 3))

    if viewmats is not None:
        centers = _cam_centers(viewmats)
        n_traj = int(round(n_views * traj_frac))
        if n_traj > 0 and centers.shape[0] >= 1:
            a = rng.integers(0, centers.shape[0], n_traj)
            b = rng.integers(0, centers.shape[0], n_traj)
            alpha = rng.uniform(0.0, 1.0, (n_traj, 1))
            interp = centers[a] * (1 - alpha) + centers[b] * alpha
            extent = np.linalg.norm(his - los)
            jitter = rng.normal(size=(n_traj, 3)) * 0.05 * extent
            sel = rng.choice(n_views, n_traj, replace=False)
            eyes[sel] = interp + jitter

    # opacity-weighted look-at targets, pushed to >= min_dist
    tgt_idx = rng.choice(means.shape[0], n_views, p=w)
    targets = means[tgt_idx]
    d = targets - eyes
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    close = dist[:, 0] < min_dist
    if close.any():
        # push the target out along the view dir (or a random dir if degenerate)
        dir_ = d[close] / (dist[close] + 1e-12)
        degen = dist[close, 0] < 1e-9
        if degen.any():
            r = rng.normal(size=(int(degen.sum()), 3))
            dir_[degen] = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
        targets[close] = eyes[close] + dir_ * min_dist

    return _lookat_viewmats(eyes, targets, up)


# --------------------------------------------------------------------------- #
# teacher rendering
# --------------------------------------------------------------------------- #
def render_views(
    teacher: Mapping[str, jax.Array | np.ndarray] | tuple,
    viewmats: np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    depth: bool = False,
    background: np.ndarray | None = None,
    block_size: int = 16,
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Render the teacher from ``viewmats`` (n,4,4), host-side, memory-safe.

    Returns ``(images, depths)``: a ``uint8`` array ``(n, H, W, 3)`` of frames,
    plus -- with ``depth=True`` -- a ``float16`` ``(n, H, W)`` expected-depth stack
    (teacher depth targets for depth distillation), ``None`` otherwise. Renders one
    view at a time with the grad-free inference path (depth uses
    ``splax.render(render_depth=True)``, whose forward is identical), so 500-1000
    views never hold float32 on device.
    """
    means, scales, quats, colors, opac = _as_teacher(teacher)
    means = jnp.asarray(means, jnp.float32)
    scales = jnp.asarray(scales, jnp.float32)
    quats = jnp.asarray(quats, jnp.float32)
    colors = jnp.asarray(colors, jnp.float32)
    opac = jnp.asarray(opac, jnp.float32)
    H, W = img_shape
    bg = jnp.ones(3) if background is None else jnp.asarray(background, jnp.float32)
    vms = np.asarray(viewmats, np.float32)
    n = vms.shape[0]

    imgs = np.empty((n, H, W, 3), np.uint8)
    depths = np.empty((n, H, W), np.float16) if depth else None
    for i in range(n):
        vm = jnp.asarray(vms[i])
        if depths is not None:
            img, d = splax.render(
                means,
                scales,
                quats,
                colors,
                opac,
                viewmat=vm,
                background=bg,
                render_depth=True,
                img_shape=(H, W),
                f=f,
                c=c,
                glob_scale=glob_scale,
                clip_thresh=clip_thresh,
                block_size=block_size,
            )
            depths[i] = np.asarray(d, np.float16)
        else:
            img = splax.inference.render(
                means,
                scales,
                quats,
                colors,
                opac,
                viewmat=vm,
                background=bg,
                img_shape=(H, W),
                f=f,
                c=c,
                glob_scale=glob_scale,
                clip_thresh=clip_thresh,
                block_size=block_size,
            )
        imgs[i] = (np.clip(np.asarray(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    return imgs, depths


# --------------------------------------------------------------------------- #
# student init
# --------------------------------------------------------------------------- #
def _logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float64), eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _student_init(
    teacher: Mapping[str, jax.Array | np.ndarray] | tuple,
    n_student: int,
    init: str,
    seed: int,
    init_opa: float,
) -> dict[str, jax.Array]:
    """Build the raw (logit-space) student param dict from the teacher.

    ``prune``: the top-``n_student``-opacity teacher subset, all params kept.
    ``random``: ``n_student`` random positions inside the teacher's inner opacity
    bbox, with the teacher's median log-scale, random colors and ``init_opa``
    opacity (a from-scratch init that ignores the teacher's structure).
    """
    means, scales, quats, colors, opac = _as_teacher(teacher)
    opac1 = opac.reshape(-1)
    rng = np.random.default_rng(seed)
    if init == "prune":
        sel = np.argsort(-opac1)[:n_student]
        return {
            "means": jnp.asarray(means[sel], jnp.float32),
            "log_scales": jnp.asarray(
                np.log(np.clip(scales[sel], 1e-8, None)), jnp.float32
            ),
            "quats": jnp.asarray(quats[sel], jnp.float32),
            "colors_logit": jnp.asarray(_logit(colors[sel]), jnp.float32),
            "opac_logit": jnp.asarray(_logit(opac1[sel])[:, None], jnp.float32),
        }
    if init == "random":
        w = opac1 / (opac1.sum() + 1e-12)
        los, his = _opacity_bbox(means.astype(np.float64), w, 20.0, 80.0)
        pos = rng.uniform(los, his, size=(n_student, 3)).astype(np.float32)
        med_ls = np.median(np.log(np.clip(scales, 1e-8, None)), axis=0)
        ls = np.broadcast_to(med_ls, (n_student, 3)).astype(np.float32)
        col = rng.uniform(0.2, 0.8, size=(n_student, 3)).astype(np.float32)
        q = rng.normal(size=(n_student, 4)).astype(np.float32)
        return {
            "means": jnp.asarray(pos),
            "log_scales": jnp.asarray(ls),
            "quats": jnp.asarray(q),
            "colors_logit": jnp.asarray(_logit(col)),
            "opac_logit": jnp.full(
                (n_student, 1), float(_logit(np.array(init_opa))), jnp.float32
            ),
        }
    raise ValueError(f"unknown init {init!r} (expected 'prune' or 'random')")


def _render_student(
    p: dict[str, jax.Array],
    viewmat: jax.Array,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
    render_depth: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    fx, fy, cx, cy = intr
    means = p["means"]
    scales = jnp.exp(p["log_scales"])
    quats = p["quats"] / (jnp.linalg.norm(p["quats"], axis=-1, keepdims=True) + 1e-8)
    colors = jax.nn.sigmoid(p["colors_logit"])
    opac = jax.nn.sigmoid(p["opac_logit"])
    return splax.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(fx, fy),
        c=(cx, cy),
        glob_scale=1.0,
        clip_thresh=0.01,
        render_depth=render_depth,
    )


def _to_render_space(p: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Raw logit-space student params -> render-space dict (write_ply / render inputs)."""
    return {
        "means": p["means"],
        "scales": jnp.exp(p["log_scales"]),
        "quats": p["quats"]
        / (jnp.linalg.norm(p["quats"], axis=-1, keepdims=True) + 1e-8),
        "colors": jax.nn.sigmoid(p["colors_logit"]),
        "opacities": jax.nn.sigmoid(p["opac_logit"]),
    }


# --------------------------------------------------------------------------- #
# training step
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
    depth_lambda: float,
) -> Callable:
    """Jitted train step mirroring ``scripts/train_colmap.py``'s step builder, with
    optional *dense* depth distillation (masked, scale-normalized L1 between the
    student and teacher expected-depth maps) in place of the sparse-point depth loss.
    """
    depth_on = depth_lambda > 0.0

    def per_view(
        p: dict[str, jax.Array], gt: jax.Array, vm: jax.Array, gt_depth: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if depth_on:
            img, depth = _render_student(p, vm, H, W, intr, render_depth=True)
            assert depth is not None  # render_depth=True fills the depth slot
            mask = (gt_depth > 0.0).astype(jnp.float32)
            npx = jnp.sum(mask) + 1e-8
            scale = jnp.sum(mask * gt_depth) / npx + 1e-8  # scene-scale normalize
            dl = jnp.sum(mask * jnp.abs(depth - gt_depth)) / npx / scale
        else:
            img, _ = _render_student(p, vm, H, W, intr)
            dl = jnp.array(0.0, jnp.float32)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = jnp.asarray(1.0 - dm_pix.ssim(img, gt))
        return l1, dssim, dl

    def loss_fn(
        p: dict[str, jax.Array], gt: jax.Array, vm: jax.Array, gt_depth: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        l1s, dssims, dls = jax.vmap(per_view, in_axes=(None, 0, 0, 0))(
            p, gt, vm, gt_depth
        )
        l1 = jnp.mean(l1s)
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * jnp.mean(dssims)
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(p["log_scales"]))
        if depth_on:
            loss = loss + depth_lambda * jnp.mean(dls)
        return loss, l1

    @jax.jit
    def step(
        p: dict[str, jax.Array],
        opt_state: optax.OptState,
        gt: jax.Array,
        vm: jax.Array,
        gt_depth: jax.Array,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        (loss, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, gt, vm, gt_depth
        )
        updates, opt_state = opt.update(grads, opt_state, p)
        # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
        new_p = cast(dict[str, jax.Array], optax.apply_updates(p, updates))
        return new_p, opt_state, l1

    return step


# --------------------------------------------------------------------------- #
# distillation driver
# --------------------------------------------------------------------------- #
def distill(
    teacher: Mapping[str, jax.Array | np.ndarray] | tuple,
    n_student: int,
    *,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    n_views: int = 500,
    steps: int = 3000,
    depth_lambda: float = 0.0,
    init: str = "prune",
    seed: int = 0,
    viewmats: np.ndarray | None = None,
    batch: int = 1,
    means_lr: float = 1.5e-3,
    scales_lr: float = 5e-3,
    quats_lr: float = 1e-3,
    colors_lr: float = 1e-2,
    opac_lr: float = 5e-2,
    ssim_lambda: float = 0.2,
    opacity_reg: float = 0.01,
    scale_reg: float = 0.01,
    noise_lr: float = 5e5,
    noise_stop_iter: int = -1,
    min_opacity: float = 0.005,
    init_opa: float = 0.1,
    relocate_every: int = 100,
    refine_start: int = 200,
    refine_stop: int | None = None,
    log_every: int = 200,
    eval_hook: Callable | None = None,
    info: dict | None = None,
) -> dict[str, jax.Array]:
    """Distill ``teacher`` (render-space splat) into an ``n_student``-gaussian student.

    Renders the teacher from ``n_views`` synthetic in-scene poses (``sample_poses``,
    biased toward ``viewmats`` when given) at ``img_shape``/``f``/``c``, then fits the
    student on those frames with the phase-6d MCMC recipe. With ``depth_lambda > 0``
    the teacher expected-depth maps are rendered too and a dense masked depth L1 is
    added to the loss. Returns the student as a render-space param dict
    (``means, scales, quats, colors, opacities``). If ``info`` is a dict it is filled
    with ``curve`` / ``wall`` / ``n_views`` / ``teacher_n``.

    ``batch`` views/step averages the loss over the batch and scales all LRs by
    sqrt(batch) (gsplat ``batch_size``); at ``batch==1`` the recipe is the default.
    """
    H, W = img_shape
    intr = (f[0], f[1], c[0], c[1])
    if refine_stop is None:
        refine_stop = int(0.9 * steps)
    depth_on = depth_lambda > 0.0
    t_means, _, _, _, t_opac = _as_teacher(teacher)

    # --- synthetic dataset: sample poses, render teacher (host-side) ----------
    vms = sample_poses(t_means, t_opac, n_views, seed=seed, viewmats=viewmats)
    syn_imgs, syn_depth = render_views(teacher, vms, img_shape, f, c, depth=depth_on)

    # --- student init + optimizer (phase-6d recipe, sqrt(batch) LR scaling) ---
    params = _student_init(teacher, n_student, init, seed, init_opa)
    B = int(batch)
    lr_scale = float(np.sqrt(B))
    rel_every = max(1, round(relocate_every / B)) if relocate_every else 0
    ref_start = round(refine_start / B)
    means_sched = optax.exponential_decay(means_lr * lr_scale, steps, 0.01)
    txs: dict[Hashable, optax.GradientTransformation] = {
        "means": optax.adam(means_sched),
        "log_scales": optax.adam(scales_lr * lr_scale),
        "quats": optax.adam(quats_lr * lr_scale),
        "colors_logit": optax.adam(colors_lr * lr_scale),
        "opac_logit": optax.adam(opac_lr * lr_scale),
    }
    opt = optax.multi_transform(txs, {k: k for k in params})
    opt_state = opt.init(params)
    binoms = splax.mcmc.make_binoms(51)
    step_fn = _make_step(
        opt, H, W, intr, ssim_lambda, opacity_reg, scale_reg, depth_lambda
    )

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
            min_opacity=min_opacity,
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

    # --- training loop --------------------------------------------------------
    curve = []
    if eval_hook is not None:
        p0 = float(eval_hook(_to_render_space(params)))
        curve.append({"step": 0, "eval_psnr": round(p0, 3)})
    key = jax.random.key(seed + 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_views)
    t0 = time.perf_counter()
    for it in range(1, steps + 1):
        vis = np.asarray([order[((it - 1) * B + j) % n_views] for j in range(B)])
        gt = jnp.asarray(syn_imgs[vis].astype(np.float32) / 255.0)
        vm = jnp.asarray(vms[vis])
        if syn_depth is not None:
            gd = jnp.asarray(syn_depth[vis].astype(np.float32))
        else:
            gd = jnp.zeros((B, H, W), jnp.float32)
        params, opt_state, l1 = step_fn(params, opt_state, gt, vm, gd)

        if rel_every and ref_start < it < refine_stop and it % rel_every == 0:
            key, sk = jax.random.split(key)
            params, opt_state = relocate(params, opt_state, sk)
        if (
            noise_lr > 0
            and it < steps
            and (noise_stop_iter < 0 or it < noise_stop_iter)
        ):
            scaler = float(jnp.asarray(means_sched(it))) * noise_lr
            key, sk = jax.random.split(key)
            params = add_noise(params, sk, scaler)

        if it % log_every == 0 or it == steps:
            l1.block_until_ready()
            entry = {"step": it, "train_l1": round(float(l1), 5)}
            if eval_hook is not None:
                entry["eval_psnr"] = round(
                    float(eval_hook(_to_render_space(params))), 3
                )
            curve.append(entry)
    wall = time.perf_counter() - t0

    student = _to_render_space(params)
    if info is not None:
        info.update(
            {
                "curve": curve,
                "wall": wall,
                "n_views": n_views,
                "teacher_n": int(t_means.shape[0]),
                "n_student": int(n_student),
                "steps": steps,
                "batch": B,
                "depth_lambda": depth_lambda,
                "init": init,
            }
        )
    return student


__all__ = ["sample_poses", "render_views", "distill"]
