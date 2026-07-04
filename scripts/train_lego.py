"""Fit a fixed set of Gaussians to the lego train images with the Warp backend.

Two modes share ``splax.render``'s differentiable path and a fixed (static-shape)
gaussian budget -- no densification that grows N:

* **smoke** (default): the Phase 6a gradient sanity check. 25k random gaussians,
  per-parameter Adam, L1 loss, 1500 steps @ 400x400 -> ~24 dB on the held-out
  view. Fast (<3 min), unchanged.
* **--quality**: Phase 6d. Ports gsplat's MCMC training recipe into JAX with
  fixed shapes (splax.mcmc): per-parameter LR schedules (means exponentially
  decayed ~100x), MCMC relocation of dead gaussians + per-step covariance noise
  (Kheradmand et al. 2024), an L1+D-SSIM loss mix, and opacity/scale
  regularizers. Evaluated at the full 800x800 test resolution. Target > 26 dB.

Usage:
  python scripts/train_lego.py                        # smoke (24 dB, ~2.5 s)
  python scripts/train_lego.py --quality [flags]      # MCMC quality run
Writes results/train_lego_{before,after}.png (render|GT) and a fit JSON; with
--quality also results/lego_fit_6d_curve.* and the 800x800 side-by-sides.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import cast

import dm_pix
import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import splax

LEGO = Path("data/nerf_synthetic/lego")


def nerf_camera(frame: dict) -> np.ndarray:
    """NeRF c2w (OpenGL, -z forward) -> w2c viewmat (OpenCV, +z forward)."""
    c2w = np.array(frame["transform_matrix"], np.float64)
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w).astype(np.float32)


def load_view(frame: dict, res: int) -> tuple[jax.Array, jax.Array]:
    """Load a frame's image (composited on white) at `res`x`res` and its viewmat."""
    img = (
        iio.imread(LEGO / (frame["file_path"].lstrip("./") + ".png")).astype(np.float32)
        / 255.0
    )
    img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])  # composite on white
    H = img.shape[0]
    if H != res:
        f = H // res
        img = img.reshape(res, f, res, f, 3).mean((1, 3))  # box downsample
    return jnp.asarray(img), jnp.asarray(nerf_camera(frame))


def load_view_alpha(frame: dict, res: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Load a frame's raw (un-composited) RGB + alpha at `res`x`res` and its viewmat.

    Used only by the ``--random-bkgd`` path: keeps the straight-alpha PNG channels
    apart so the caller can composite over a different background color each step.
    """
    img = (
        iio.imread(LEGO / (frame["file_path"].lstrip("./") + ".png")).astype(np.float32)
        / 255.0
    )
    H = img.shape[0]
    if H != res:
        f = H // res
        img = img.reshape(res, f, res, f, 4).mean(
            (1, 3)
        )  # box downsample (rgba jointly)
    return (
        jnp.asarray(img[..., :3]),
        jnp.asarray(img[..., 3:]),
        jnp.asarray(nerf_camera(frame)),
    )


def init_params(n: int, seed: int = 0) -> dict[str, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    return {
        # means uniform in the lego scene bounds ([-1.3, 1.3]^3-ish)
        "means": jax.random.uniform(k[0], (n, 3), minval=-1.3, maxval=1.3),
        "log_scales": jnp.full((n, 3), jnp.log(0.03)),  # ~3 cm gaussians
        "quats": jax.random.normal(k[2], (n, 4)),
        "colors_logit": jax.random.normal(k[3], (n, 3)) * 0.1,  # ~gray
        "opac_logit": jnp.full((n, 1), -2.0),  # sigmoid(-2) ~ 0.12
    }


def init_params_mcmc(
    n: int, init_scale: float, init_opa: float, seed: int = 0
) -> dict[str, jax.Array]:
    """gsplat MCMC-preset init: half-opacity, uniform in the cube, ~knn scale."""
    k = jax.random.split(jax.random.key(seed), 5)
    return {
        "means": jax.random.uniform(k[0], (n, 3), minval=-1.3, maxval=1.3),
        "log_scales": jnp.full((n, 3), jnp.log(init_scale)),
        "quats": jax.random.normal(k[2], (n, 4)),
        "colors_logit": jax.random.normal(k[3], (n, 3)) * 0.1,
        "opac_logit": jnp.full((n, 1), float(np.log(init_opa / (1 - init_opa)))),
    }


def render_params(
    p: dict[str, jax.Array],
    viewmat: jax.Array,
    res: int,
    f: float,
    background: jax.Array | None = None,
    antialiased: bool = False,
) -> jax.Array:
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
        img_shape=(res, res),
        f=(f, f),
        c=(res // 2, res // 2),
        glob_scale=1.0,
        clip_thresh=0.01,
        antialiased=antialiased,
    )[0]


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
# smoke mode (Phase 6a, unchanged)                                            #
# --------------------------------------------------------------------------- #
def run_smoke(args: argparse.Namespace) -> None:
    train_meta = json.loads((LEGO / "transforms_train.json").read_text())
    test_meta = json.loads((LEGO / "transforms_test.json").read_text())
    res = args.res
    f = 0.5 * res / np.tan(0.5 * train_meta["camera_angle_x"])

    print(f"loading {len(train_meta['frames'])} train views at {res}x{res} ...")
    train_imgs, train_vms = zip(*[load_view(fr, res) for fr in train_meta["frames"]])
    train_imgs = jnp.stack(train_imgs)
    train_vms = jnp.stack(train_vms)
    test_img, test_vm = load_view(test_meta["frames"][0], res)  # held-out view

    params = init_params(args.n, args.seed)

    # per-parameter Adam (real 3DGS uses distinct LRs; scaled up for a short run).
    lrs = {
        "means": 2e-3,
        "log_scales": 5e-3,
        "quats": 1e-3,
        "colors_logit": 1e-2,
        "opac_logit": 3e-2,
    }
    opt = optax.multi_transform(
        {k: optax.adam(v) for k, v in lrs.items()}, {k: k for k in params}
    )
    opt_state = opt.init(params)

    def loss_fn(p: dict[str, jax.Array], gt: jax.Array, vm: jax.Array) -> jax.Array:
        img = render_params(p, vm, res, f, antialiased=args.antialiased)
        return jnp.mean(jnp.abs(img - gt))  # L1 photometric loss

    @jax.jit
    def step(
        p: dict[str, jax.Array], opt_state: optax.OptState, gt: jax.Array, vm: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(p, gt, vm)
        updates, opt_state = opt.update(grads, opt_state, p)
        # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
        return (
            cast(dict[str, jax.Array], optax.apply_updates(p, updates)),
            opt_state,
            loss,
        )

    aa = args.antialiased
    before = np.clip(
        np.asarray(render_params(params, test_vm, res, f, antialiased=aa)), 0, 1
    )
    p0 = psnr(before, test_img)
    print(f"random-init test PSNR: {p0:.2f} dB")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_imgs))
    traj = [{"step": 0, "test_psnr": round(p0, 3)}]
    t0 = time.perf_counter()
    for it in range(1, args.steps + 1):
        vi = int(order[it % len(order)])
        params, opt_state, loss = step(params, opt_state, train_imgs[vi], train_vms[vi])
        if it % 100 == 0 or it == args.steps:
            loss.block_until_ready()
            tp = psnr(render_params(params, test_vm, res, f, antialiased=aa), test_img)
            traj.append(
                {
                    "step": it,
                    "test_psnr": round(tp, 3),
                    "train_l1": round(float(loss), 5),
                }
            )
            print(f"step {it:5d}  train L1 {float(loss):.4f}  test PSNR {tp:5.2f} dB")
    wall = time.perf_counter() - t0

    after = np.clip(
        np.asarray(render_params(params, test_vm, res, f, antialiased=aa)), 0, 1
    )
    p_final = psnr(after, test_img)
    print(
        f"\nfinal test PSNR: {p_final:.2f} dB  ({args.steps} steps in {wall:.1f}s, "
        f"{wall / args.steps * 1000:.1f} ms/step)"
    )

    Path("results").mkdir(exist_ok=True)
    gt = np.asarray(test_img)
    iio.imwrite(
        "results/train_lego_before.png",
        (np.concatenate([before, gt], 1) * 255).astype(np.uint8),
    )
    iio.imwrite(
        "results/train_lego_after.png",
        (np.concatenate([after, gt], 1) * 255).astype(np.uint8),
    )
    out = {
        "n": args.n,
        "res": res,
        "steps": args.steps,
        "wall_s": round(wall, 2),
        "ms_per_step": round(wall / args.steps * 1000, 3),
        "init_test_psnr": round(p0, 3),
        "final_test_psnr": round(p_final, 3),
        "trajectory": traj,
    }
    Path("results/phase6a_train_fit.json").write_text(json.dumps(out, indent=2))
    print("wrote results/train_lego_{before,after}.png, results/phase6a_train_fit.json")

    if args.out_ply:
        save_ply(args.out_ply, params)


# --------------------------------------------------------------------------- #
# quality mode (Phase 6d, MCMC recipe ported from gsplat)                      #
# --------------------------------------------------------------------------- #
def _make_step(
    opt: optax.GradientTransformation,
    res: int,
    f: float,
    ssim_lambda: float,
    opacity_reg: float,
    scale_reg: float,
    antialiased: bool = False,
) -> Callable:
    """Build a jitted train step for a given resolution / loss weighting.

    ``gt_rgb``/``gt_alpha``/``bg`` implement gsplat's ``random_bkgd``: the render
    and the GT are composited over the *same* background each step (``gt =
    alpha*gt_rgb + (1-alpha)*bg``). When random background is disabled the caller
    passes ``gt_alpha=1`` (a pre-composited-on-white image) and a fixed white
    ``bg``, which reduces the formula to ``gt = gt_rgb`` -- bit-identical to the
    pre-T1 code path.
    """

    def loss_fn(
        p: dict[str, jax.Array],
        gt_rgb: jax.Array,
        gt_alpha: jax.Array,
        bg: jax.Array,
        vm: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        img = render_params(p, vm, res, f, background=bg, antialiased=antialiased)
        gt = gt_alpha * gt_rgb + (1.0 - gt_alpha) * bg
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = 1.0 - dm_pix.ssim(img, gt)  # D-SSIM
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * dssim
        # MCMC regularizers (paper Eq.: encourage low opacity / small scale)
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(p["log_scales"]))
        return loss, l1

    @jax.jit
    def step(
        p: dict[str, jax.Array],
        opt_state: optax.OptState,
        gt_rgb: jax.Array,
        gt_alpha: jax.Array,
        bg: jax.Array,
        vm: jax.Array,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        (loss, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, gt_rgb, gt_alpha, bg, vm
        )
        updates, opt_state = opt.update(grads, opt_state, p)
        # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
        return (
            cast(dict[str, jax.Array], optax.apply_updates(p, updates)),
            opt_state,
            l1,
        )

    return step


def _reset_opt_state(
    opt_state: optax.OptState, reset_mask: jax.Array
) -> optax.OptState:
    """Zero Adam moments (leading-dim == N leaves) at relocated rows."""
    n = reset_mask.shape[0]
    keep = (~reset_mask).astype(jnp.float32)

    def z(x: jax.Array) -> jax.Array:
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == n:
            return x * keep.reshape((-1,) + (1,) * (x.ndim - 1))
        return x

    return jax.tree.map(z, opt_state)


def run_quality(args: argparse.Namespace) -> dict:
    train_meta = json.loads((LEGO / "transforms_train.json").read_text())
    test_meta = json.loads((LEGO / "transforms_test.json").read_text())
    cax = train_meta["camera_angle_x"]

    # phase schedule: train at args.res, optionally fine-tune the tail at res_ft
    res_ft = args.res_ft or args.res
    ft_start = (
        int(args.steps * (1.0 - args.ft_frac)) if res_ft != args.res else args.steps
    )

    def frames_at(res: int) -> tuple[jax.Array, jax.Array]:
        """imgs pre-composited on white, vms; used when --random-bkgd is off."""
        imgs, vms = zip(*[load_view(fr, res) for fr in train_meta["frames"]])
        return jnp.stack(imgs), jnp.stack(vms)

    def frames_at_alpha(res: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        """raw (rgb, alpha) + vms; used to re-composite over a random bg each step."""
        rgb, alpha, vms = zip(
            *[load_view_alpha(fr, res) for fr in train_meta["frames"]]
        )
        return jnp.stack(rgb), jnp.stack(alpha), jnp.stack(vms)

    print(
        f"loading {len(train_meta['frames'])} train views "
        f"(random_bkgd={args.random_bkgd}) ..."
    )
    if args.random_bkgd:
        train = {args.res: frames_at_alpha(args.res)}
        if res_ft != args.res:
            train[res_ft] = frames_at_alpha(res_ft)
    else:
        # gt_alpha=1 (constant, one copy per res, not per-view) makes the
        # random-bkgd compositing formula reduce exactly to gt=rgb below.
        ones_alpha = {r: jnp.ones((r, r, 1), jnp.float32) for r in {args.res, res_ft}}
        train = {}
        imgs, vms = frames_at(args.res)
        train[args.res] = (imgs, ones_alpha[args.res], vms)
        if res_ft != args.res:
            imgs, vms = frames_at(res_ft)
            train[res_ft] = (imgs, ones_alpha[res_ft], vms)

    # held-out eval at full 800x800 on a few test frames
    eval_res = args.eval_res
    eval_views = [load_view(test_meta["frames"][i], eval_res) for i in args.eval_frames]
    eval_imgs = [np.asarray(im) for im, _ in eval_views]
    eval_vms = [vm for _, vm in eval_views]
    f_eval = 0.5 * eval_res / np.tan(0.5 * cax)

    def eval_psnr() -> tuple[float, list[float]]:
        ps = [
            psnr(
                render_params(
                    params, vm, eval_res, f_eval, antialiased=args.antialiased
                ),
                gt,
            )
            for gt, vm in zip(eval_imgs, eval_vms)
        ]
        return float(np.mean(ps)), ps

    params = init_params_mcmc(args.n, args.init_scale, args.init_opa, args.seed)

    # per-parameter LRs: means exponentially decayed ~100x, rest constant (gsplat).
    means_sched = optax.exponential_decay(args.means_lr, args.steps, 0.01)
    txs: dict[Hashable, optax.GradientTransformation] = {
        "means": optax.adam(means_sched),
        "log_scales": optax.adam(args.scales_lr),
        "quats": optax.adam(args.quats_lr),
        "colors_logit": optax.adam(args.colors_lr),
        "opac_logit": optax.adam(args.opac_lr),
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

    # build steppers for each resolution phase
    steppers = {
        r: _make_step(
            opt,
            r,
            0.5 * r / np.tan(0.5 * cax),
            args.ssim_lambda,
            args.opacity_reg,
            args.scale_reg,
            antialiased=args.antialiased,
        )
        for r in train
    }

    p0, _ = eval_psnr()
    print(f"random-init eval PSNR (800x800): {p0:.2f} dB")
    curve = [{"step": 0, "eval_psnr": round(p0, 3), "res": None}]

    key = jax.random.key(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_meta["frames"]))
    white = jnp.ones(3)
    t0 = time.perf_counter()
    for it in range(1, args.steps + 1):
        res = args.res if it < ft_start else res_ft
        imgs, gt_alpha, vms = train[res]
        vi = int(order[it % len(order)])
        if args.random_bkgd:
            key, bk = jax.random.split(key)
            bg = jax.random.uniform(bk, (3,))
            alpha = gt_alpha[vi]
        else:
            bg = white
            alpha = gt_alpha  # single (res,res,1) ones template, shared across views
        params, opt_state, l1 = steppers[res](
            params, opt_state, imgs[vi], alpha, bg, vms[vi]
        )

        # MCMC relocation (every refine_every steps, in a refine window)
        if (
            args.relocate_every
            and args.refine_start < it < args.refine_stop
            and it % args.relocate_every == 0
        ):
            key, sk = jax.random.split(key)
            params, opt_state = relocate(params, opt_state, sk)

        # MCMC noise injection every step, annealed by the means LR; stops for good
        # after args.noise_stop_iter (gsplat's `noise_injection_stop_iter`, -1=never).
        if (
            args.noise_lr > 0
            and it < args.steps
            and (args.noise_stop_iter < 0 or it < args.noise_stop_iter)
        ):
            scaler = float(jnp.asarray(means_sched(it))) * args.noise_lr
            key, sk = jax.random.split(key)
            params = add_noise(params, sk, scaler)

        if it % args.eval_every == 0 or it == args.steps:
            l1.block_until_ready()
            ep, _ = eval_psnr()
            curve.append(
                {
                    "step": it,
                    "eval_psnr": round(ep, 3),
                    "train_l1": round(float(l1), 5),
                    "res": res,
                }
            )
            print(
                f"step {it:5d}  res {res}  train L1 {float(l1):.4f}  eval PSNR {ep:5.2f} dB"
            )
    wall = time.perf_counter() - t0

    ep_final, per_frame = eval_psnr()
    print(
        f"\nfinal eval PSNR (800x800, frames {args.eval_frames}): {ep_final:.2f} dB "
        f"{[round(x, 2) for x in per_frame]}"
    )
    print(
        f"{args.steps} steps / {args.n} gaussians in {wall:.1f}s "
        f"({wall / args.steps * 1000:.1f} ms/step)"
    )

    Path("results").mkdir(exist_ok=True)
    # side-by-side render|GT for each eval frame at 800x800
    for fi, (gt, vm) in zip(args.eval_frames, zip(eval_imgs, eval_vms)):
        r = np.clip(
            np.asarray(
                render_params(
                    params, vm, eval_res, f_eval, antialiased=args.antialiased
                )
            ),
            0,
            1,
        )
        iio.imwrite(
            f"results/lego_fit_6d_f{fi}.png",
            (np.concatenate([r, gt], 1) * 255).astype(np.uint8),
        )
    # keep the smoke output names populated too (frame 0)
    r0 = np.clip(
        np.asarray(
            render_params(
                params, eval_vms[0], eval_res, f_eval, antialiased=args.antialiased
            )
        ),
        0,
        1,
    )
    iio.imwrite(
        "results/train_lego_after.png",
        (np.concatenate([r0, eval_imgs[0]], 1) * 255).astype(np.uint8),
    )

    out = {
        "mode": "quality",
        "n": args.n,
        "steps": args.steps,
        "res": args.res,
        "res_ft": res_ft,
        "ft_start": ft_start,
        "eval_res": eval_res,
        "wall_s": round(wall, 2),
        "ms_per_step": round(wall / args.steps * 1000, 3),
        "init_eval_psnr": round(p0, 3),
        "final_eval_psnr": round(ep_final, 3),
        "per_frame_psnr": {
            fi: round(x, 3) for fi, x in zip(args.eval_frames, per_frame)
        },
        "hparams": {
            k: getattr(args, k)
            for k in (
                "means_lr",
                "scales_lr",
                "quats_lr",
                "colors_lr",
                "opac_lr",
                "ssim_lambda",
                "opacity_reg",
                "scale_reg",
                "noise_lr",
                "min_opacity",
                "relocate_every",
                "refine_start",
                "refine_stop",
                "init_scale",
                "init_opa",
                "random_bkgd",
                "noise_stop_iter",
            )
        },
        "curve": curve,
    }
    Path("results/phase6d_train_fit.json").write_text(json.dumps(out, indent=2))
    print("wrote results/phase6d_train_fit.json, results/lego_fit_6d_f*.png")

    if args.out_ply:
        save_ply(args.out_ply, params)

    if args.plot:
        _plot_curve(curve, ft_start if res_ft != args.res else None, wall, ep_final)
    return out


def _plot_curve(
    curve: list[dict], ft_start: int | None, wall: float, final: float
) -> None:
    steps = [c["step"] for c in curve]
    ps = [c["eval_psnr"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, ps, "-o", ms=3, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out PSNR @ 800x800 (dB)")
    ax.axhline(26.0, ls="--", color="0.6", lw=1, label="6d target 26 dB")
    if ft_start:
        ax.axvline(ft_start, ls=":", color="C3", lw=1, label="800x800 fine-tune")
    ax.set_title(f"lego MCMC fit: {final:.2f} dB in {wall:.0f}s")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig("reports/figures/phase6d_curve.png", dpi=130)
    print("wrote reports/figures/phase6d_curve.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quality", action="store_true", help="run the Phase 6d MCMC recipe"
    )
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--antialiased",
        action="store_true",
        help="Mip-Splatting opacity compensation (gsplat rasterize_mode=antialiased)",
    )
    ap.add_argument(
        "--out-ply", help="optional path to save the fitted splats as a 3DGS .ply"
    )
    # quality-only knobs (gsplat-derived defaults)
    ap.add_argument(
        "--res-ft", type=int, default=800, help="fine-tune resolution (tail)"
    )
    ap.add_argument(
        "--ft-frac", type=float, default=0.25, help="fraction of steps at res-ft"
    )
    ap.add_argument("--eval-res", type=int, default=800)
    ap.add_argument("--eval-frames", type=int, nargs="+", default=[0, 25, 50])
    ap.add_argument("--eval-every", type=int, default=200)
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
    ap.add_argument("--refine-stop", type=int, default=None, help="default: 0.9*steps")
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--init-opa", type=float, default=0.2)
    ap.add_argument(
        "--random-bkgd",
        action="store_true",
        help="composite render+GT over a random bg color each step (gsplat random_bkgd)",
    )
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()

    if args.quality:
        if args.n is None:
            args.n = 100_000
        if args.steps is None:
            args.steps = 4000
        if args.refine_stop is None:
            args.refine_stop = int(0.9 * args.steps)
        run_quality(args)
    else:
        if args.n is None:
            args.n = 25_000
        if args.steps is None:
            args.steps = 1500
        run_smoke(args)


if __name__ == "__main__":
    main()
