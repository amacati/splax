"""Train Gaussians on the synthetic lego scene.

Usage:
    python scripts/train_lego.py [flags]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import dm_pix
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax

import splax

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

LEGO = Path("data/nerf_synthetic/lego")
SPLAT_KEYS = ("means", "log_scales", "quats", "colors_logit", "opac_logit")


def load_view(frame: dict, res: int) -> tuple[jax.Array, jax.Array]:
    """Load a composited frame and view matrix."""
    img = iio.imread(LEGO / (frame["file_path"].lstrip("./") + ".png")).astype(np.float32) / 255.0
    img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
    height = img.shape[0]
    if height != res:
        factor = height // res
        img = img.reshape(res, factor, res, factor, 3).mean((1, 3))
    return jnp.asarray(img), jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"]))


def load_view_alpha(frame: dict, res: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Load a frame with separate RGB, alpha, and view matrix."""
    img = iio.imread(LEGO / (frame["file_path"].lstrip("./") + ".png")).astype(np.float32) / 255.0
    height = img.shape[0]
    if height != res:
        factor = height // res
        img = img.reshape(res, factor, res, factor, 4).mean((1, 3))
    viewmat = jnp.asarray(splax.utils.nerf_camera(frame["transform_matrix"]))
    return jnp.asarray(img[..., :3]), jnp.asarray(img[..., 3:]), viewmat


def init_params(n: int, init_scale: float, init_opa: float, seed: int = 0) -> dict[str, jax.Array]:
    """Initialize lego training parameters."""
    key_means, _, key_quats, key_colors, _ = jax.random.split(jax.random.key(seed), 5)
    return {
        "means": jax.random.uniform(key_means, (n, 3), minval=-1.3, maxval=1.3),
        "log_scales": jnp.full((n, 3), jnp.log(init_scale)),
        "quats": jax.random.normal(key_quats, (n, 4)),
        "colors_logit": jax.random.normal(key_colors, (n, 3)) * 0.1,
        "opac_logit": jnp.full((n, 1), float(np.log(init_opa / (1 - init_opa)))),
    }


def psnr(a: np.ndarray | jax.Array, b: np.ndarray | jax.Array) -> float:
    """Compute the PSNR between two images."""
    mse = float(np.mean((np.clip(np.asarray(a), 0, 1) - np.asarray(b)) ** 2))
    return -10 * np.log10(mse) if mse > 0 else float("inf")


def save_ply(path: str | Path, params: dict[str, jax.Array]) -> None:
    """Write the fitted Gaussians to a PLY file."""
    scales = jnp.exp(params["log_scales"])
    quats = params["quats"] / (jnp.linalg.norm(params["quats"], axis=-1, keepdims=True) + 1e-8)
    colors = jax.nn.sigmoid(params["colors_logit"])
    opac = jax.nn.sigmoid(params["opac_logit"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    splax.io.write_ply(path, params["means"], scales, quats, colors, opac)
    logger.info(f"wrote {path}")


def _make_step(
    opt: optax.GradientTransformation,
    res: int,
    focal: float,
    ssim_lambda: float,
    opacity_reg: float,
    scale_reg: float,
    antialiased: bool = False,
) -> Callable:
    """Build a jitted training step."""
    camera: dict = {"img_shape": (res, res), "f": (focal, focal), "antialiased": antialiased}

    def loss_fn(
        params: dict[str, jax.Array],
        gt_rgb: jax.Array,
        gt_alpha: jax.Array,
        bg: jax.Array,
        viewmat: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        splats = tuple(params[k] for k in SPLAT_KEYS)
        img, _ = splax.render_log(*splats, viewmat=viewmat, background=bg, **camera)
        gt = gt_alpha * gt_rgb + (1.0 - gt_alpha) * bg
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = 1.0 - dm_pix.ssim(img, gt)
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * dssim
        loss = loss + opacity_reg * jnp.mean(jax.nn.sigmoid(params["opac_logit"]))
        loss = loss + scale_reg * jnp.mean(jnp.exp(params["log_scales"]))
        return loss, l1

    @jax.jit
    def step(
        params: dict[str, jax.Array],
        opt_state: optax.OptState,
        gt_rgb: jax.Array,
        gt_alpha: jax.Array,
        bg: jax.Array,
        viewmat: jax.Array,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        (loss, l1), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, gt_rgb, gt_alpha, bg, viewmat
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        return cast("dict[str, jax.Array]", optax.apply_updates(params, updates)), opt_state, l1

    return step


def _reset_opt_state(opt_state: optax.OptState, reset_mask: jax.Array) -> optax.OptState:
    """Reset optimizer state for relocated rows."""
    n = reset_mask.shape[0]
    keep = (~reset_mask).astype(jnp.float32)

    def reset_leaf(x: jax.Array) -> jax.Array:
        if isinstance(x, jnp.ndarray) and x.ndim >= 1 and x.shape[0] == n:
            return x * keep.reshape((-1,) + (1,) * (x.ndim - 1))
        return x

    return jax.tree.map(reset_leaf, opt_state)


def train(args: argparse.Namespace) -> dict:
    """Train the lego scene and return metrics."""
    train_meta = json.loads((LEGO / "transforms_train.json").read_text())
    test_meta = json.loads((LEGO / "transforms_test.json").read_text())
    camera_angle_x = train_meta["camera_angle_x"]

    res_ft = args.res_ft or args.res
    ft_start = int(args.steps * (1.0 - args.ft_frac)) if res_ft != args.res else args.steps

    def frames_at(res: int) -> tuple[jax.Array, jax.Array]:
        """Load composited training frames."""
        imgs, viewmats = zip(*[load_view(frame, res) for frame in train_meta["frames"]])
        return jnp.stack(imgs), jnp.stack(viewmats)

    def frames_at_alpha(res: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Load training frames with alpha."""
        rgb, alpha, viewmats = zip(*[load_view_alpha(frame, res) for frame in train_meta["frames"]])
        return jnp.stack(rgb), jnp.stack(alpha), jnp.stack(viewmats)

    logger.info(
        f"loading {len(train_meta['frames'])} train views at resolutions {args.res} and {res_ft}"
    )
    if args.random_bkgd:
        train_views = {args.res: frames_at_alpha(args.res)}
        if res_ft != args.res:
            train_views[res_ft] = frames_at_alpha(res_ft)
    else:
        ones_alpha = {res: jnp.ones((res, res, 1), jnp.float32) for res in {args.res, res_ft}}
        train_views = {}
        imgs, viewmats = frames_at(args.res)
        train_views[args.res] = (imgs, ones_alpha[args.res], viewmats)
        if res_ft != args.res:
            imgs, viewmats = frames_at(res_ft)
            train_views[res_ft] = (imgs, ones_alpha[res_ft], viewmats)

    eval_res = args.eval_res
    eval_views = [load_view(test_meta["frames"][index], eval_res) for index in args.eval_frames]
    eval_imgs = [np.asarray(img) for img, _ in eval_views]
    eval_viewmats = [viewmat for _, viewmat in eval_views]
    eval_focal = 0.5 * eval_res / np.tan(0.5 * camera_angle_x)

    params = init_params(args.n, args.init_scale, args.init_opa, args.seed)
    eval_camera: dict = {"img_shape": (eval_res, eval_res), "f": (eval_focal, eval_focal)}
    eval_camera |= {"background": jnp.ones(3), "antialiased": args.antialiased}

    def eval_render(viewmat: jax.Array) -> jax.Array:
        splats = tuple(params[k] for k in SPLAT_KEYS)
        return splax.render_log(*splats, viewmat=viewmat, **eval_camera)[0]

    def eval_psnr() -> tuple[float, list[float]]:
        """Evaluate the current parameters on held out frames."""
        scores = [psnr(eval_render(viewmat), gt) for gt, viewmat in zip(eval_imgs, eval_viewmats)]
        return float(np.mean(scores)), scores

    means_sched = optax.exponential_decay(args.means_lr, args.steps, 0.01)
    transforms: dict[Hashable, optax.GradientTransformation] = {
        "means": optax.adam(means_sched),
        "log_scales": optax.adam(args.scales_lr),
        "quats": optax.adam(args.quats_lr),
        "colors_logit": optax.adam(args.colors_lr),
        "opac_logit": optax.adam(args.opac_lr),
    }
    opt = optax.multi_transform(transforms, {key: key for key in params})
    opt_state = opt.init(params)

    binoms = splax.mcmc.make_binoms(51)

    @jax.jit
    def relocate(
        params: dict[str, jax.Array], opt_state: optax.OptState, key: jax.Array
    ) -> tuple[dict[str, jax.Array], optax.OptState]:
        splats = tuple(params[k] for k in SPLAT_KEYS)
        new, reset_mask = splax.mcmc.relocate(key, *splats, binoms, min_opacity=args.min_opacity)
        return dict(zip(SPLAT_KEYS, new)), _reset_opt_state(opt_state, reset_mask)

    @jax.jit
    def add_noise(
        params: dict[str, jax.Array], key: jax.Array, scale: float
    ) -> dict[str, jax.Array]:
        splats = (params["means"], params["log_scales"], params["quats"], params["opac_logit"])
        noisy_means = splax.mcmc.inject_noise(key, *splats, scale, min_opacity=args.min_opacity)
        return {**params, "means": noisy_means}

    steppers = {
        r: _make_step(
            opt,
            r,
            0.5 * r / np.tan(0.5 * camera_angle_x),
            args.ssim_lambda,
            args.opacity_reg,
            args.scale_reg,
            antialiased=args.antialiased,
        )
        for r in train_views
    }

    init_psnr, _ = eval_psnr()
    logger.info(f"random-init eval PSNR: {init_psnr:.2f} dB")
    curve = [{"step": 0, "eval_psnr": round(init_psnr, 3), "res": None}]

    key = jax.random.key(args.seed + 1)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_meta["frames"]))
    white = jnp.ones(3)
    t0 = time.perf_counter()
    for step_idx in range(1, args.steps + 1):
        res = args.res if step_idx < ft_start else res_ft
        imgs, gt_alpha, viewmats = train_views[res]
        view_idx = int(order[step_idx % len(order)])
        if args.random_bkgd:
            key, bg_key = jax.random.split(key)
            background = jax.random.uniform(bg_key, (3,))
            alpha = gt_alpha[view_idx]
        else:
            background = white
            alpha = gt_alpha
        params, opt_state, l1 = steppers[res](
            params, opt_state, imgs[view_idx], alpha, background, viewmats[view_idx]
        )

        if (
            args.relocate_every
            and args.refine_start < step_idx < args.refine_stop
            and step_idx % args.relocate_every == 0
        ):
            key, step_key = jax.random.split(key)
            params, opt_state = relocate(params, opt_state, step_key)

        if (
            args.noise_lr > 0
            and step_idx < args.steps
            and (args.noise_stop_iter < 0 or step_idx < args.noise_stop_iter)
        ):
            scale = float(jnp.asarray(means_sched(step_idx))) * args.noise_lr
            key, step_key = jax.random.split(key)
            params = add_noise(params, step_key, scale)

        if step_idx % args.eval_every == 0 or step_idx == args.steps:
            l1.block_until_ready()
            eval_score, _ = eval_psnr()
            curve.append(
                {
                    "step": step_idx,
                    "eval_psnr": round(eval_score, 3),
                    "train_l1": round(float(l1), 5),
                    "res": res,
                }
            )
            logger.info(
                f"step {step_idx:5d}  res {res}  train L1 {float(l1):.4f}  "
                f"eval PSNR {eval_score:5.2f} dB"
            )
    wall = time.perf_counter() - t0

    final_psnr, per_frame = eval_psnr()
    logger.info(f"final eval PSNR: {final_psnr:.2f} dB {[round(score, 2) for score in per_frame]}")
    logger.info(
        f"{args.steps} steps / {args.n} gaussians in {wall:.1f}s "
        f"({wall / args.steps * 1000:.1f} ms/step)"
    )

    Path("results").mkdir(exist_ok=True)
    for frame_idx, (gt, viewmat) in zip(args.eval_frames, zip(eval_imgs, eval_viewmats)):
        rendered = np.clip(np.asarray(eval_render(viewmat)), 0, 1)
        iio.imwrite(
            f"results/train_lego_eval_f{frame_idx}.png",
            (np.concatenate([rendered, gt], 1) * 255).astype(np.uint8),
        )

    preview = np.clip(np.asarray(eval_render(eval_viewmats[0])), 0, 1)
    iio.imwrite(
        "results/train_lego_after.png",
        (np.concatenate([preview, eval_imgs[0]], 1) * 255).astype(np.uint8),
    )

    out = {
        "scene": "lego",
        "n": args.n,
        "steps": args.steps,
        "res": args.res,
        "res_ft": res_ft,
        "ft_start": ft_start,
        "eval_res": eval_res,
        "wall_s": round(wall, 2),
        "ms_per_step": round(wall / args.steps * 1000, 3),
        "init_eval_psnr": round(init_psnr, 3),
        "final_eval_psnr": round(final_psnr, 3),
        "per_frame_psnr": {
            frame_idx: round(score, 3) for frame_idx, score in zip(args.eval_frames, per_frame)
        },
        "hparams": {
            key: getattr(args, key)
            for key in (
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
    Path("results/train_lego_fit.json").write_text(json.dumps(out, indent=2))
    logger.info("wrote results/train_lego_fit.json and results/train_lego_eval_f*.png")

    if args.out_ply:
        save_ply(args.out_ply, params)

    if args.plot:
        _plot_curve(curve, ft_start if res_ft != args.res else None, wall, final_psnr)
    return out


def _plot_curve(curve: list[dict], ft_start: int | None, wall: float, final_psnr: float) -> None:
    """Plot the evaluation PSNR curve."""
    steps = [point["step"] for point in curve]
    psnrs = [point["eval_psnr"] for point in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, psnrs, "-o", ms=3, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out PSNR at 800x800 (dB)")
    if ft_start:
        ax.axvline(ft_start, ls=":", color="C3", lw=1, label="fine-tune")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"lego fit: {final_psnr:.2f} dB in {wall:.0f}s")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/train_lego_curve.png", dpi=130)
    plt.close(fig)
    logger.info("wrote results/train_lego_curve.png")


def main() -> None:
    """Parse CLI arguments and train the lego scene."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--antialiased", action="store_true", help="enable opacity compensation")
    ap.add_argument("--out-ply", help="save the fitted splats as a 3DGS .ply")
    ap.add_argument("--res-ft", type=int, default=800, help="set the fine-tune resolution")
    ap.add_argument("--ft-frac", type=float, default=0.25, help="set the fine-tune fraction")
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
        "--noise-stop-iter", type=int, default=-1, help="stop noise injection after this step"
    )
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--relocate-every", type=int, default=100)
    ap.add_argument("--refine-start", type=int, default=200)
    ap.add_argument("--refine-stop", type=int, default=None, help="default: 0.9 * steps")
    ap.add_argument("--init-scale", type=float, default=0.05)
    ap.add_argument("--init-opa", type=float, default=0.2)
    ap.add_argument("--random-bkgd", action="store_true", help="sample a new background each step")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()

    if args.refine_stop is None:
        args.refine_stop = int(0.9 * args.steps)
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main()
