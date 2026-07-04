"""Fit a gaussian splat to the splax logo from a single fixed camera view.

A demo/asset script (not part of the test suite): it overfits ~30k gaussians to
one image -- docs/assets/logo.png -- from a random initialisation and captures renders
along the way so the logo can be seen emerging out of faint noise.

Pipeline:
  * load the logo, composite its alpha over white, downscale to ~800px wide.
  * scatter N gaussians in a thin slab at depth ~1 facing an identity camera; a
    pinhole focal chosen so the slab fills the frame. Random colours, tiny random
    scales, low initial opacity -> the step-0 render is near-blank.
  * plain per-group Adam (LRs borrowed from train_lego.py) on an L1 + 0.2 D-SSIM
    photometric loss against the logo, ~3000 steps, white background.
  * render checkpoints on a power-law step schedule (dense early where the image
    changes fastest) and write:
      results/logo_splat/step_<n>.png          -- individual checkpoints
      results/logo_splat/progression.png       -- 2x5 labelled strip
      results/logo_splat/logo_emerges.gif      -- 90-frame 30 FPS emergence loop

The background color is parametrized (--bg): it controls BOTH the training
target composite (logo alpha over that color) and the render background, so a
dark-mode README variant is just `--bg '#0d1117'`. GIF transparency is 1-bit
and would fringe, so we ship one opaque GIF per theme instead.

Usage:
  .venv/bin/python scripts/logo_splat.py                                 # light
  .venv/bin/python scripts/logo_splat.py --bg '#0d1117' \
    --gif-out docs/assets/logo_emerges_dark.gif                        # dark
"""

from __future__ import annotations

import argparse
import sys
import time
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
from PIL import Image

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import splax

OUT = Path("results/logo_splat")
LOGO = Path("docs/assets/logo.png")

N = 30_000
STEPS = 3000
SEED = 0
TARGET_W = 800  # training width; height follows the logo aspect ratio
F = 800.0  # pinhole focal length (world unit == 1 at slab depth z=1)

# progression strip steps (2 rows x 5 cols)
STRIP_STEPS = [0, 10, 25, 50, 100, 200, 400, 800, 1500, STEPS]

# GIF target: 90 frames @ 30 FPS (3 s). Rather than fix a step schedule, we render
# a DENSE candidate set, drop checkpoints that don't make monotone progress toward
# the final image (converged gaussians quiver back and forth; see monotone_filter),
# and resample 90 frames at equal *perceptual* spacing (equal cumulative
# frame-to-frame change, crossfading where checkpoints are sparse) so the visual
# change per frame is even regardless of how fast/slow training converges.
GIF_N = 90


def candidate_steps() -> list[int]:
    """Dense schedule to sample the emergence, finest where it changes fastest."""
    s = set(STRIP_STEPS)
    s |= set(range(0, 80))  # every step for the fast noise->logo phase
    s |= set(range(80, 400, 4))
    s |= set(range(400, STEPS + 1, 40))
    return sorted(x for x in s if x <= STEPS)


def monotone_filter(
    steps: list[int], frames: dict[int, np.ndarray], rel_improve: float = 0.03
) -> list[int]:
    """Drop checkpoints that don't make monotone progress toward the final image.

    Converged gaussians oscillate around the optimum, so late raw checkpoints
    quiver back and forth. Keep a frame only if its L1 distance to the FINAL
    frame beats the best-so-far by `rel_improve` (relative): every kept frame
    is then strictly closer to the end state, removing back-and-forth motion
    by construction.
    """
    final = np.asarray(frames[steps[-1]], np.float32)
    kept, best = [], float("inf")
    for s in steps:
        d = float(np.abs(np.asarray(frames[s], np.float32) - final).mean())
        if d < best * (1.0 - rel_improve):
            kept.append(s)
            best = d
    if kept[-1] != steps[-1]:
        kept.append(steps[-1])  # d(final)=0 always passes, but be explicit
    return kept


def perceptual_resample(
    steps: list[int], frames: dict[int, np.ndarray], n: int
) -> tuple[list[np.ndarray], list[str], int]:
    """Exactly `n` frames at equal cumulative-L1-change spacing along `steps`.

    Where a sample level falls between two kept checkpoints, emit a linear
    crossfade of the two instead of snapping to the nearest one. Early in
    training the checkpoints are dense so blends are ~exact frames; in the
    sparse monotone-filtered tail this becomes the smooth fade toward the
    final image that raw checkpoints can't provide.
    """
    fs = [np.asarray(frames[s], np.float32) for s in steps]
    d = np.array([0.0] + [np.abs(fs[i] - fs[i - 1]).mean() for i in range(1, len(fs))])
    cum = np.cumsum(d)
    levels = np.linspace(0.0, cum[-1], n)
    pos = np.interp(levels, cum, np.arange(len(steps)))  # fractional index
    out, labels, n_blend = [], [], 0
    for x in pos:
        i, a = int(np.floor(x)), float(x - np.floor(x))
        if a < 0.02 or i + 1 >= len(steps):
            out.append(fs[i].astype(np.uint8))
            labels.append(str(steps[i]))
        elif a > 0.98:
            out.append(fs[i + 1].astype(np.uint8))
            labels.append(str(steps[i + 1]))
        else:
            out.append(((1 - a) * fs[i] + a * fs[i + 1] + 0.5).astype(np.uint8))
            labels.append(f"{steps[i]}~{steps[i + 1]}")
            n_blend += 1
    return out, labels, n_blend


def parse_bg(s: str) -> np.ndarray:
    """'#rrggbb' or 'white' -> float RGB triple in [0,1]."""
    if s.lower() == "white":
        return np.ones(3, np.float32)
    h = s.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"--bg expects '#rrggbb' or 'white', got {s!r}")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def load_target(bg: np.ndarray) -> tuple[jax.Array, int, int]:
    """Logo -> float RGB in [0,1], alpha composited over `bg`, ~TARGET_W wide."""
    im = Image.open(LOGO).convert("RGBA")
    W0, H0 = im.size
    H = int(round(H0 * TARGET_W / W0))
    im = im.resize((TARGET_W, H), Image.Resampling.LANCZOS)
    rgba = np.asarray(im, np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:]) * bg
    return jnp.asarray(rgb), H, TARGET_W


def init_params(n: int, H: int, W: int, seed: int = 0) -> dict[str, jax.Array]:
    """Random gaussians in a thin slab at z~1, sized to the image frustum.

    At z=1 a pixel spans 1/F world units, so the visible frame is
    x in [-W/2F, W/2F], y in [-H/2F, H/2F]. Means are scattered a touch beyond
    that box; scales are a few pixels; opacity starts low so step 0 is faint.
    """
    k = jax.random.split(jax.random.key(seed), 6)
    xr, yr = 0.55 * W / F, 0.55 * H / F
    means = jnp.stack(
        [
            jax.random.uniform(k[0], (n,), minval=-xr, maxval=xr),
            jax.random.uniform(k[1], (n,), minval=-yr, maxval=yr),
            jax.random.uniform(k[2], (n,), minval=0.95, maxval=1.05),  # thin z slab
        ],
        axis=-1,
    )
    # ~2-6 px gaussians (log-uniform around 0.005 world units)
    log_s = jax.random.uniform(k[3], (n, 3), minval=np.log(0.003), maxval=np.log(0.008))
    return {
        "means": means,
        "log_scales": log_s,
        "quats": jax.random.normal(k[4], (n, 4)),
        "colors_logit": jax.random.normal(k[5], (n, 3)) * 0.6,  # varied colours
        "opac_logit": jnp.full((n, 1), -2.0),  # sigmoid(-2) ~ 0.12: faint start
    }


VIEWMAT = jnp.eye(4)  # camera at origin, +z forward (OpenCV); slab sits at z~1


def render_params(p: dict[str, jax.Array], H: int, W: int, bg: np.ndarray) -> jax.Array:
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
        viewmat=VIEWMAT,
        background=jnp.asarray(bg),
        img_shape=(H, W),
        f=(F, F),
        c=(W // 2, H // 2),
        glob_scale=1.0,
        clip_thresh=0.01,
    )[0]


def frame(p: dict[str, jax.Array], H: int, W: int, bg: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(render_params(p, H, W, bg)), 0, 1) * 255).astype(
        np.uint8
    )


def psnr(frame_u8: np.ndarray, target01: jax.Array) -> float:
    a = np.asarray(frame_u8, np.float32) / 255.0
    mse = float(np.mean((a - np.asarray(target01)) ** 2))
    return -10 * np.log10(mse) if mse > 0 else float("inf")


def main() -> tuple[float, float]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bg",
        default="white",
        help="background color, '#rrggbb' or 'white'; used for both "
        "the target composite and the render background",
    )
    ap.add_argument(
        "--gif-out",
        default=str(OUT / "logo_emerges.gif"),
        help="output path for the animated gif",
    )
    args = ap.parse_args()
    bg = parse_bg(args.bg)
    gif_out = Path(args.gif_out)

    OUT.mkdir(parents=True, exist_ok=True)
    gif_out.parent.mkdir(parents=True, exist_ok=True)
    target, H, W = load_target(bg)
    print(f"target {W}x{H}, bg {args.bg}, {N} gaussians, {STEPS} steps")

    params = init_params(N, H, W, SEED)

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

    def loss_fn(p: dict[str, jax.Array]) -> jax.Array:
        img = render_params(p, H, W, bg)
        l1 = jnp.mean(jnp.abs(img - target))
        dssim = 1.0 - dm_pix.ssim(img, target)
        return 0.8 * l1 + 0.2 * dssim

    @jax.jit
    def step(
        p: dict[str, jax.Array], opt_state: optax.OptState
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = opt.update(grads, opt_state, p)
        # apply_updates is typed as the broad optax ArrayTree; the params stay a dict.
        return (
            cast(dict[str, jax.Array], optax.apply_updates(p, updates)),
            opt_state,
            loss,
        )

    render_at = set(STRIP_STEPS) | set(candidate_steps())
    frames = {}  # step -> uint8 HxWx3

    frames[0] = frame(params, H, W, bg)
    p0 = psnr(frames[0], target)
    print(f"random-init PSNR: {p0:.2f} dB")

    t0 = time.perf_counter()
    for it in range(1, STEPS + 1):
        params, opt_state, loss = step(params, opt_state)
        if it in render_at:
            loss.block_until_ready()
            frames[it] = frame(params, H, W, bg)
        if it % 500 == 0 or it == STEPS:
            loss.block_until_ready()
            print(f"step {it:5d}  loss {float(loss):.4f}")
    wall = time.perf_counter() - t0

    final = frames[STEPS]
    p_final = psnr(final, target)
    print(f"final PSNR: {p_final:.2f} dB  ({STEPS} steps in {wall:.1f}s)")

    # individual checkpoints (strip steps)
    for s in STRIP_STEPS:
        iio.imwrite(OUT / f"step_{s}.png", frames[s])

    write_progression(frames, bg)
    write_gif(frames, gif_out)

    print(f"wrote {OUT}/step_*.png, progression.png, {gif_out}")
    return p_final, wall


def write_progression(frames: dict[int, np.ndarray], bg: np.ndarray) -> None:
    face = tuple(float(c) for c in bg)
    # relative luminance decides label color so titles stay readable on dark bg
    text = (
        "white" if (0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]) < 0.5 else "black"
    )
    fig, axes = plt.subplots(2, 5, figsize=(15, 4.2), facecolor=face)
    for ax, s in zip(axes.ravel(), STRIP_STEPS):
        ax.imshow(frames[s])
        ax.set_title(
            f"step {s}" if s < STEPS else f"step {s} (final)",
            fontsize=11,
            pad=4,
            color=text,
        )
        ax.axis("off")
    fig.suptitle("splax logo emerging from noise", fontsize=14, y=0.99, color=text)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "progression.png", dpi=110, facecolor=face)
    plt.close(fig)


def write_gif(frames: dict[int, np.ndarray], gif_out: Path) -> None:
    kept = monotone_filter(candidate_steps(), frames)
    seq, labels, n_blend = perceptual_resample(kept, frames, GIF_N)
    print(
        f"gif: {len(kept)} monotone checkpoints -> {len(seq)} frames "
        f"({n_blend} blended)"
    )
    print(f"gif frame steps: {labels}")
    seq = seq + [frames[STEPS]] * 45  # hold the final frame ~1.5 s

    def encode(seq: list[np.ndarray]) -> float:
        iio.imwrite(gif_out, seq, duration=1000 / 30, loop=0, subrectangles=True)
        return gif_out.stat().st_size / 1e6

    mb = encode(seq)
    H, W = seq[0].shape[:2]
    for w2 in (640, 560, 480):  # downscale rather than dropping frames
        if mb <= 8.0:
            break
        h2 = int(round(H * w2 / W))
        small = [
            np.asarray(Image.fromarray(f).resize((w2, h2), Image.Resampling.LANCZOS))
            for f in seq
        ]
        mb = encode(small)
    print(f"gif: {len(seq)} frames incl. hold, {mb:.2f} MB")


if __name__ == "__main__":
    main()
