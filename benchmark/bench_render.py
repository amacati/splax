"""Benchmark splax against gsplat on the forward render, sweeping the camera batch.

Both frameworks render the same random gaussian scene from a batch of cameras. splax renders the
batch with a single jitted jax.vmap over the viewmats, gsplat renders it through its native
camera-batch axis.

The scene is a set of compact gaussian clusters spread through the view volume rather than a uniform
cloud, so the tiler sees the uneven per-tile occupancy of a real reconstruction.

Run in the pixi tests environment:

    pixi run -e tests python benchmark/bench_render.py
"""

from __future__ import annotations

import argparse
import json
import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import gsplat
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

# Fixed scene defaults. Override on the command line.
N_GAUSSIANS = 100_000
HEIGHT = 256
WIDTH = 256
BATCHES = [1, 2, 4, 8, 16, 32, 64]
N_CLUSTERS = 64
WARMUP = 5
ITERS = 30
REPEAT = 5
GLOB_SCALE = 1.0
CLIP_THRESH = 0.01
EPS2D = 0.3
SEED = 0

# Cluster centers spread through the volume in front of the camera (depth ~5), each
# cluster a compact blob whose extent dwarfs a single gaussian. These give the tiler
# realistic uneven occupancy without depending on a captured scene.
CLUSTER_SPREAD = 1.5
CLUSTER_RADIUS = 0.25

Scene = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def build_scene(n: int, seed: int, n_clusters: int = N_CLUSTERS) -> Scene:
    """Build a random gaussian scene of compact clusters, like objects on surfaces."""
    k = jax.random.split(jax.random.key(seed), 8)
    centers = jax.random.normal(k[0], (n_clusters, 3)) * CLUSTER_SPREAD
    assign = jax.random.randint(k[1], (n,), 0, n_clusters)
    means = centers[assign] + jax.random.normal(k[2], (n, 3)) * CLUSTER_RADIUS
    scales = jax.random.uniform(k[3], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[4], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[5], (n, 3))
    opacities = jax.random.uniform(k[6], (n, 1))
    background = jax.random.uniform(k[7], (3,))
    return means, scales, quats, colors, opacities, background


def viewmats(batch: int) -> jax.Array:
    """Stack ``batch`` slightly translated world-to-camera OpenCV matrices."""
    base = jnp.array([[1, 0, 0, 0], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    offsets = jnp.linspace(-0.3, 0.3, batch)
    return jax.vmap(lambda dx: base.at[0, 3].set(dx))(offsets)


def bench(call: Callable[[], object], iters: int) -> float:
    """Return the best-of-``REPEAT`` mean seconds per call with the ``timeit`` module."""
    return min(timeit.Timer(call).repeat(repeat=REPEAT, number=iters)) / iters


def make_splax_run(
    scene: Scene, batch: int, img_shape: tuple[int, int]
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """Build a jitted vmapped splax render and the batch of viewmats to feed it."""
    means, scales, quats, colors, opacities, background = scene
    H, _ = img_shape
    views = viewmats(batch)

    @jax.jit
    def render(vm: jax.Array) -> jax.Array:
        def one(v: jax.Array) -> jax.Array:
            return splax.inference.render(
                means,
                scales,
                quats,
                colors,
                opacities,
                viewmat=v,
                background=background,
                img_shape=img_shape,
                f=(float(H), float(H)),
                c=(WIDTH // 2, HEIGHT // 2),
                glob_scale=GLOB_SCALE,
                clip_thresh=CLIP_THRESH,
            )

        return jax.vmap(one)(vm)

    return render, views


def make_gsplat_run(
    scene: Scene, batch: int, img_shape: tuple[int, int]
) -> Callable[[], torch.Tensor]:
    """Build a gsplat render over its native camera-batch axis (C = batch)."""
    means, scales, quats, colors, opacities, background = scene
    H, W = img_shape

    def to_torch(x: jax.Array) -> torch.Tensor:
        return torch.as_tensor(np.array(x, np.float32), dtype=torch.float32, device="cuda")

    means_t = to_torch(means)
    quats_t = to_torch(quats)
    scales_t = to_torch(scales) * float(GLOB_SCALE)
    opac_t = to_torch(opacities).reshape(-1)
    colors_t = to_torch(colors)
    K = np.array([[H, 0.0, W / 2], [0.0, H, H / 2], [0.0, 0.0, 1.0]], np.float32)
    ks_t = torch.as_tensor(K, device="cuda")[None].repeat(batch, 1, 1)
    views_t = to_torch(viewmats(batch))
    bg_t = to_torch(background).reshape(3)

    def run() -> torch.Tensor:
        # Composite over the background manually (out + (1 - alpha) * bg), matching
        # splax.render and the parity adapter rather than gsplat's backgrounds arg.
        out, alpha, _meta = gsplat.rasterization(
            means_t,
            quats_t,
            scales_t,
            opac_t,
            colors_t,
            views_t,
            ks_t,
            W,
            H,
            near_plane=float(CLIP_THRESH),
            eps2d=EPS2D,
            render_mode="RGB",
        )
        return out + (1.0 - alpha) * bg_t

    return run


def plot(rows: list[dict], title: str, path: Path) -> None:
    """Plot render time and throughput against batch size for both frameworks."""
    batches = np.array([r["batch"] for r in rows])
    splax_ms = np.array([r["splax_ms"] for r in rows])
    gsplat_ms = np.array([r["gsplat_ms"] for r in rows])

    fig, (ax_time, ax_thru) = plt.subplots(1, 2, figsize=(12, 5))
    ax_time.plot(batches, splax_ms, "o-", label="splax", color="#1b9e77")
    ax_time.plot(batches, gsplat_ms, "s-", label="gsplat", color="#d95f02")
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("batch size (cameras)")
    ax_time.set_ylabel("render time per call (ms)")
    ax_time.set_title("Render time vs batch size")
    ax_time.legend()
    ax_time.grid(True, which="both", alpha=0.3)

    ax_thru.plot(batches, batches / (splax_ms / 1e3), "o-", label="splax", color="#1b9e77")
    ax_thru.plot(batches, batches / (gsplat_ms / 1e3), "s-", label="gsplat", color="#d95f02")
    ax_thru.set_xscale("log", base=2)
    ax_thru.set_xlabel("batch size (cameras)")
    ax_thru.set_ylabel("throughput (images / s)")
    ax_thru.set_title("Throughput vs batch size")
    ax_thru.legend()
    ax_thru.grid(True, which="both", alpha=0.3)

    for ax in (ax_time, ax_thru):
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    """Run the batch-scaling benchmark and write the plot and raw numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-gaussians", type=int, default=N_GAUSSIANS)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()

    img_shape = (args.height, args.width)
    scene = build_scene(args.n_gaussians, SEED, args.n_clusters)

    rows = []
    header = f"splax vs gsplat, N={args.n_gaussians:,} gaussians, {args.width}x{args.height}"
    print(header)
    print(f"{'batch':>6} {'splax ms':>10} {'gsplat ms':>10} {'splax/gsplat':>13}")
    for batch in args.batches:
        render, views = make_splax_run(scene, batch, img_shape)
        gsplat_run = make_gsplat_run(scene, batch, img_shape)

        def splax_call() -> None:
            jax.block_until_ready(render(views))

        def gsplat_call() -> None:
            gsplat_run()
            torch.cuda.synchronize()

        # Warm up both, then assert the jitted render compiled a single variant
        for _ in range(args.warmup):
            splax_call()
            gsplat_call()
        cache = render._cache_size()  # ty: ignore[unresolved-attribute]
        assert cache == 1, f"expected 1 jit cache entry after warmup, got {cache}"

        splax_ms = bench(splax_call, args.iters) * 1e3
        gsplat_ms = bench(gsplat_call, args.iters) * 1e3

        cache = render._cache_size()  # ty: ignore[unresolved-attribute]
        assert cache == 1, f"expected 1 jit cache entry after timing, got {cache}"

        ratio = splax_ms / gsplat_ms
        print(f"{batch:>6} {splax_ms:>10.3f} {gsplat_ms:>10.3f} {ratio:>13.2f}")
        rows.append({"batch": batch, "splax_ms": splax_ms, "gsplat_ms": gsplat_ms})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_gaussians": args.n_gaussians,
        "n_clusters": args.n_clusters,
        "img_shape": img_shape,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeat": REPEAT,
        "rows": rows,
    }
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2))
    plot(rows, header, args.out_dir / "batch_scaling.png")
    print(f"wrote {args.out_dir / 'results.json'} and {args.out_dir / 'batch_scaling.png'}")


if __name__ == "__main__":
    main()
