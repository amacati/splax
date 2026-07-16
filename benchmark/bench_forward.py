"""Benchmark suite comparing splax against gsplat on the forward render.

Three scenarios exercise the rasterizer under different Gaussian distributions:

  - synthetic: compact random clusters, uneven per-tile occupancy with no captured
    scene, standing in for concentration in bins.
  - lego: the trained lego splat rendered from the real NeRF-synthetic test cameras.
  - hf: an online reconstruction pulled from the ``amacati/splats`` dataset, orbited
    by synthetic cameras around the scene bulk.

For every scenario the suite sweeps the camera batch and, per framework, records the
best-of-repeat render time, the derived throughput, and the framework's own peak GPU
allocator use. splax renders the batch with one jitted ``jax.vmap`` over the viewmats,
gsplat renders it through its native camera-batch axis. The two frameworks keep
separate device allocators, so their peak numbers are measured independently in the
same process.

JAX preallocation is disabled so ``bytes_in_use`` reflects real on-demand use. The JAX
peak is the process cumulative peak read after each batch. The sweep is ascending, so
that value is the footprint needed to render the current batch. gsplat's peak is reset
per batch through ``torch.cuda.reset_peak_memory_stats``.

Writes ``reports/benchmark_suite.json`` (the machine-readable basis of the report) and
one sample splax render per scenario under ``reports/benchmark_assets/``. Run the
report generator afterwards, or pass ``--report`` to build the PDF in the same call.

    pixi run -e tests python benchmark/bench_suite.py
"""

from __future__ import annotations

import os

# Disable JAX preallocation before jax is imported anywhere, so device memory stats
# track real on-demand allocation rather than the reserved arena.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import timeit
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import gsplat
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import torch

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports"
ASSET_DIR = OUT_DIR / "benchmark_assets"

BATCHES = [1, 2, 4, 8, 16, 32]
WARMUP = 3
ITERS = 20
REPEAT = 3
SEED = 0
CLIP_THRESH = 0.01
EPS2D = 0.3

# Synthetic scene: compact clusters spread through the view volume.
SYN_N = 100_000
SYN_CLUSTERS = 64
SYN_RES = 256
SYN_SPREAD = 1.5
SYN_RADIUS = 0.25

LEGO_PLY = REPO / "data/scenes/lego.ply"
LEGO_TF = REPO / "data/nerf_synthetic/lego/transforms_test.json"
LEGO_RES = 400

HF_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main/robot_hall.ply"
HF_RES = 400

Scene = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass
class Scenario:
    """A benchmark scene with its camera batch and intrinsics."""

    name: str
    description: str
    scene: Scene
    viewmats: np.ndarray  # (max_batch, 4, 4) world-to-camera, OpenCV convention
    res: int
    focal: float


# --------------------------------------------------------------------------- cameras


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """World-to-camera OpenCV matrix looking from ``eye`` to ``target``.

    The camera looks along +z toward the target. ``up`` picks the world up axis; when
    omitted it defaults to +y, or +z if the view direction is nearly vertical. Roll
    about the view axis only rotates the image and leaves render cost unchanged.
    """
    z = _normalize(target - eye)
    if up is None:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.95 else np.array([0.0, 0.0, 1.0])
    # x = right, y = down, so image up (-y) aligns with the world up axis.
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # view direction parallel to up, pick another axis
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = _normalize(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return np.linalg.inv(c2w).astype(np.float32)


def orbit(
    means: np.ndarray, n: int, res: int, fov_deg: float = 50.0, elev_deg: float = 20.0
) -> tuple[np.ndarray, float]:
    """Orbit ``n`` cameras around the robust center of ``means``, framing the bulk.

    The center is the coordinate median and the radius the 90th percentile distance
    from it, so far floaters do not blow up the framing. Returns the viewmat stack and
    the focal length matching the field of view at resolution ``res``.
    """
    center = np.median(means, axis=0)
    obj_radius = float(np.percentile(np.linalg.norm(means - center, axis=1), 90))
    half_fov = np.radians(fov_deg / 2.0)
    dist = obj_radius / np.tan(half_fov) * 1.2
    focal = 0.5 * res / np.tan(half_fov)
    az = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    el = np.radians(elev_deg)
    eyes = center + dist * np.stack(
        [np.cos(el) * np.cos(az), np.full_like(az, np.sin(el)), np.cos(el) * np.sin(az)], axis=1
    )
    viewmats = np.stack([look_at(e, center) for e in eyes])
    return viewmats, float(focal)


def _halton(i: int, base: int) -> float:
    """Halton low-discrepancy sample in [0, 1) for index ``i`` and ``base``."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def box_views(
    means: np.ndarray,
    n: int,
    res: int,
    fov_deg: float = 60.0,
    x_range: tuple[float, float] = (-3.0, 3.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    z_range: tuple[float, float] = (1.0, 4.0),
) -> tuple[np.ndarray, float]:
    """Place ``n`` cameras in a box around the scene, each looking at its center.

    The eyes are spread by a Halton sequence over the box. The x and y ranges are
    offsets from the robust scene center (its coordinate median), so the cameras stay
    centered on the hall. The z range is an absolute height, since the reconstruction's
    floor sits near z = 0. Each camera faces outward, away from the center, with a
    per-camera azimuth jitter so the directions are not all perfectly radial, and looks
    level with world +z up. The fixed constraint is the box of positions.
    """
    center = np.median(means, axis=0)
    focal = 0.5 * res / np.tan(np.radians(fov_deg / 2.0))
    world_up = np.array([0.0, 0.0, 1.0])
    jitter = np.radians(45.0)  # +/- azimuth spread around the outward direction
    viewmats = []
    for i in range(1, n + 1):
        eye = np.array(
            [
                center[0] + x_range[0] + (x_range[1] - x_range[0]) * _halton(i, 2),
                center[1] + y_range[0] + (y_range[1] - y_range[0]) * _halton(i, 3),
                z_range[0] + (z_range[1] - z_range[0]) * _halton(i, 5),
            ]
        )
        d = eye - center
        base = np.arctan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 1e-3 else 0.0
        phi = base + (2.0 * _halton(i, 7) - 1.0) * jitter
        target = eye + np.array([np.cos(phi), np.sin(phi), 0.0])
        viewmats.append(look_at(eye, target, world_up))
    return np.stack(viewmats), float(focal)


def nerf_camera(frame: dict) -> np.ndarray:
    """Convert a NeRF blender camera pose to a world-to-camera view matrix."""
    c2w = np.array(frame["transform_matrix"], np.float64)
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w).astype(np.float32)


# --------------------------------------------------------------------------- scenes


def build_synthetic(n: int, clusters: int, res: int, seed: int) -> Scenario:
    """Random Gaussian clusters orbited by synthetic cameras."""
    k = jax.random.split(jax.random.key(seed), 8)
    centers = jax.random.normal(k[0], (clusters, 3)) * SYN_SPREAD
    assign = jax.random.randint(k[1], (n,), 0, clusters)
    means = centers[assign] + jax.random.normal(k[2], (n, 3)) * SYN_RADIUS
    scales = jax.random.uniform(k[3], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[4], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[5], (n, 3))
    opacities = jax.random.uniform(k[6], (n, 1))
    background = jnp.ones(3)
    viewmats, focal = orbit(np.asarray(means), max(BATCHES), res)
    scene = (means, scales, quats, colors, opacities, background)
    return Scenario(
        "synthetic",
        f"Random Gaussian clusters, {n:,} splats in {clusters} compact blobs.",
        scene,
        viewmats,
        res,
        focal,
    )


def build_lego(res: int) -> Scenario:
    """Trained lego splat rendered from the real NeRF-synthetic test cameras."""
    means, scales, quats, colors, opacities = splax.io.load_ply(LEGO_PLY)
    background = jnp.ones(3)
    tf = json.loads(LEGO_TF.read_text())
    angle = float(tf["camera_angle_x"])
    focal = 0.5 * res / np.tan(0.5 * angle)
    frames = tf["frames"][: max(BATCHES)]
    viewmats = np.stack([nerf_camera(f) for f in frames])
    scene = (means, scales, quats, colors, opacities, background)
    return Scenario(
        "lego",
        f"Trained lego splat ({means.shape[0]:,} splats), real held-out test cameras.",
        scene,
        viewmats,
        res,
        float(focal),
    )


def build_hf(res: int) -> Scenario:
    """Online reconstruction from the amacati/splats dataset, viewed from inside."""
    path = splax.io.fetch(HF_URL)
    means, scales, quats, colors, opacities = splax.io.load_ply(path)
    background = jnp.ones(3)
    viewmats, focal = box_views(np.asarray(means), max(BATCHES), res)
    scene = (means, scales, quats, colors, opacities, background)
    return Scenario(
        "hf",
        f"Online robot_hall reconstruction ({means.shape[0]:,} splats) from HF, "
        "cameras in a +/-3 m x/y box at 1-4 m height facing outward across the hall.",
        scene,
        viewmats,
        res,
        float(focal),
    )


# --------------------------------------------------------------------------- runners


def bench(call: Callable[[], object], iters: int) -> float:
    """Best-of-REPEAT mean seconds per call."""
    return min(timeit.Timer(call).repeat(repeat=REPEAT, number=iters)) / iters


def make_splax(sc: Scenario, batch: int) -> tuple[Callable[[], object], jax.Array, Callable]:
    """Build a jitted vmapped splax render over ``batch`` viewmats.

    Returns the timed call, the viewmat batch, and the jitted render itself so the
    caller can read its jit cache size and render a sample.
    """
    means, scales, quats, colors, opacities, background = sc.scene
    res, focal = sc.res, sc.focal
    views = jnp.asarray(sc.viewmats[:batch])

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
                img_shape=(res, res),
                f=(focal, focal),
                c=(res / 2, res / 2),
                clip_thresh=CLIP_THRESH,
            )

        return jax.vmap(one)(vm)

    return lambda: jax.block_until_ready(render(views)), views, render


def make_gsplat(sc: Scenario, batch: int) -> Callable[[], object]:
    """Build a gsplat render over its native camera-batch axis (C = batch)."""
    means, scales, quats, colors, opacities, background = sc.scene
    res, focal = sc.res, sc.focal

    def tt(x: jax.Array) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, np.float32), dtype=torch.float32, device="cuda")

    means_t = tt(means)
    quats_t = tt(quats)
    scales_t = tt(scales)
    opac_t = tt(opacities).reshape(-1)
    colors_t = tt(colors)
    bg_t = tt(background).reshape(3)
    k = np.array([[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]], np.float32)
    ks_t = torch.as_tensor(k, device="cuda")[None].repeat(batch, 1, 1)
    views_t = torch.as_tensor(sc.viewmats[:batch], dtype=torch.float32, device="cuda")

    def run() -> None:
        out, alpha, _meta = gsplat.rasterization(
            means_t,
            quats_t,
            scales_t,
            opac_t,
            colors_t,
            views_t,
            ks_t,
            res,
            res,
            near_plane=float(CLIP_THRESH),
            eps2d=EPS2D,
            render_mode="RGB",
        )
        _ = out + (1.0 - alpha) * bg_t
        torch.cuda.synchronize()

    return run


def save_sample(sc: Scenario) -> str:
    """Render view 0 with splax and save a PNG thumbnail, return its relative path."""
    call, _views, render = make_splax(sc, 1)
    img = np.asarray(render(jnp.asarray(sc.viewmats[:1]))[0])
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rel = f"benchmark_assets/{sc.name}.png"
    iio.imwrite(OUT_DIR / rel, img)
    return rel


def jax_stats() -> dict:
    return jax.devices()[0].memory_stats()


def run_scenario(sc: Scenario) -> dict:
    """Sweep the batch for one scenario, timing both frameworks and their peak memory."""
    n = sc.scene[0].shape[0]
    print(f"\n== {sc.name}: {n:,} gaussians, {sc.res}x{sc.res}, focal {sc.focal:.1f} ==")
    print(f"{'batch':>6} {'splax ms':>9} {'gsplat ms':>10} {'sp MB':>8} {'gs MB':>8} {'g/s':>6}")

    sample = save_sample(sc)
    rows = []
    for batch in BATCHES:
        splax_call, views, render = make_splax(sc, batch)
        gsplat_call = make_gsplat(sc, batch)

        for _ in range(WARMUP):
            splax_call()
            gsplat_call()
        cache = render._cache_size()  # ty: ignore[unresolved-attribute]
        assert cache == 1, f"expected 1 splax jit cache entry, got {cache}"

        splax_ms = bench(splax_call, ITERS) * 1e3
        splax_peak = int(jax_stats().get("peak_bytes_in_use", 0))

        torch.cuda.reset_peak_memory_stats()
        gsplat_ms = bench(gsplat_call, ITERS) * 1e3
        gsplat_peak = int(torch.cuda.max_memory_allocated())

        row = {
            "batch": batch,
            "splax": {
                "time_ms": splax_ms,
                "throughput_ips": batch / (splax_ms / 1e3),
                "peak_bytes": splax_peak,
            },
            "gsplat": {
                "time_ms": gsplat_ms,
                "throughput_ips": batch / (gsplat_ms / 1e3),
                "peak_bytes": gsplat_peak,
            },
            "speedup_gsplat_over_splax": gsplat_ms / splax_ms,
            "mem_ratio_splax_over_gsplat": splax_peak / gsplat_peak if gsplat_peak else None,
        }
        rows.append(row)
        print(
            f"{batch:>6} {splax_ms:>9.3f} {gsplat_ms:>10.3f} "
            f"{splax_peak / 1e6:>8.1f} {gsplat_peak / 1e6:>8.1f} "
            f"{gsplat_ms / splax_ms:>6.2f}"
        )

    return {
        "name": sc.name,
        "description": sc.description,
        "n_gaussians": int(n),
        "img_shape": [sc.res, sc.res],
        "focal": sc.focal,
        "cameras": int(sc.viewmats.shape[0]),
        "sample_render": sample,
        "rows": rows,
    }


BUILDERS = {
    "synthetic": lambda: build_synthetic(SYN_N, SYN_CLUSTERS, SYN_RES, SEED),
    "lego": lambda: build_lego(LEGO_RES),
    "hf": lambda: build_hf(HF_RES),
}


def run_worker(scene: str, frag: Path) -> None:
    """Benchmark one scenario in this process and write its result dict to ``frag``.

    Each scenario runs in a fresh subprocess so the JAX process-cumulative memory peak
    reflects only this scene, not whatever a previous scene left resident.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_scenario(BUILDERS[scene]())
    frag.write_text(json.dumps(result))


def main() -> None:
    """Run every scenario in isolated subprocesses, write the JSON, build the PDF."""
    import subprocess
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+", default=["synthetic", "lego", "hf"], help="subset of scenes to run"
    )
    parser.add_argument("--report", action="store_true", help="build the PDF after benchmarking")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--frag", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.frag)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for name in args.scenes:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            frag = Path(tmp.name)
        subprocess.run(
            [sys.executable, __file__, "--worker", name, "--frag", str(frag)], check=True
        )
        scenarios.append(json.loads(frag.read_text()))
        frag.unlink(missing_ok=True)

    data = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "jax_version": jax.__version__,
            "gsplat_version": gsplat.__version__,
            "torch_version": torch.__version__,
            "no_jax_preallocation": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false",
            "warmup": WARMUP,
            "iters": ITERS,
            "repeat": REPEAT,
            "batches": BATCHES,
            "metric": "forward render, best-of-repeat mean per call",
            "memory_note": (
                "Peak GPU bytes from each framework's own allocator, measured "
                "independently. splax (JAX) is the process cumulative peak after the "
                "ascending batch sweep; gsplat (torch) is reset per batch. JAX "
                "preallocation is off. The two are separate pools and not strictly "
                "comparable in absolute terms (CUDA context and kernel workspace "
                "accounting differ), but each tracks its own workload."
            ),
        },
        "scenarios": scenarios,
    }
    out = OUT_DIR / "benchmark_suite.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")

    if args.report:
        from suite_report import build_report

        pdf = OUT_DIR / "benchmark_suite.pdf"
        build_report(data, pdf)
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
