"""Benchmark suite comparing splax against gsplat on the forward render.

Three scenes cover different Gaussian distributions: random synthetic clusters, the trained lego
splat with its real test cameras, and an online reconstruction from the ``amacati/splats`` dataset.
Each scene sweeps the camera batch and records render time, throughput, and peak GPU memory for
both frameworks.

Results are written to ``reports/benchmark_suite.json`` plus one sample render per scene under
``reports/benchmark_assets/``. Run the benchmark with:

    pixi run -e tests python benchmark/bench_forward.py
"""

from __future__ import annotations

import os

# Disable JAX preallocation before jax is imported anywhere, so device memory stats track real
# on-demand allocation rather than the reserved arena.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import multiprocessing
import timeit
from collections import namedtuple
from datetime import datetime, timezone
from functools import partial
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

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports"
ASSET_DIR = OUT_DIR / "benchmark_assets"

BATCHES = [2**i for i in range(10)]
WARMUP = 1
ITERS = 20
REPEAT = 3
SEED = 0

# Synthetic scene settings
SYN_N = 100_000
SYN_CLUSTERS = 64  # Create dense clusters to mimic real-world splats
SYN_RES = 256
SYN_SPREAD = 1.5
SYN_RADIUS = 0.25

LEGO_PLY = REPO / "data/scenes/lego.ply"
LEGO_TF = REPO / "data/nerf_synthetic/lego/transforms_test.json"
LEGO_RES = 400

HF_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main/robot_hall.ply"
HF_RES = 400

# A benchmark scene with its cameras. scene is the (means, scales, quats, colors, opacities,
# background) arrays, viewmats is (max_batch, 4, 4) world-to-camera in OpenCV convention.
Scene = namedtuple("Scene", ["name", "description", "scene", "viewmats", "res", "focal"])


# region cameras


def orbit(
    center: np.ndarray, n: int, res: int, radius: float, fov_deg: float = 60.0
) -> tuple[np.ndarray, float]:
    """Orbit ``n`` cameras in a full circle of ``radius`` around ``center``, looking at it.

    Returns the viewmat stack and the focal length.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    eyes = center + radius * np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], axis=1)
    viewmats = splax.utils.look_at(eyes, center, up=(0, 0, 1))
    return viewmats, float(0.5 * res / np.tan(np.radians(fov_deg / 2.0)))


# region scenes


def build_synthetic() -> Scene:
    """Random Gaussian clusters orbited by synthetic cameras."""
    n, res = SYN_N, SYN_RES
    k = jax.random.split(jax.random.key(SEED), 8)
    centers = jax.random.normal(k[0], (SYN_CLUSTERS, 3)) * SYN_SPREAD
    assign = jax.random.randint(k[1], (n,), 0, SYN_CLUSTERS)
    means = centers[assign] + jax.random.normal(k[2], (n, 3)) * SYN_RADIUS
    scales = jax.random.uniform(k[3], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[4], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[5], (n, 3))
    opacities = jax.random.uniform(k[6], (n, 1))
    scene = (means, scales, quats, colors, opacities, jnp.ones(3))
    viewmats, focal = orbit(np.zeros(3), max(BATCHES), res, radius=9.0)
    description = f"Random Gaussian clusters, {n:,} splats in {SYN_CLUSTERS} compact blobs."
    return Scene("synthetic", description, scene, viewmats, res, focal)


def build_lego() -> Scene:
    """Trained lego splat rendered from the real NeRF-synthetic test cameras."""
    means, scales, quats, colors, opacities = splax.io.load_ply(LEGO_PLY)
    scene = (means, scales, quats, colors, opacities, jnp.ones(3))
    tf = json.loads(LEGO_TF.read_text())
    focal = 0.5 * LEGO_RES / np.tan(0.5 * tf["camera_angle_x"])
    viewmats = splax.utils.nerf_camera([f["transform_matrix"] for f in tf["frames"]])
    viewmats = np.resize(viewmats, (max(BATCHES), 4, 4))  # repeat to max batch
    description = f"Trained lego splat ({means.shape[0]:,} splats), "
    description += f"{viewmats.shape[0]} real held-out test cameras."
    return Scene("lego", description, scene, viewmats, LEGO_RES, float(focal))


def build_hf() -> Scene:
    """Online reconstruction from the amacati/splats dataset, viewed from inside."""
    means, scales, quats, colors, opacities = splax.io.load_ply(splax.io.fetch(HF_URL))
    scene = (means, scales, quats, colors, opacities, jnp.ones(3))
    viewmats, focal = orbit(np.array([0.0, 0.0, 3.0]), max(BATCHES), HF_RES, radius=4.0)
    description = f"Online robot_hall reconstruction ({means.shape[0]:,} splats) from HF, "
    description += "cameras orbiting inside the hall."
    return Scene("hf", description, scene, viewmats, HF_RES, focal)


# region runners


def make_splax(sc: Scene, batch: int) -> tuple[Callable[[], object], Callable]:
    """Build a splax render over ``batch`` viewmats.

    Returns the timed call and the jitted render itself.
    """
    *params, background = sc.scene
    views = jnp.asarray(sc.viewmats[:batch])
    camera = {"background": background, "img_shape": (sc.res, sc.res), "f": (sc.focal, sc.focal)}
    render_view = partial(splax.render, *params, **camera)
    render = jax.jit(jax.vmap(lambda viewmat: render_view(viewmat=viewmat)[0]))
    return lambda: jax.block_until_ready(render(viewmat=views)), render


def make_gsplat(sc: Scene, batch: int) -> Callable[[], object]:
    """Build a gsplat render over ``batch`` viewmats."""
    res, focal = sc.res, sc.focal
    # Older torch versions crash for asarray from jax Arrays
    tensors = [torch.asarray(np.asarray(x, np.float32), device="cuda") for x in sc.scene]
    means_t, scales_t, quats_t, colors_t, opac_t, bg_t = tensors
    opac_t = opac_t.reshape(-1)
    params = [means_t, quats_t, scales_t, opac_t, colors_t]
    k = np.array([[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]], np.float32)
    ks_t = torch.as_tensor(k, device="cuda")[None].repeat(batch, 1, 1)
    views_t = torch.as_tensor(sc.viewmats[:batch], device="cuda")

    def run() -> None:
        out, alpha, _ = gsplat.rasterization(*params, views_t, ks_t, res, res)
        _ = out + (1.0 - alpha) * bg_t
        torch.cuda.synchronize()

    return run


def save_example_view(sc: Scene) -> str:
    """Render view 0 with splax and save a PNG thumbnail, return its relative path."""
    _, render = make_splax(sc, 1)
    img = np.asarray(render(viewmat=jnp.asarray(sc.viewmats[:1]))[0])
    img = (img * 255).astype(np.uint8)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rel = f"benchmark_assets/{sc.name}.png"
    iio.imwrite(OUT_DIR / rel, img)
    return rel


def measure_splax(sc: Scene, batch: int) -> dict:
    """Time the splax render at ``batch`` and read the JAX allocator peak."""
    splax_call, render = make_splax(sc, batch)
    for _ in range(WARMUP):
        splax_call()
    cache = render._cache_size()  # ty: ignore[unresolved-attribute]
    assert cache == 1, f"expected 1 splax jit cache entry, got {cache}"

    ms = (min(timeit.Timer(splax_call).repeat(repeat=REPEAT, number=ITERS)) / ITERS) * 1e3
    cache = render._cache_size()  # ty: ignore[unresolved-attribute]
    assert cache == 1, f"expected 1 splax jit cache entry after timing, got {cache}"
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    return {"time_ms": ms, "peak_bytes": peak}


def measure_gsplat(sc: Scene, batch: int) -> dict:
    """Time the gsplat render at ``batch`` and read the torch allocator peak."""
    gsplat_call = make_gsplat(sc, batch)
    for _ in range(WARMUP):
        gsplat_call()
    torch.cuda.reset_peak_memory_stats()
    ms = (min(timeit.Timer(gsplat_call).repeat(repeat=REPEAT, number=ITERS)) / ITERS) * 1e3
    return {"time_ms": ms, "peak_bytes": torch.cuda.max_memory_allocated()}


def run_scene(name: str, framework: str) -> dict:
    """Sweep the batch for one scene with one framework, so each framework has the GPU alone."""
    sc = BUILDERS[name]()
    n = sc.scene[0].shape[0]
    print(f"\n== {framework} {sc.name}: {n:,} gaussians, {sc.res}x{sc.res} ==")
    measure = measure_splax if framework == "splax" else measure_gsplat

    rows = []
    oom = False
    for batch in BATCHES:
        row = {"time_ms": float("nan"), "peak_bytes": float("nan")}
        try:
            if not oom:  # once a batch runs out of memory, all larger ones keep the NaN row
                row = measure(sc, batch)
        except (jax.errors.JaxRuntimeError, torch.OutOfMemoryError):
            print(f"batch {batch} ran out of memory, NaN from here on")
            oom = True
        rows.append(row)
        print(f"{batch:>6} {row['time_ms']:>9.3f} ms {row['peak_bytes'] / 1e6:>9.1f} MB")

    return {
        "name": sc.name,
        "description": sc.description,
        "n_gaussians": n,
        "img_shape": [sc.res, sc.res],
        "focal": sc.focal,
        "cameras": sc.viewmats.shape[0],
        "rows": rows,
    }


BUILDERS = {"synthetic": build_synthetic, "lego": build_lego, "hf": build_hf}


def main() -> None:
    """Run every scene in an isolated process and merge the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(BUILDERS), help="subset of scenes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    scenes = []
    for name in args.scenes:
        rows = {}
        for framework in ("splax", "gsplat"):
            with ctx.Pool(1) as pool:  # We do not want to parallelize to avoid GPU contention
                result = pool.apply(run_scene, (name, framework))
            rows[framework] = result.pop("rows")
        pairs = zip(BATCHES, rows["splax"], rows["gsplat"])
        result["rows"] = [{"batch": b, "splax": s, "gsplat": g} for b, s, g in pairs]
        scenes.append(result)
    for scene in scenes:
        scene["sample_render"] = save_example_view(BUILDERS[scene["name"]]())

    data = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "jax_version": jax.__version__,
            "gsplat_version": gsplat.__version__,
            "torch_version": torch.__version__,
            "warmup": WARMUP,
            "iters": ITERS,
            "repeat": REPEAT,
            "batches": BATCHES,
            "metric": "best-of-repeat mean of one forward render",
            "memory_note": "splax peak combines the JAX and Warp memory sum",
        },
        "scenes": scenes,
    }
    out = OUT_DIR / "benchmark_suite.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
