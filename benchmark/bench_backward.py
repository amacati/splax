"""Benchmark suite comparing splax against gsplat on one training step.

We measure the full forward plus backward pass. Both frameworks render the camera batch, take a
scalar L2 loss against a fixed random target image stack, and compute gradients with respect to the
five splat parameter arrays.

Results are written to ``reports/bench_backward.json``. Run the benchmark with:

    pixi run -e tests python benchmark/bench_backward.py
"""

from __future__ import annotations

import os
import timeit

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import multiprocessing
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import gsplat
import jax
import jax.numpy as jnp
import numpy as np
import torch
import warp as wp
from bench_forward import BATCHES, BUILDERS, ITERS, REPEAT, WARMUP, Scene

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"


def random_image_batch(sc: Scene, batch: int) -> np.ndarray:
    """Fixed random target image stack shared by both frameworks."""
    return np.random.default_rng(0).uniform(size=(batch, sc.res, sc.res, 3)).astype(np.float32)


def make_splax_step(sc: Scene, batch: int) -> tuple[Callable[[], object], Callable]:
    """Build a splax training step over ``batch`` viewmats.

    Returns the timed call and the jitted step itself.
    """
    *params, background = sc.scene
    views = jnp.asarray(sc.viewmats[:batch])
    target = jnp.asarray(random_image_batch(sc, batch))
    camera = {"background": background, "img_shape": (sc.res, sc.res), "f": (sc.focal, sc.focal)}

    def loss_fn(params: list[jax.Array], viewmats: jax.Array) -> jax.Array:
        imgs, _ = jax.vmap(partial(splax.render, *params, **camera))(viewmat=viewmats)
        return jnp.mean((imgs - target) ** 2)

    step = jax.jit(jax.value_and_grad(loss_fn))
    return lambda: jax.block_until_ready(step(params, views)), step


def make_gsplat_step(sc: Scene, batch: int) -> Callable[[], object]:
    """Build a gsplat training step over ``batch`` viewmats."""
    res, focal = sc.res, sc.focal
    # Older torch versions crash for asarray from jax Arrays
    tensors = [torch.asarray(np.asarray(x, np.float32), device="cuda") for x in sc.scene]
    means_t, scales_t, quats_t, colors_t, opac_t, bg_t = tensors
    opac_t = opac_t.reshape(-1)
    params = [p.requires_grad_() for p in (means_t, quats_t, scales_t, opac_t, colors_t)]
    k = np.array([[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]], np.float32)
    ks_t = torch.as_tensor(k, device="cuda")[None].repeat(batch, 1, 1)
    views_t = torch.as_tensor(sc.viewmats[:batch], device="cuda")
    target_t = torch.as_tensor(random_image_batch(sc, batch), device="cuda")

    def run() -> None:
        for p in params:
            p.grad = None
        out, alpha, _ = gsplat.rasterization(*params, views_t, ks_t, res, res)
        img = out + (1.0 - alpha) * bg_t
        loss = ((img - target_t) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()

    return run


def measure_splax(sc: Scene, batch: int) -> dict:
    """Time the splax step at ``batch`` and read the JAX plus Warp allocator peak."""
    splax_call, step = make_splax_step(sc, batch)
    for _ in range(WARMUP):
        splax_call()
    cache = step._cache_size()  # ty: ignore[unresolved-attribute]
    assert cache == 1, f"expected 1 splax jit cache entry, got {cache}"
    ms = (min(timeit.Timer(splax_call).repeat(repeat=REPEAT, number=ITERS)) / ITERS) * 1e3
    jax_peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"]
    warp_peak = wp.get_mempool_used_mem_high(wp.get_device())
    return {"time_ms": ms, "peak_bytes": jax_peak + warp_peak}


def measure_gsplat(sc: Scene, batch: int) -> dict:
    """Time the gsplat step at ``batch`` and read the torch allocator peak."""
    gsplat_call = make_gsplat_step(sc, batch)
    for _ in range(WARMUP):
        gsplat_call()
    torch.cuda.reset_peak_memory_stats()
    ms = (min(timeit.Timer(gsplat_call).repeat(repeat=REPEAT, number=ITERS)) / ITERS) * 1e3
    return {"time_ms": ms, "peak_bytes": torch.cuda.max_memory_allocated()}


def run_scene(name: str, framework: str) -> dict:
    """Sweep the batch for one scene with one framework, so each framework has the GPU alone."""
    # Multiprocessing with spawn requires reconfiguring logging in the child process
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
        "n_gaussians": n,
        "img_shape": [sc.res, sc.res],
        "focal": sc.focal,
        "rows": rows,
    }


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

    data = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "jax_version": jax.__version__,
            "gsplat_version": gsplat.__version__,
            "torch_version": torch.__version__,
            "warp_version": wp.__version__,
            "warmup": WARMUP,
            "iters": ITERS,
            "batches": BATCHES,
            "metric": "best-of-repeat mean of one forward + backward step",
            "memory_note": "splax peak combines the JAX and Warp memory sum",
        },
        "scenes": scenes,
    }
    out = OUT_DIR / "bench_backward.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
