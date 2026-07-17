"""Benchmark suite comparing splax against gsplat on one training step.

The measured quantity is a full forward plus backward pass. Both frameworks render the camera batch,
take a scalar L2 loss against a fixed random target image stack, and compute gradients with respect
to the five splat parameter arrays (means, scales, quats, colors, opacities). splax runs a jitted
``jax.value_and_grad`` over a vmapped ``splax.training.render``, gsplat runs ``loss.backward()``
through its native camera-batch rasterization with a CUDA synchronize inside the timed call.

Writes ``reports/bench_backward.json``.

    pixi run -e tests python benchmark/bench_backward.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import gsplat
import jax
import jax.numpy as jnp
import numpy as np
import torch
import warp as wp
from bench_forward import (
    BATCHES,
    BUILDERS,
    CLIP_THRESH,
    EPS2D,
    ITERS,
    SEED,
    WARMUP,
    Scenario,
    bench,
)

import splax

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports"


def _target(sc: Scenario, batch: int) -> np.ndarray:
    """Fixed random target image stack shared by both frameworks."""
    rng = np.random.default_rng(SEED)
    return rng.uniform(size=(batch, sc.res, sc.res, 3)).astype(np.float32)


def make_splax_step(sc: Scenario, batch: int) -> tuple[Callable[[], object], Callable]:
    """Build a jitted value_and_grad of the L2 loss over ``batch`` viewmats.

    Returns the timed call and the jitted step itself so the caller can read its jit cache size.
    """
    means, scales, quats, colors, opacities, background = sc.scene
    res, focal = sc.res, sc.focal
    views = jnp.asarray(sc.viewmats[:batch])
    target = jnp.asarray(_target(sc, batch))

    def loss_fn(params: tuple[jax.Array, ...], vm: jax.Array) -> jax.Array:
        m, s, q, col, o = params

        def one(v: jax.Array) -> jax.Array:
            img, _ = splax.training.render(
                m,
                s,
                q,
                col,
                o,
                viewmat=v,
                background=background,
                img_shape=(res, res),
                f=(focal, focal),
                c=(res / 2, res / 2),
                clip_thresh=CLIP_THRESH,
            )
            return img

        imgs = jax.vmap(one)(vm)
        return jnp.mean((imgs - target) ** 2)

    step = jax.jit(jax.value_and_grad(loss_fn))
    params = (means, scales, quats, colors, opacities)
    return lambda: jax.block_until_ready(step(params, views)), step


def make_gsplat_step(sc: Scenario, batch: int) -> Callable[[], object]:
    """Build a gsplat training step with the same loss and gradient set."""
    means, scales, quats, colors, opacities, background = sc.scene
    res, focal = sc.res, sc.focal

    def tt(x: jax.Array, grad: bool = True) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x, np.float32), dtype=torch.float32, device="cuda")
        return t.requires_grad_(grad) if grad else t

    means_t = tt(means)
    quats_t = tt(quats)
    scales_t = tt(scales)
    opac_t = tt(jnp.reshape(opacities, (-1,)))
    colors_t = tt(colors)
    bg_t = tt(background, grad=False).reshape(3)
    params = [means_t, quats_t, scales_t, opac_t, colors_t]
    k = np.array([[focal, 0.0, res / 2], [0.0, focal, res / 2], [0.0, 0.0, 1.0]], np.float32)
    ks_t = torch.as_tensor(k, device="cuda")[None].repeat(batch, 1, 1)
    views_t = torch.as_tensor(sc.viewmats[:batch], dtype=torch.float32, device="cuda")
    target_t = torch.as_tensor(_target(sc, batch), device="cuda")

    def run() -> None:
        for p in params:
            p.grad = None
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
        img = out + (1.0 - alpha) * bg_t
        loss = ((img - target_t) ** 2).mean()
        loss.backward()
        torch.cuda.synchronize()

    return run


def nvml_used_bytes() -> int:
    """Current GPU memory of this process from nvidia-smi.

    Ground-truth cross-check for the allocator peaks. The Warp mempool watermark and the JAX peak
    each miss what the other pool holds, and neither sees CUDA context or cub workspace memory.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    pid = str(os.getpid())
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == pid:
            return int(parts[1]) * 1024 * 1024
    return 0


def run_scenario(sc: Scenario) -> dict:
    """Sweep the batch for one scenario, timing both frameworks and their peak memory."""
    n = sc.scene[0].shape[0]
    logger.info(f"\n== {sc.name}: {n:,} gaussians, {sc.res}x{sc.res} ==")
    logger.info(f"{'batch':>6} {'splax ms':>9} {'gsplat ms':>10} {'sp MB':>8} {'gs MB':>8}")

    device = wp.get_device()
    rows = []
    for batch in BATCHES:
        splax_call, step = make_splax_step(sc, batch)
        for _ in range(WARMUP):
            splax_call()
        cache = step._cache_size()  # ty: ignore[unresolved-attribute]
        assert cache == 1, f"expected 1 splax jit cache entry, got {cache}"

        splax_ms = bench(splax_call, ITERS) * 1e3
        jax_peak = int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0))
        warp_peak = int(wp.get_mempool_used_mem_high(device))
        splax_peak = jax_peak + warp_peak

        gsplat_call = make_gsplat_step(sc, batch)
        for _ in range(WARMUP):
            gsplat_call()
        torch.cuda.reset_peak_memory_stats()
        gsplat_ms = bench(gsplat_call, ITERS) * 1e3
        gsplat_peak = int(torch.cuda.max_memory_allocated())

        rows.append(
            {
                "batch": batch,
                "splax": {
                    "time_ms": splax_ms,
                    "throughput_ips": batch / (splax_ms / 1e3),
                    "peak_bytes": splax_peak,
                    "jax_peak_bytes": jax_peak,
                    "warp_peak_bytes": warp_peak,
                    "nvml_used_bytes": nvml_used_bytes(),
                },
                "gsplat": {
                    "time_ms": gsplat_ms,
                    "throughput_ips": batch / (gsplat_ms / 1e3),
                    "peak_bytes": gsplat_peak,
                    "nvml_used_bytes": nvml_used_bytes(),
                },
                "speedup_gsplat_over_splax": gsplat_ms / splax_ms,
                "mem_ratio_splax_over_gsplat": splax_peak / gsplat_peak if gsplat_peak else None,
            }
        )
        logger.info(
            f"{batch:>6} {splax_ms:>9.3f} {gsplat_ms:>10.3f} "
            f"{splax_peak / 1e6:>8.1f} {gsplat_peak / 1e6:>8.1f}"
        )

    return {
        "name": sc.name,
        "n_gaussians": int(n),
        "img_shape": [sc.res, sc.res],
        "focal": sc.focal,
        "rows": rows,
    }


def run_worker(scene: str, frag: Path) -> None:
    """Benchmark one scenario in this process and write its result dict to ``frag``."""
    result = run_scenario(BUILDERS[scene]())
    frag.write_text(json.dumps(result))


def main() -> None:
    """Run every scenario in an isolated subprocess and merge the JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", nargs="+", default=["synthetic", "lego", "hf"], help="subset of scenes"
    )
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
            "warp_version": wp.__version__,
            "warmup": WARMUP,
            "iters": ITERS,
            "batches": BATCHES,
            "metric": (
                "one training step (forward + backward of a scalar L2 loss against a "
                "fixed random target, grads wrt means/scales/quats/colors/opacities), "
                "best-of-repeat mean per call"
            ),
            "memory_note": (
                "splax peak is the JAX allocator process-cumulative peak plus the Warp "
                "mempool high watermark, both cumulative over the ascending batch "
                "sweep. gsplat is the torch allocator peak, reset per batch. Separate "
                "pools, not strictly comparable in absolute terms. nvml_used_bytes is "
                "the process's current device memory from nvidia-smi after the batch, "
                "a ground-truth cross-check that also covers the CUDA context."
            ),
        },
        "scenarios": scenarios,
    }
    out = OUT_DIR / "bench_backward.json"
    out.write_text(json.dumps(data, indent=2))
    logger.info(f"\nwrote {out}")


if __name__ == "__main__":
    main()
