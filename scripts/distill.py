"""Distill a large trained splat into a smaller student, end to end.

Renders a teacher ``.ply`` from many synthetic in-scene poses and retrains a capped
student on that dense synthetic dataset (``splax.distillation``), then measures the
student on the REAL held-out COLMAP views of the scene the teacher was trained on --
never on the synthetic views. The teacher's own held-out PSNR is reported as the
compression ceiling.

The teacher ``.ply`` lives in the scene's similarity-normalized frame (the frame
``scripts/train_colmap.py`` exports); this script re-derives that exact frame from the
same COLMAP scene (normalization is downscale-independent), so the teacher, the
synthetic poses, and the eval cameras all share one coordinate system.

Usage:
  python scripts/distill.py --teacher-ply data/scenes/drone_1p5M.ply \
      --data data/drone --n-student 150000 --n-views 500 --steps 3000 \
      --depth-lambda 0.5 --downscale 2 --out-ply data/scenes/drone_150k_distill.ply \
      --out-json results/phase8u_distill_150k.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import splax
from train_colmap import load_scene, psnr  # reuse the COLMAP loader + metric


def _student_render(
    student: dict[str, jax.Array],
    vm: jax.Array,
    H: int,
    W: int,
    intr: tuple[float, float, float, float],
) -> jax.Array:
    fx, fy, cx, cy = intr
    return splax.inference.render(
        student["means"],
        student["scales"],
        student["quats"],
        student["colors"],
        student["opacities"],
        viewmat=vm,
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(fx, fy),
        c=(cx, cy),
        glob_scale=1.0,
        clip_thresh=0.01,
    )


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ply", default="data/scenes/drone_1p5M.ply")
    ap.add_argument(
        "--data", default="data/drone", help="COLMAP scene the teacher was trained on"
    )
    ap.add_argument("--sparse-model", type=int, default=0)
    ap.add_argument("--n-student", type=int, default=150_000)
    ap.add_argument("--n-views", type=int, default=500, help="synthetic teacher views")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--depth-lambda",
        type=float,
        default=0.0,
        help="dense teacher-depth distillation weight (0 = off)",
    )
    ap.add_argument("--init", choices=["prune", "random"], default="prune")
    ap.add_argument(
        "--downscale", type=int, default=2, help="eval (real-view) downscale"
    )
    ap.add_argument(
        "--synth-downscale",
        type=int,
        default=4,
        help="synthetic render downscale (coarser keeps host memory modest)",
    )
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-lr", type=float, default=5e5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out-ply", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    # --- scene (eval intrinsics + real held-out views), same normalized frame ----
    scene = load_scene(
        args.data,
        args.downscale,
        args.eval_every,
        seed=args.seed,
        sparse_model=args.sparse_model,
    )
    H, W, intr = scene["H"], scene["W"], scene["intr"]
    eval_vms = [
        jnp.asarray(scene["eval_vms"][i]) for i in range(len(scene["eval_names"]))
    ]
    eval_imgs = [scene["eval_imgs"][i] for i in range(len(eval_vms))]
    eval_idxs = list(range(min(args.n_eval, len(eval_imgs))))
    train_vms = np.asarray(scene["train_vms"])

    # --- synthetic render intrinsics (scaled to synth-downscale) -----------------
    ratio = args.downscale / args.synth_downscale
    Hs, Ws = max(8, round(H * ratio)), max(8, round(W * ratio))
    fs = (intr[0] * ratio, intr[1] * ratio)
    cs = (intr[2] * ratio, intr[3] * ratio)
    print(
        f"eval {W}x{H} (ds{args.downscale}) | synthetic {Ws}x{Hs} (ds{args.synth_downscale}), "
        f"{args.n_views} views -> {args.n_student} student gaussians, init={args.init}, "
        f"depth_lambda={args.depth_lambda}"
    )

    # --- teacher (render-space) + ceiling ---------------------------------------
    tm, ts, tq, tc, to = splax.load_ply(args.teacher_ply)
    teacher = {"means": tm, "scales": ts, "quats": tq, "colors": tc, "opacities": to}
    print(f"teacher {tm.shape[0]} gaussians from {args.teacher_ply}")
    teacher_pf = [
        psnr(_student_render(teacher, eval_vms[i], H, W, intr), eval_imgs[i])
        for i in eval_idxs
    ]
    teacher_psnr = float(np.mean(teacher_pf))
    print(
        f"teacher held-out PSNR (ceiling): {teacher_psnr:.2f} dB {[round(x, 2) for x in teacher_pf]}"
    )

    def eval_hook(student: dict[str, jax.Array]) -> float:
        return float(
            np.mean(
                [
                    psnr(
                        _student_render(student, eval_vms[i], H, W, intr), eval_imgs[i]
                    )
                    for i in eval_idxs
                ]
            )
        )

    # --- distill -----------------------------------------------------------------
    info: dict = {}
    t0 = time.perf_counter()
    student = splax.distillation.distill(
        teacher,
        args.n_student,
        img_shape=(Hs, Ws),
        f=fs,
        c=cs,
        n_views=args.n_views,
        steps=args.steps,
        depth_lambda=args.depth_lambda,
        init=args.init,
        seed=args.seed,
        viewmats=train_vms,
        batch=args.batch_size,
        noise_lr=args.noise_lr,
        log_every=args.log_every,
        eval_hook=eval_hook,
        info=info,
    )
    total_wall = time.perf_counter() - t0

    per_frame = [
        psnr(_student_render(student, eval_vms[i], H, W, intr), eval_imgs[i])
        for i in eval_idxs
    ]
    final = float(np.mean(per_frame))
    for c in info["curve"]:
        print(
            f"  step {c['step']:5d}  "
            + (f"L1 {c.get('train_l1', '   -')}  " if "train_l1" in c else "")
            + (f"eval {c['eval_psnr']:.2f} dB" if "eval_psnr" in c else "")
        )
    print(f"\nstudent held-out PSNR: {final:.2f} dB {[round(x, 2) for x in per_frame]}")
    print(f"teacher ceiling {teacher_psnr:.2f} dB | gap {teacher_psnr - final:.2f} dB")
    print(f"full pipeline wall: {total_wall:.1f}s (fit {info['wall']:.1f}s)")

    if args.out_ply:
        Path(args.out_ply).parent.mkdir(parents=True, exist_ok=True)
        splax.write_ply(
            args.out_ply,
            student["means"],
            student["scales"],
            student["quats"],
            student["colors"],
            student["opacities"],
        )
        print(f"wrote {args.out_ply}")

    result = {
        "teacher_ply": args.teacher_ply,
        "teacher_n": int(tm.shape[0]),
        "teacher_psnr": teacher_psnr,
        "teacher_per_frame": teacher_pf,
        "n_student": args.n_student,
        "n_views": args.n_views,
        "steps": args.steps,
        "batch": args.batch_size,
        "depth_lambda": args.depth_lambda,
        "init": args.init,
        "downscale": args.downscale,
        "synth_downscale": args.synth_downscale,
        "final": final,
        "per_frame": per_frame,
        "gap": teacher_psnr - final,
        "wall": total_wall,
        "fit_wall": info["wall"],
        "curve": info["curve"],
        "names": [scene["eval_names"][i] for i in eval_idxs],
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out_json}")
    return result


if __name__ == "__main__":
    main()
