"""Lego render-quality regression gate (pure splax, no external reference).

Renders the pretrained lego splat (``data/scenes/lego.ply``, ~313k gaussians,
SH degree 0) at three held-out test poses and asserts the PSNR against the
ground-truth images stays at or above the established floor. This is the
correctness gate that used to live in ``scripts/render_lego.py``, reproduced here
without any CUDA reference: splax renders the scene itself.

Protocol (identical to the historical render_lego.py):
  - poses/intrinsics from ``data/nerf_synthetic/lego/transforms_test.json``,
    frames 0 / 25 / 50,
  - NeRF c2w (OpenGL, -z forward) -> w2c viewmat (OpenCV, +z forward) via the
    diag(1, -1, -1, 1) flip then inverse,
  - focal length from ``camera_angle_x``, principal point at the image center,
    ``glob_scale=1.0``, ``clip_thresh=0.01``,
  - white background, ground truth alpha-composited onto white.

The floors are the established reference values (30.89 / 31.43 / 32.08 dB) minus a
0.05 dB slack for float32 blend-order jitter. splax currently reproduces them to
better than 0.01 dB. Skips gracefully when the (unshipped) dataset is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest
import imageio.v3 as iio

import splax

ROOT = Path(__file__).resolve().parents[1]
LEGO = ROOT / "data/nerf_synthetic/lego"
PLY = ROOT / "data/scenes/lego.ply"

# frame index -> established PSNR floor (dB) at that held-out test pose.
KNOWN_PSNR = {0: 30.89, 25: 31.43, 50: 32.08}
SLACK = 0.05


def _nerf_camera(frame: dict[str, object]) -> np.ndarray:
    """NeRF c2w (OpenGL, -z forward) -> w2c viewmat (OpenCV, +z forward)."""
    c2w = np.array(frame["transform_matrix"], np.float64)
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w).astype(np.float32)


@pytest.mark.skipif(
    not (LEGO.exists() and PLY.exists()), reason="lego dataset / ply missing"
)
@pytest.mark.parametrize("frame_idx", [0, 25, 50])
def test_lego_render_psnr_regression(frame_idx: int) -> None:
    meta = json.loads((LEGO / "transforms_test.json").read_text())
    means, scales, quats, colors, opac = splax.load_ply(PLY)

    frame = meta["frames"][frame_idx]
    gt = iio.imread(LEGO / (frame["file_path"].lstrip("./") + ".png"))
    H, W = gt.shape[:2]
    gt = gt.astype(np.float32) / 255.0
    gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])  # composite on white

    ff = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
    img = splax.inference.render(
        means, scales, quats, colors, opac,
        viewmat=jnp.asarray(_nerf_camera(frame)),
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(float(ff), float(ff)),
        c=(W // 2, H // 2),
        glob_scale=1.0,
        clip_thresh=0.01,
    )
    img = np.clip(np.asarray(img), 0.0, 1.0)
    psnr = -10.0 * np.log10(float(np.mean((img - gt) ** 2)))

    floor = KNOWN_PSNR[frame_idx] - SLACK
    assert psnr >= floor, f"frame {frame_idx} PSNR {psnr:.3f} dB below floor {floor:.3f}"
