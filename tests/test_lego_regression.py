"""Lego render-quality regression gate (pure splax, no external reference).

Renders the pretrained lego splat (``data/scenes/lego.ply``, ~313k gaussians,
SH degree 0) at three held-out test poses and asserts the PSNR against the
ground-truth images stays at or above the established floor. This is the
correctness gate for the pretrained lego scene, reproduced here
without any CUDA reference. splax renders the scene itself.

Protocol:
  - poses/intrinsics from ``data/nerf_synthetic/lego/transforms_test.json``,
    frames 0 / 25 / 50,
  - NeRF c2w (OpenGL, -z forward) to w2c viewmat (OpenCV, +z forward) via the
    diag(1, -1, -1, 1) flip then inverse,
  - focal length from ``camera_angle_x``, principal point at the image center,
    ``glob_scale=1.0``, ``clip_thresh=0.01``,
  - white background, ground truth alpha-composited onto white.

The floors are the established reference values (30.89 / 31.43 / 32.08 dB) minus a
0.05 dB slack for float32 blend-order jitter. splax currently reproduces them to
better than 0.01 dB. The scene and the pretrained ply are downloaded from huggingface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

import splax

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# frame index to the established PSNR floor (dB) at that held-out test pose.
KNOWN_PSNR = {0: 30.89, 25: 31.43, 50: 32.08}
SLACK = 0.05


@pytest.mark.parametrize("frame_idx", [0, 25, 50])
def test_lego_render_psnr_regression(
    frame_idx: int, lego_meta: dict, lego_view: Callable[[str], np.ndarray], lego_ply: Path
) -> None:
    means, scales, quats, colors, opac = splax.io.load_ply(lego_ply)

    frame = lego_meta["frames"][frame_idx]
    gt = lego_view(frame["file_path"])
    H, W = gt.shape[:2]
    gt = gt.astype(np.float32) / 255.0
    gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])  # composite on white
    viewmat = splax.utils.nerf_camera(frame["transform_matrix"])

    ff = 0.5 * W / np.tan(0.5 * lego_meta["camera_angle_x"])
    img, _ = splax.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=jnp.asarray(viewmat),
        background=jnp.ones(3),
        img_shape=(H, W),
        f=(float(ff), float(ff)),
        c=(W // 2, H // 2),
    )
    img = np.clip(np.asarray(img), 0.0, 1.0)
    psnr = -10.0 * np.log10(float(np.mean((img - gt) ** 2)))

    floor = KNOWN_PSNR[frame_idx] - SLACK
    assert psnr >= floor, f"frame {frame_idx} PSNR {psnr:.3f} dB below floor {floor:.3f}"
