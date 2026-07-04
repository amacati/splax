"""Phase 6c: PLY export round-trips and the inference/training split parity.

``splax.io.write_ply`` must be the exact inverse of ``splax.io.load_ply``.
Two round-trips assert that:

1. random render-space splats -> write_ply -> load_ply reproduce the inputs, and
2. a real scene (lego.ply) -> write copy -> reload renders to the same image
   (fit-free: no training, just the load/write/load/render loop),

plus that ``splax.inference.render`` and ``splax.training.render`` produce the
identical forward image (the Phase 6c split is numerically zero-cost).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import splax
from splax import load_ply


def lookat_viewmats(center: np.ndarray, radius: float, num_views: int) -> jax.Array:
    """World-to-camera matrices orbiting ``center`` (OpenCV convention, +z forward)."""
    mats = []
    for i in range(num_views):
        az = 2 * np.pi * i / num_views
        eye = center + radius * np.array([np.sin(az), 0.3, np.cos(az)])
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])  # rows: cam axes in world
        t = -R @ eye
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, t
        mats.append(m)
    return jnp.asarray(np.stack(mats), jnp.float32)


LEGO_PLY = Path("data/scenes/lego.ply")


class _RenderKw(TypedDict):
    background: jax.Array
    glob_scale: float
    clip_thresh: float


RENDER_KW: _RenderKw = {
    "background": jnp.ones(3),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}


def _render(
    splats: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    viewmat: jax.Array,
    H: int,
    W: int,
) -> jax.Array:
    means, scales, quats, colors, opac = splats
    return splax.inference.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )


def _random_splats(
    seed: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    opac = rng.uniform(0.05, 0.95, (n, 1)).astype(np.float32)
    return means, scales, quats, colors, opac


def test_write_ply_is_load_ply_inverse(tmp_path: Path) -> None:
    """Random splats -> write_ply -> load_ply reproduce the render-space inputs."""
    means, scales, quats, colors, opac = _random_splats(seed=0, n=5000)
    out = tmp_path / "rand.ply"
    splax.write_ply(out, means, scales, quats, colors, opac)

    lm, ls, lq, lc, lo = (np.asarray(x) for x in load_ply(out))

    np.testing.assert_allclose(lm, means, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ls, scales, rtol=1e-5, atol=1e-6)
    # quats are normalized on both sides; compare up to sign is unnecessary since
    # write_ply preserves the stored raw quat direction and load re-normalizes.
    np.testing.assert_allclose(lq, quats, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lc, colors, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lo, opac, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not LEGO_PLY.exists(), reason="lego.ply reference not present")
def test_ply_render_roundtrip(tmp_path: Path) -> None:
    """Fit-free: load lego.ply -> write copy -> reload -> identical render."""
    splats = load_ply(LEGO_PLY)
    copy = tmp_path / "lego_copy.ply"
    splax.write_ply(copy, *splats)
    splats2 = load_ply(copy)

    center = np.asarray(splats[0].mean(axis=0))
    radius = float(
        np.percentile(np.linalg.norm(np.asarray(splats[0]) - center, axis=-1), 90)
    )
    viewmat = lookat_viewmats(center, radius, 1)[0]

    H = W = 200
    img1 = np.asarray(_render(splats, viewmat, H, W))
    img2 = np.asarray(_render(splats2, viewmat, H, W))
    # Activation round-trip (log/exp, logit/sigmoid) is ULP-level; the render is
    # essentially identical. Splatting's hard 1/255 cull can flip a handful of
    # pixels, so bound by max abs diff rather than requiring bit-exactness.
    assert np.max(np.abs(img1 - img2)) < 1e-3


@pytest.mark.skipif(not LEGO_PLY.exists(), reason="lego.ply reference not present")
def test_inference_equals_training_forward() -> None:
    """The Phase 6c split is numerically zero-cost: identical forward image."""
    splats = load_ply(LEGO_PLY)
    center = np.asarray(splats[0].mean(axis=0))
    radius = float(
        np.percentile(np.linalg.norm(np.asarray(splats[0]) - center, axis=-1), 90)
    )
    viewmat = lookat_viewmats(center, radius, 1)[0]
    means, scales, quats, colors, opac = splats

    H = W = 200
    inf_img = splax.inference.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )
    train_img, _ = splax.training.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )
    np.testing.assert_array_equal(np.asarray(inf_img), np.asarray(train_img))
