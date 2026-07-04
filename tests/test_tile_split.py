"""Phase 8t: split-heavy-tile load-balanced inference blend.

The split path (SPLAX_TILE_SPLIT) rebalances heavy tile bins across blocks via
associative segment compositing. It is inference-only and must agree with the
unsplit blend to a few ULP (segment adds reorder across boundaries only), and be
byte-identical when no tile exceeds the split threshold.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import warp as wp

from splax import inference
from splax import _rasterize as _R


def _scene(
    n: int, seed: int = 0, concentrate: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    spread = jnp.array([0.05, 0.05, 1.0]) if concentrate else jnp.array([1.0, 1.0, 1.0])
    means = jax.random.normal(k[0], (n, 3)) * spread
    scales = jax.random.uniform(k[1], (n, 3), minval=0.01, maxval=0.06)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    # semi-transparent so heavy tiles do NOT saturate early -> segments do real work
    opac = jax.random.uniform(k[4], (n, 1), minval=0.02, maxval=0.6)
    bg = jax.random.uniform(k[5], (3,))
    vm = jnp.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 5.0], [0, 0, 0, 1]], jnp.float32)
    return means, scales, quats, colors, opac, bg, vm


def _render(
    scene: tuple[jax.Array, ...],
    H: int,
    W: int,
    split: bool,
    monkeypatch: pytest.MonkeyPatch,
    threshold: int | None = None,
) -> np.ndarray:
    monkeypatch.setattr(_R, "_TILE_SPLIT", split)
    if threshold is not None:
        monkeypatch.setattr(_R, "_SPLIT_THRESHOLD", threshold)
    m, s, q, c, o, bg, vm = scene
    fn = jax.jit(lambda: inference.render(
        m, s, q, c, o, viewmat=vm, background=bg, img_shape=(H, W),
        f=(float(H), float(H)), c=(W // 2, H // 2), glob_scale=1.0, clip_thresh=0.01))
    img = np.asarray(fn())
    jax.block_until_ready(img)
    return img


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


@pytest.mark.parametrize("concentrate", [False, True])
def test_split_agrees_with_unsplit(monkeypatch: pytest.MonkeyPatch, concentrate: bool) -> None:
    """Split blend must match the unsplit blend to > 80 dB (segment-boundary ULP)."""
    scene = _scene(400_000, concentrate=concentrate)
    ref = _render(scene, 256, 256, False, monkeypatch)
    got = _render(scene, 256, 256, True, monkeypatch)
    assert got.shape == ref.shape == (256, 256, 3)
    psnr = _psnr(ref, got)
    assert psnr > 80.0, f"split vs unsplit PSNR {psnr:.1f} dB below 80 dB"
    assert np.abs(ref - got).max() < 1e-3


def test_heavy_scene_triggers_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """The concentrated scene must actually exercise the split path (else vacuous)."""
    scene = _scene(400_000, concentrate=True)
    _render(scene, 256, 256, True, monkeypatch, threshold=8000)
    sc = _R._split_scratch_cache.get("split:" + str(wp.get_device("cuda:0")))
    assert sc is not None and int(sc["counters"].numpy()[0]) > 0


def test_not_triggered_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a threshold no tile reaches, the split path is byte-for-byte the unsplit
    image (only the unsplit-light kernel runs; its blend math/order is identical)."""
    scene = _scene(50_000)
    ref = _render(scene, 256, 256, False, monkeypatch)
    got = _render(scene, 256, 256, True, monkeypatch, threshold=10_000_000)
    assert np.array_equal(ref, got)


def test_split_batched_vmap(monkeypatch: pytest.MonkeyPatch) -> None:
    """vmap over viewmats renders each view as the split path would unbatched."""
    monkeypatch.setattr(_R, "_TILE_SPLIT", True)
    m, s, q, c, o, bg, vm = _scene(300_000, concentrate=True)
    vms = jnp.stack([vm, vm.at[0, 3].set(0.1)])
    H = W = 256

    def one(v: jax.Array) -> jax.Array:
        return inference.render(m, s, q, c, o, viewmat=v, background=bg, img_shape=(H, W),
                                f=(float(H), float(H)), c=(W // 2, H // 2),
                                glob_scale=1.0, clip_thresh=0.01)

    batched = np.asarray(jax.jit(jax.vmap(one))(vms))
    assert batched.shape == (2, H, W, 3)
    for i in range(2):
        single = np.asarray(jax.jit(one)(vms[i]))
        assert _psnr(single, batched[i]) > 80.0
