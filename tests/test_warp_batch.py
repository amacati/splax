"""Native batching: jax.vmap over the pure-Warp splax pipeline.

splax.project / splax.rasterize / splax.render carry vmap_method="expand_dims" and
launch a single grid over the whole batch (gsplat-style, no host loop). These tests
assert that a vmapped call equals ``jnp.stack`` of the per-element unbatched calls.

Batching must NOT change blend order within an image. The global sort packs the
image id above the tile bits, so each image's (tile, depth) ordering is identical to
the unbatched sort, and the front-to-back accumulation is bit-identical. We therefore
require **bit-exact** equality (array_equal), not just an allclose tolerance.
"""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

import splax
import splax._cache as _cache
import splax._rasterize._sort._sort as _sort


class _KW(TypedDict):
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


class _ProjKW(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


@pytest.fixture(autouse=True)
def _faithful_64bit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the 64-bit sort key for the bit-exact batch-native assertions.

    Batch-native == stack-of-unbatched is bit-exact only for the 64-bit key, whose
    per-image (tile, depth) order is independent of B. The default packed 32-bit
    key sizes its depth field as 31 - image_bits - tile_bits, which shrinks as B
    grows, so a batched render quantizes depth slightly coarser than the B=1
    reference and matches only up to a perceptual bound (asserted in
    test_render_vmap_packed_matches_stack).
    """
    monkeypatch.setattr(_sort, "_use_32bit_keys", lambda depth_bits: False)


def _rand_scene(
    n: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opacities = jax.random.uniform(k[4], (n, 1))
    return means, scales, quats, colors, opacities


def _viewmat(dx: float, dy: float = -0.1, dz: float = 5.0) -> jax.Array:
    return jnp.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]], jnp.float32)


N = 8_000
H = W = 128
KW: _KW = {
    "background": jnp.zeros(3),
    "img_shape": (H, W),
    "f": (float(H), float(H)),
    "c": (W // 2, H // 2),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}
PROJ_KW: _ProjKW = {
    "img_shape": (H, W),
    "f": (float(H), float(H)),
    "c": (W // 2, H // 2),
    "glob_scale": 1.0,
    "clip_thresh": 0.01,
}
VIEWS = jnp.stack([_viewmat(0.0), _viewmat(0.3), _viewmat(-0.2)])  # B=3


def _render(
    m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array, vm: jax.Array
) -> jax.Array:
    return splax.render(m, s, q, c, o, viewmat=vm, **KW)[0]


def test_project_vmap_over_viewmats() -> None:
    """vmap(project) over B viewmats: per-image outputs are bit-exact vs unbatched.

    All outputs except cum_tiles_hit are per-gaussian and batch-invariant, so they
    match the stacked unbatched projections exactly. cum_tiles_hit is intentionally a
    *global* inclusive prefix sum across the whole batch (gsplat's single-sort
    layout: every image's intersections are laid out contiguously), so it equals the
    global cumsum of the flattened num_tiles_hit rather than the per-image cumsum.
    """
    m, s, q, _c, o = _rand_scene(N, seed=1)

    def f(vm: jax.Array) -> tuple:
        return splax.project(m, s, q, vm, opacities=o, **PROJ_KW)

    B = VIEWS.shape[0]
    batched = jax.vmap(f)(VIEWS)
    # outputs 0..4 = xys, depths, radii, conics, num_tiles_hit, bit-exact per image
    for i in range(B):
        ref = f(VIEWS[i])
        for k in range(5):
            np.testing.assert_array_equal(np.asarray(batched[k][i]), np.asarray(ref[k]))
    # cum_tiles_hit (output 5) is the global inclusive scan of flattened num_tiles_hit
    nth = np.asarray(batched[4]).reshape(-1).astype(np.int64)
    cum = np.asarray(batched[5]).reshape(-1).astype(np.int64)
    np.testing.assert_array_equal(cum, np.cumsum(nth))


def test_render_vmap_over_viewmats() -> None:
    """vmap(render) over B=3 viewmats == jnp.stack of 3 unbatched renders."""
    m, s, q, c, o = _rand_scene(N, seed=2)
    ref = jnp.stack([_render(m, s, q, c, o, VIEWS[i]) for i in range(3)])
    out = jax.vmap(lambda vm: _render(m, s, q, c, o, vm))(VIEWS)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_vmap_over_splats() -> None:
    """Vmap over batched splat params (shared viewmat) == stacked unbatched."""
    scenes = [_rand_scene(N, seed=s) for s in (3, 4, 5)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    vm = _viewmat(0.1)
    ref = jnp.stack([_render(*scenes[i], vm) for i in range(3)])
    out = jax.vmap(lambda m, s, q, c, o: _render(m, s, q, c, o, vm))(mb, sb, qb, cb, ob)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_vmap_mixed_both_batched() -> None:
    """Vmap over both splat params AND viewmats (mixed dims in one call)."""
    scenes = [_rand_scene(N, seed=s) for s in (6, 7, 8)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    ref = jnp.stack([_render(*scenes[i], VIEWS[i]) for i in range(3)])
    out = jax.vmap(_render)(mb, sb, qb, cb, ob, VIEWS)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_nested_vmap_grid() -> None:
    """Nested vmap over an A splats x B viewmats grid == the A x B double loop."""
    scenes = [_rand_scene(N, seed=s) for s in (3, 4, 5)]
    mb, sb, qb, cb, ob = (jnp.stack([sc[i] for sc in scenes]) for i in range(5))
    B = VIEWS.shape[0]
    ref = jnp.stack(
        [jnp.stack([_render(*scenes[a], VIEWS[b]) for b in range(B)]) for a in range(len(scenes))]
    )
    grid = jax.vmap(lambda m, s, q, c, o: jax.vmap(lambda vm: _render(m, s, q, c, o, vm))(VIEWS))
    out = grid(mb, sb, qb, cb, ob)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_nested_shared_splat_transforms() -> None:
    """Nested vmap, shared splat, viewmat on both axes, transforms on the outer axis only."""
    m, s, q, c, o = _rand_scene(N, seed=6)
    slices = ((0, N // 2), (N // 2, N))
    n_worlds, n_cams = 2, 3
    vms = jnp.stack(
        [
            jnp.stack([_viewmat(0.1 * w + 0.03 * cam) for cam in range(n_cams)])
            for w in range(n_worlds)
        ]
    )
    tfs = jnp.stack(
        [jnp.broadcast_to(jnp.eye(4).at[0, 3].set(0.02 * w), (2, 4, 4)) for w in range(n_worlds)]
    )

    def one(vm: jax.Array, tf: jax.Array) -> jax.Array:
        tf_kw = {"gaussian_transforms": tf, "gaussian_slices": slices}
        return splax.render(m, s, q, c, o, viewmat=vm, **KW, **tf_kw)[0]

    ref = jnp.stack(
        [jnp.stack([one(vms[w, cam], tfs[w]) for cam in range(n_cams)]) for w in range(n_worlds)]
    )
    out = jax.vmap(lambda vmw, tfw: jax.vmap(lambda vm: one(vm, tfw))(vmw))(vms, tfs)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_nested_vmap_grad() -> None:
    """Nested vmap over jax.grad of the differentiable render == the double loop of gradients."""
    m, s, q, c, o = _rand_scene(N, seed=12)
    means = jnp.stack([m, m * 1.01])

    def loss(mm: jax.Array, vm: jax.Array) -> jax.Array:
        return _render(mm, s, q, c, o, vm).sum()

    ref = jnp.stack(
        [
            jnp.stack([jax.grad(loss)(means[a], VIEWS[b]) for b in range(VIEWS.shape[0])])
            for a in range(means.shape[0])
        ]
    )
    out = jax.vmap(lambda mm: jax.vmap(lambda vm: jax.grad(loss)(mm, vm))(VIEWS))(means)
    assert out.shape == ref.shape
    # gradient magnitudes reach ~1e3 and batched reductions reorder, so allclose not bit-exact
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-2)


def test_render_jit_vmap() -> None:
    """jit(vmap(render)) matches unbatched."""
    m, s, q, c, o = _rand_scene(N, seed=9)
    ref = jnp.stack([_render(m, s, q, c, o, VIEWS[i]) for i in range(3)])
    fn = jax.jit(jax.vmap(lambda vm: _render(m, s, q, c, o, vm)))
    np.testing.assert_array_equal(np.asarray(fn(VIEWS)), np.asarray(ref))


def test_render_vmap_b1_equals_unbatched() -> None:
    """B=1 vmap is identical to the plain unbatched render."""
    m, s, q, c, o = _rand_scene(N, seed=10)
    vm = _viewmat(0.15)
    unb = _render(m, s, q, c, o, vm)
    b1 = jax.vmap(lambda v: _render(m, s, q, c, o, v))(vm[None])
    assert b1.shape == (1,) + unb.shape
    np.testing.assert_array_equal(np.asarray(b1[0]), np.asarray(unb))


def test_render_vmap_larger_batch_and_res() -> None:
    """B=8 at a larger resolution: image-id/tile-id key packing stays correct."""
    m, s, q, c, o = _rand_scene(12_000, seed=11)
    B, hh, ww = 8, 512, 512
    kw: _KW = {**KW, "img_shape": (hh, ww), "f": (float(hh), float(hh)), "c": (ww // 2, hh // 2)}
    views = jnp.stack([_viewmat(0.1 * i) for i in range(B)])
    ref = jnp.stack([splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(B)])
    out = jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def test_render_vmap_packed_matches_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match packed batched render output against stacked unbatched output."""
    monkeypatch.setattr(_sort, "_use_32bit_keys", lambda depth_bits: depth_bits >= 16)
    m, s, q, c, o = _rand_scene(12_000, seed=11)
    B, hh, ww = 8, 512, 512
    kw: _KW = {**KW, "img_shape": (hh, ww), "f": (float(hh), float(hh)), "c": (ww // 2, hh // 2)}
    views = jnp.stack([_viewmat(0.1 * i) for i in range(B)])
    ref = jnp.stack(
        [splax.render(m, s, q, c, o, viewmat=views[i], **kw)[0] for i in range(B)]
    )  # B=1 renders each pack with depth_bits=21
    splax.clear_cache()
    out = np.asarray(
        jax.jit(jax.vmap(lambda vm: splax.render(m, s, q, c, o, viewmat=vm, **kw)[0]))(views)
    )  # B=8 render packs with depth_bits=18 (image 3 + tile 10)
    assert _cache._scratch_cache[str(wp.get_device("cuda:0"))]["isect_dtype"] == wp.int32
    d = np.abs(out - np.asarray(ref))
    mse = float(np.mean((out - np.asarray(ref)) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert d.max() < 0.05, f"packed batched vs stacked max abs diff {d.max():.2e}"
    assert psnr > 65, f"packed batched vs stacked PSNR only {psnr:.1f} dB"
    splax.clear_cache()
