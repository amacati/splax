"""Post-sync CUDA graph capture, opt-in via SPLAX_POSTSYNC_GRAPHS.

The captured graph replays the whole map/sort/bin/blend sequence and must be
byte-identical to the plain path. The bin kernel reads the real intersection
count from device memory at replay time, so the sentinel-padded sort bucket
never leaks into the bins. Above the count threshold the path falls back to
plain launches, and any scratch reallocation purges the graph cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import splax
import splax._intersect as _isect
import splax._rasterize._kernels as _rast

if TYPE_CHECKING:
    import pytest


class _KW(TypedDict):
    viewmat: jax.Array
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def _scene(n: int, seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3))
    scales = jax.random.uniform(k[1], (n, 3), minval=0.005, maxval=0.05)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1))
    return means, scales, quats, colors, opac


def _kw(H: int, W: int) -> _KW:
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return {
        "viewmat": vm,
        "background": jnp.zeros(3),
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }


def test_graph_replay_matches_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match captured graph output against plain output and replay cache."""
    splats = _scene(20_000, seed=3)
    kw = _kw(256, 256)

    splax.clear_scratch()
    ref = np.asarray(splax.inference.render(*splats, **kw))

    monkeypatch.setattr(_rast, "_POSTSYNC_GRAPHS", True)
    splax.clear_scratch()
    fn = jax.jit(lambda *s: splax.inference.render(*s, **kw))
    first = np.asarray(fn(*splats))
    assert len(_isect._graph_cache) > 0, "no graph was captured"
    n_graphs = len(_isect._graph_cache)
    second = np.asarray(fn(*splats))
    assert len(_isect._graph_cache) == n_graphs, "steady state re-captured"
    assert np.array_equal(first, ref), "graph capture changed the image"
    assert np.array_equal(second, ref), "graph replay changed the image"
    splax.clear_scratch()


def test_graph_threshold_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above SPLAX_GRAPH_THRESHOLD the plain path runs and stays byte-identical."""
    splats = _scene(20_000, seed=4)
    kw = _kw(256, 256)

    splax.clear_scratch()
    ref = np.asarray(splax.inference.render(*splats, **kw))

    monkeypatch.setattr(_rast, "_POSTSYNC_GRAPHS", True)
    monkeypatch.setattr(_rast, "_GRAPH_THRESHOLD", 1)
    splax.clear_scratch()
    out = np.asarray(splax.inference.render(*splats, **kw))
    assert len(_isect._graph_cache) == 0, "graph captured despite threshold"
    assert np.array_equal(out, ref)
    splax.clear_scratch()


def test_clear_scratch_purges_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear_scratch drops the graph cache, whose graphs hold scratch addresses."""
    splats = _scene(10_000, seed=5)
    kw = _kw(128, 128)
    monkeypatch.setattr(_rast, "_POSTSYNC_GRAPHS", True)
    splax.clear_scratch()
    jax.jit(lambda *s: splax.inference.render(*s, **kw))(*splats).block_until_ready()
    assert len(_isect._graph_cache) > 0
    splax.clear_scratch()
    assert len(_isect._graph_cache) == 0
