"""Batch-native gradient tests for the Warp backend (Phase 6f).

The forward path has been batch-native since Phase 4d (one kernel launch over B
images, camera id packed in the sort key). Phase 6f makes the *backward* path
batch-native too: the projection and rasterization backward FFIs are
``vmap_method="expand_dims"``, so ``jax.vmap(jax.grad(loss))`` runs a single
batched backward launch instead of falling back to a per-sample Python loop.

What these tests pin down (the correctness contract):

  1. For every ``diff_wrt`` selection -- ``("gaussians",)``, ``("viewmat",)`` and
     ``("gaussians","viewmat")`` -- and every differentiable parameter,
     ``jax.vmap(jax.grad(loss))`` over a B=3 batch matches the per-sample sequential
     ``jax.grad`` loop to tight tolerance.
  2. Mixed batched/broadcast operands. The realistic multi-view regime batches the
     camera pose (``viewmat``) -- which makes the whole projected geometry batched --
     while the gaussians are shared (broadcast). A gradient w.r.t. a broadcast
     parameter is the SUM over the batch axis of the per-sample gradients (the vjp of
     a broadcast is a sum); a gradient w.r.t. a per-image parameter is per-image.
  3. The degenerate broadcast-geometry case: a single shared render differentiated
     against B different target images (nothing inside ``render`` is batched, only the
     per-sample loss weight). Here the projected geometry is broadcast (B_geom=1) while
     the image cotangent is batched (B_out=3); the backward must still index geometry
     from image 0 and scatter per-output-image grads.
  4. A finite-difference spot check of one batched viewmat gradient, so the batched
     camera grad is validated against numerics and not only against the (identical)
     sequential kernel.

The gsplat reference has no batched-gradient entry point, so there is no external
parity here -- the sequential splax grad (itself parity-checked against gsplat on
the unbatched legacy path in test_warp_grad.py) is the reference. Tolerance mirrors
test_warp_grad.py: vmap vs sequential differ only by float32 atomic-add ordering
across the block/tile launch geometry, which is well under 1e-4 relative.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import splax

B = 3


def _scene(n: int, H: int, W: int, seed: int = 0) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1), minval=0.1, maxval=0.6)
    bg = jax.random.uniform(k[5], (3,))
    vm = jnp.array(
        [[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32
    )
    return means, scales, quats, colors, opac, bg, vm


class _PK(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


def _pk(H: int, W: int) -> _PK:
    return {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }


def _poses(vm: jax.Array, seed: int) -> jax.Array:
    """B distinct camera poses (perturbed viewmat with an exact bottom row)."""
    d = 0.02 * jax.random.normal(jax.random.key(seed), (B, 4, 4))
    vms = jnp.broadcast_to(vm, (B, 4, 4)) + d
    return vms.at[:, 3, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))


def _render(
    m: jax.Array,
    s: jax.Array,
    q: jax.Array,
    c: jax.Array,
    o: jax.Array,
    v: jax.Array,
    bg: jax.Array,
    H: int,
    W: int,
    diff: tuple[str, ...],
) -> jax.Array:
    return splax.training.render(
        m, s, q, c, o, viewmat=v, background=bg, diff_wrt=diff, **_pk(H, W)
    )[0]


def _rel(a: jax.Array | np.ndarray, b: jax.Array | np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _assert_close(name: str, gv: jax.Array | np.ndarray, gs: jax.Array | np.ndarray) -> None:
    gv, gs = np.asarray(gv), np.asarray(gs)
    assert gv.shape == gs.shape, f"{name}: shape {gv.shape} vs {gs.shape}"
    assert np.all(np.isfinite(gv)), f"{name}: non-finite vmap grad"
    rel = _rel(gv, gs)
    assert rel < 1e-3, f"{name}: vmap vs sequential rel error {rel:.2e}"
    assert np.allclose(gv, gs, rtol=1e-4, atol=1e-6), (
        f"{name}: max|d|={np.abs(gv - gs).max():.2e}"
    )


# --- Multi-view regime: batched viewmat -> batched geometry -------------------
# in_axes map the per-image viewmat and the per-image loss-weight index; the
# differentiated parameter is either mapped (per-image grad) or broadcast (summed).


@pytest.mark.parametrize("diff", [("gaussians",), ("gaussians", "viewmat")])
def test_batched_gaussians_per_image(diff: tuple[str, ...]) -> None:
    """Per-image gaussian input (means batched): vmap(grad) == sequential stack."""
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=2)
    vms = _poses(vm, 7)
    w = jax.random.uniform(jax.random.key(3), (B, H, W, 3))
    bmeans = means + 0.02 * jax.random.normal(jax.random.key(4), (B, n, 3))

    def loss(m_i: jax.Array, v_i: jax.Array, i: jax.Array | int) -> jax.Array:
        return jnp.mean(w[i] * _render(m_i, scales, quats, colors, opac, v_i, bg, H, W, diff))

    gv = jax.vmap(jax.grad(loss), in_axes=(0, 0, 0))(bmeans, vms, jnp.arange(B))
    gs = jnp.stack([jax.grad(loss)(bmeans[i], vms[i], i) for i in range(B)])
    _assert_close(f"{diff} means(batched)", gv, gs)


@pytest.mark.parametrize("diff", [("gaussians",), ("gaussians", "viewmat")])
@pytest.mark.parametrize("param", ["means", "scales", "quats", "colors", "opac"])
def test_broadcast_param_summed(diff: tuple[str, ...], param: str) -> None:
    """Broadcast (shared) gaussian/appearance parameter: the vmap per-sample grads
    SUM over the batch to the grad of the summed loss (the vjp of a broadcast)."""
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=5)
    vms = _poses(vm, 8)
    w = jax.random.uniform(jax.random.key(6), (B, H, W, 3))
    base = dict(means=means, scales=scales, quats=quats, colors=colors, opac=opac)

    def loss(p: jax.Array, v_i: jax.Array, i: jax.Array | int) -> jax.Array:
        kw = dict(base)
        kw[param] = p
        return jnp.mean(
            w[i]
            * _render(kw["means"], kw["scales"], kw["quats"], kw["colors"], kw["opac"],
                      v_i, bg, H, W, diff)
        )

    gper = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(base[param], vms, jnp.arange(B))
    gsummed = jnp.sum(gper, axis=0)

    def total(p: jax.Array) -> jax.Array:
        # sum() is typed with a Literal[0] start; the runtime value is always an Array.
        return cast(jax.Array, sum(loss(p, vms[i], i) for i in range(B)))

    _assert_close(f"{diff} {param}(broadcast summed)", gsummed, jax.grad(total)(base[param]))


@pytest.mark.parametrize("diff", [("viewmat",), ("gaussians", "viewmat")])
def test_batched_viewmat_per_pose(diff: tuple[str, ...]) -> None:
    """Per-image camera pose (viewmat batched): vmap(grad) recovers per-pose camera
    gradients matching the sequential loop -- the mechanism scripts/optimize_pose.py
    --batch relies on."""
    n, H, W = 800, 96, 96
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=11)
    vms = _poses(vm, 12)
    w = jax.random.uniform(jax.random.key(13), (B, H, W, 3))

    def loss(v_i: jax.Array, i: jax.Array | int) -> jax.Array:
        return jnp.mean(w[i] * _render(means, scales, quats, colors, opac, v_i, bg, H, W, diff))

    gv = jax.vmap(jax.grad(loss), in_axes=(0, 0))(vms, jnp.arange(B))
    gs = jnp.stack([jax.grad(loss)(vms[i], i) for i in range(B)])
    _assert_close(f"{diff} viewmat(batched)", gv, gs)
    assert np.allclose(np.asarray(gv)[:, 3, :], 0.0), "viewmat bottom row must be constant"


# --- Degenerate broadcast-geometry: shared render, batched target -------------


@pytest.mark.parametrize("param", ["means", "colors", "opac"])
def test_broadcast_geometry_shared_render(param: str) -> None:
    """A single shared render (broadcast viewmat AND gaussians) differentiated against
    B distinct per-sample loss weights. Nothing inside render is batched, so the
    projected geometry is broadcast (B_geom=1) while the image cotangent is batched
    (B_out=3). vmap(grad) per-sample must still match the sequential stack."""
    n, H, W = 600, 80, 80
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=17)
    w = jax.random.uniform(jax.random.key(18), (B, H, W, 3))
    base = dict(means=means, scales=scales, quats=quats, colors=colors, opac=opac)

    def loss(p: jax.Array, i: jax.Array | int) -> jax.Array:
        kw = dict(base)
        kw[param] = p
        return jnp.mean(
            w[i]
            * _render(kw["means"], kw["scales"], kw["quats"], kw["colors"], kw["opac"],
                      vm, bg, H, W, ("gaussians",))
        )

    gv = jax.vmap(jax.grad(loss), in_axes=(None, 0))(base[param], jnp.arange(B))
    gs = jnp.stack([jax.grad(loss)(base[param], i) for i in range(B)])
    _assert_close(f"broadcast-geometry {param}", gv, gs)


# --- Finite-difference spot check of a batched viewmat gradient ---------------


def test_batched_viewmat_finite_difference() -> None:
    """Directional-derivative FD check on ONE image of a batched viewmat grad: the
    vmap-produced camera gradient for image 0 matches central differences of that
    image's loss. Grounds the batched camera grad in numerics, not only in the
    (kernel-identical) sequential comparison."""
    n, H, W = 3000, 110, 110
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=21)
    vms = _poses(vm, 22)
    w = jax.random.uniform(jax.random.key(23), (B, H, W, 3))

    def loss(v_i: jax.Array, i: jax.Array | int) -> jax.Array:
        return jnp.mean(w[i] * _render(means, scales, quats, colors, opac, v_i, bg, H, W, ("viewmat",)))

    g = np.asarray(jax.vmap(jax.grad(loss), in_axes=(0, 0))(vms, jnp.arange(B)))[0]
    assert np.all(np.isfinite(g))
    d = np.zeros((4, 4), np.float32)
    d[:3] = g[:3] / (np.linalg.norm(g[:3]) + 1e-12)
    analytic = float(np.vdot(g, d))
    eps = 1e-3
    plus = float(loss(vms[0] + jnp.asarray(d * eps), 0))
    minus = float(loss(vms[0] - jnp.asarray(d * eps), 0))
    numeric = (plus - minus) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, f"batched viewmat FD mismatch: {analytic:.3e} vs {numeric:.3e} (rel {rel:.2e})"
