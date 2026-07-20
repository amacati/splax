"""Gradient tests for the Warp backend.

Three things are checked. First, gradient parity vs the gsplat reference (torch
autograd) for two scalar losses and all five splat params. gsplat is a different
CUDA kernel, so this is a numeric cross-check, not a bit-for-bit port comparison.
It fails when gsplat is unavailable. Second, finite-difference directional
derivative self-consistency, which needs no reference and always runs. Third,
grad under jit and batch-native grad under vmap matching the per-sample
sequential grads. Exhaustive batched coverage lives in test_warp_grad_batched.py.

Parity tolerance: the well-behaved parameters agree to ~1e-4 relative Frobenius,
so a 2e-3 bound holds with margin. The quaternion gradient is the one documented
convention difference. gsplat normalizes quats internally, so its grad lives in
the unit-sphere tangent space at q, while splax treats quats as already unit and
keeps the radial component too. The quaternion grads are compared after
projecting splax's onto the same tangent space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import _gsplat_ref as gref  # noqa: E402

import splax  # noqa: E402

if TYPE_CHECKING:
    import types
    from collections.abc import Callable


@pytest.fixture
def gsplat_ref() -> types.ModuleType:
    """Fail the test with a clear reason when gsplat cannot run."""
    gref.require_working()
    return gref


def _scene(
    n: int, H: int, W: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1), minval=0.1, maxval=0.6)
    bg = jax.random.uniform(k[5], (3,))
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
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


def _splax_tight(
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opac: jax.Array,
    bg: jax.Array,
    vm: jax.Array,
    H: int,
    W: int,
) -> jax.Array:
    return splax.render(means, scales, quats, colors, opac, viewmat=vm, background=bg, **_pk(H, W))[
        0
    ]


def _losses(
    H: int, W: int, seed: int = 123
) -> dict[str, Callable[[Callable[..., jax.Array]], Callable[..., jax.Array]]]:
    w = jax.random.uniform(jax.random.key(seed), (H, W, 3))

    def sum_loss(render: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
        return lambda *a: jnp.sum(render(*a))

    def wmse_loss(render: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
        return lambda *a: jnp.mean(w * render(*a) ** 2)

    return {"sum": sum_loss, "wmse": wmse_loss}


@pytest.mark.parametrize("n,H,W", [(3000, 128, 128), (8000, 160, 160)])
@pytest.mark.parametrize("which", ["sum", "wmse"])
def test_grad_parity_vs_gsplat(
    n: int, H: int, W: int, which: str, gsplat_ref: types.ModuleType
) -> None:
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=n)
    args = (means, scales, quats, colors, opac)

    w = jax.random.uniform(jax.random.key(123), (H, W, 3))

    def loss(*a: jax.Array) -> jax.Array:
        img = _splax_tight(*a, bg, vm, H, W)
        return jnp.sum(img) if which == "sum" else jnp.mean(w * img**2)

    weight = None if which == "sum" else np.asarray(w)

    g_sp = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    g_gs = gref.grad(*args, viewmat=vm, background=bg, **_pk(H, W), weight=weight)

    qn = np.asarray(quats)
    for name, a, b in zip(["means", "scales", "quats", "colors", "opac"], g_sp, g_gs):
        a = np.asarray(a)
        b = np.asarray(b)
        if name == "quats":
            # gsplat differentiates through its internal quat normalization, so its
            # grad lives in the tangent space at q (orthogonal to q). Project splax's
            # onto the same space (drop the radial component) before comparing.
            a = a - np.sum(a * qn, axis=-1, keepdims=True) * qn
            tol = 5e-3
        else:
            tol = 2e-3
        # Relative-Frobenius is the meaningful metric across the two kernels: the
        # whole gradient field agrees to ~1e-4 relative (quats tangential ~2e-5).
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
        assert rel < tol, f"{which}/{name} relative grad error {rel:.2e}"


def test_finite_difference() -> None:
    """Directional-derivative FD check: grad . v ~= (L(x+eps v) - L(x-eps v))/2eps.

    Hundreds of random parameters are exercised at once via a random unit
    direction per array. Central differences in float32 give ~1e-2 relative
    accuracy, so a loose relative bound is used.
    """
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=7)
    render = _splax_tight
    w = jax.random.uniform(jax.random.key(5), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        # Linear, mean-reduced loss keeps the loss magnitude small (float32 render,
        # so minimal FD cancellation) while giving an O(1) gradient over all five
        # parameter arrays at once (~4800 perturbed entries).
        return jnp.mean(w * render(m, s, q, c, o, bg, vm, H, W))

    args = (means, scales, quats, colors, opac)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)

    # Perturb ALONG the gradient (per-array unit direction): maximizes the
    # directional-derivative signal relative to float32 render noise, the
    # standard well-conditioned gradient check. The residual ~3% is intrinsic to
    # splatting's hard 1/255-cull / early-termination discontinuities, which FD
    # steps cross, hence the loose bound.
    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))

    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)

    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"FD mismatch: analytic {analytic:.6e} vs numeric {numeric:.6e} (rel {rel:.2e})"
    )


def test_grad_under_jit() -> None:
    n, H, W = 2000, 128, 128
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=3)
    args = (means, scales, quats, colors, opac)
    loss = _losses(H, W)["wmse"]

    def sp(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        return _splax_tight(m, s, q, c, o, bg, vm, H, W)

    g_eager = jax.grad(loss(sp), argnums=(0, 1, 2, 3, 4))(*args)
    g_jit = jax.jit(jax.grad(loss(sp), argnums=(0, 1, 2, 3, 4)))(*args)
    for a, b in zip(g_eager, g_jit):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_grad_under_vmap_matches_sequential() -> None:
    """Match vmap gaussian grads against sequential grads."""
    n, H, W, B = 500, 96, 96, 3
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=2)
    bmeans = means + 0.02 * jax.random.normal(jax.random.key(1), (B, n, 3))

    def loss(m: jax.Array) -> jax.Array:
        return jnp.sum(_splax_tight(m, scales, quats, colors, opac, bg, vm, H, W))

    gv = np.asarray(jax.vmap(jax.grad(loss))(bmeans))
    gs = np.stack([np.asarray(jax.grad(loss)(bmeans[i])) for i in range(B)])
    # The rasterize backward accumulates with atomics, so even the sequential
    # path jitters run to run. rtol=1e-4 flaked, same bound as the viewmat
    # variant below.
    assert np.allclose(gv, gs, rtol=2e-3, atol=1e-4)


# --- camera-pose (viewmat) gradients --------------------------------
# Gradient selection is pure jax.grad/argnums. Differentiating with respect to the
# viewmat runs the camera-pose accumulator, differentiating with respect to a
# gaussian input runs the gaussian-grad kernels, and both together run the joint
# kernel. The jax.grad argnums alone decide which kernels run.
def _render(
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opac: jax.Array,
    bg: jax.Array,
    vm: jax.Array,
    H: int,
    W: int,
) -> jax.Array:
    return splax.render(means, scales, quats, colors, opac, viewmat=vm, background=bg, **_pk(H, W))[
        0
    ]


def test_viewmat_finite_difference() -> None:
    """Check viewmat gradients with directional finite differences."""
    n, H, W = 4000, 120, 120
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=11)
    w = jax.random.uniform(jax.random.key(4), (H, W, 3))

    def loss(v: jax.Array) -> jax.Array:
        return jnp.mean(w * _render(means, scales, quats, colors, opac, bg, v, H, W))

    g = np.asarray(jax.grad(loss)(vm))
    assert np.all(np.isfinite(g)), "viewmat grad has non-finite entries"
    assert np.allclose(g[3], 0.0), "last viewmat row must have zero grad (constant)"
    # unit direction on the 12 differentiable entries only.
    d = np.zeros((4, 4), np.float32)
    d[:3] = g[:3] / (np.linalg.norm(g[:3]) + 1e-12)
    analytic = float(np.vdot(g, d))
    eps = 1e-3
    plus = float(loss(vm + jnp.asarray(d * eps)))
    minus = float(loss(vm - jnp.asarray(d * eps)))
    numeric = (plus - minus) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"viewmat FD mismatch: analytic {analytic:.6e} vs numeric {numeric:.6e} (rel {rel:.2e})"
    )


def test_grad_selection_consistency() -> None:
    """Match joint kernel grads against per path kernel grads."""
    n, H, W = 3000, 110, 110
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=5)
    w = jax.random.uniform(jax.random.key(6), (H, W, 3))

    def loss(m: jax.Array, v: jax.Array) -> jax.Array:
        return jnp.mean(w * _render(m, scales, quats, colors, opac, bg, v, H, W))

    # gaussian grad: means-only (gaussian kernel) vs joint (both kernel).
    gm_only = jax.grad(loss, argnums=0)(means, vm)
    gm_both, gv_both = jax.grad(loss, argnums=(0, 1))(means, vm)
    assert np.allclose(np.asarray(gm_only), np.asarray(gm_both), rtol=1e-5, atol=1e-6)

    # camera grad: viewmat-only (view kernel) vs joint (both kernel).
    gv_only = np.asarray(jax.grad(loss, argnums=1)(means, vm))
    assert np.allclose(gv_only, np.asarray(gv_both), rtol=1e-4, atol=1e-6)


def test_pose_chain_rule_fd() -> None:
    """Validate se3 chain rule gradients with finite differences."""

    def skew(v: jax.Array) -> jax.Array:
        return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    def se3(xi: jax.Array) -> jax.Array:
        theta = jnp.sqrt(jnp.sum(xi[:3] ** 2) + 1e-12)
        K = skew(xi[:3] / theta)
        R = jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
        top = jnp.concatenate([R, xi[3:].reshape(3, 1)], axis=1)
        return jnp.concatenate([top, jnp.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    n, H, W = 4000, 120, 120
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=13)
    w = jax.random.uniform(jax.random.key(8), (H, W, 3))
    xi0 = jnp.asarray(np.array([0.03, -0.02, 0.015, 0.04, -0.03, 0.02], np.float32))

    def loss(xi: jax.Array) -> jax.Array:
        return jnp.mean(w * _render(means, scales, quats, colors, opac, bg, se3(xi) @ vm, H, W))

    g = np.asarray(jax.grad(loss)(xi0))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    d = g / (np.linalg.norm(g) + 1e-12)
    eps = 1e-3
    numeric = (
        float(loss(xi0 + jnp.asarray(d * eps))) - float(loss(xi0 - jnp.asarray(d * eps)))
    ) / (2 * eps)
    analytic = float(np.dot(g, d))
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"pose chain-rule FD mismatch: {analytic:.6e} vs {numeric:.6e} (rel {rel:.2e})"
    )


def test_viewmat_grad_under_vmap_matches_sequential() -> None:
    """Match vmap viewmat grads against sequential viewmat grads."""
    n, H, W, B = 500, 96, 96, 3
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=9)
    vms = jnp.broadcast_to(vm, (B, 4, 4)) + 0.02 * jax.random.normal(jax.random.key(2), (B, 4, 4))
    vms = vms.at[:, 3, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(_render(means, scales, quats, colors, opac, bg, v, H, W))

    gv = np.asarray(jax.vmap(jax.grad(loss))(vms))
    gs = np.stack([np.asarray(jax.grad(loss)(vms[i])) for i in range(B)])
    # The rasterize backward accumulates with atomics, so even the sequential
    # path jitters up to ~2e-4 rel against itself run-to-run. 1e-4 flaked.
    assert np.allclose(gv, gs, rtol=2e-3, atol=1e-4)
