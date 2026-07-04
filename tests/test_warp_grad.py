"""Gradient tests for the Warp backend (Phase 6a).

Three things are checked:
  1. Gradient parity vs the gsplat reference (torch autograd) on the LEGACY
     projection path, for two scalar losses (img.sum and a weighted MSE) and all
     five splat params. gsplat is a different CUDA kernel, so this is a numeric
     cross-check, not a bit-for-bit port comparison. It is guarded by the
     ``gsplat_ref`` fixture and skips when gsplat is unavailable.
  2. Finite-difference directional-derivative self-consistency on both the legacy
     and the tight (O6) render paths. Needs no reference; always runs.
  3. grad under jit works; grad under vmap is batch-native (Phase 6f) -- the batched
     backward matches per-sample sequential grad. Exhaustive batched coverage across
     diff_wrt selections lives in test_warp_grad_batched.py.

Parity tolerance: the well-behaved parameters (means/scales/colors/opacities) agree
to ~1e-4 relative Frobenius, so a 2e-3 bound holds with margin. The quaternion
gradient is the one documented convention difference: gsplat normalizes quats
INTERNALLY, so its grad is projected onto the unit-sphere tangent space at q
(orthogonal to q), while splax treats quats as already unit and keeps the radial
component too. We therefore compare the quaternion grads only after projecting
splax's onto the same tangent space (subtracting the component along q); the
tangential -- physically meaningful -- part agrees to ~2e-5.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import _gsplat_ref as gref  # noqa: E402
import splax  # noqa: E402


@pytest.fixture
def gsplat_ref() -> types.ModuleType:
    """Skip a test (not the whole module) when gsplat cannot run."""
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


class _ProjKW(_PK):
    block_width: int


def _pk(H: int, W: int) -> _PK:
    return {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }


def _splax_legacy(
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
    """splax render on the LEGACY (isotropic 3-sigma bbox) projection path."""
    pk: _ProjKW = {**_pk(H, W), "block_width": 16}
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, vm, **pk
    )
    return splax.rasterize(
        colors,
        opac,
        bg,
        xys,
        depths,
        radii,
        conics,
        cum,
        img_shape=(H, W),
        block_width=16,
        tight=False,
    )


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
    return splax.render(
        means, scales, quats, colors, opac, viewmat=vm, background=bg, **_pk(H, W)
    )[0]


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
        img = _splax_legacy(*a, bg, vm, H, W)
        return jnp.sum(img) if which == "sum" else jnp.mean(w * img**2)

    weight = None if which == "sum" else np.asarray(w)

    g_sp = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    g_gs = gref.grad(
        *args, viewmat=vm, background=bg, **_pk(H, W), weight=weight
    )

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


@pytest.mark.parametrize("path", ["legacy", "tight"])
def test_finite_difference(path: str) -> None:
    """Directional-derivative FD check: grad . v ~= (L(x+eps v) - L(x-eps v))/2eps.

    ~10 random parameters are exercised at once via a random unit direction per
    array; central differences in float32 give ~1e-2 relative accuracy, so we use a
    loose relative bound. Runs on both the legacy and tight (O6) render paths -- the
    tight path has no external reference, so this FD self-consistency is its grad
    check.
    """
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=7)
    render = _splax_legacy if path == "legacy" else _splax_tight
    w = jax.random.uniform(jax.random.key(5), (H, W, 3))

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        # Linear, mean-reduced loss: keeps the loss magnitude small (float32 render
        # -> minimal FD cancellation) while giving an O(1) gradient over all five
        # parameter arrays at once (~4800 perturbed entries).
        return jnp.mean(w * render(m, s, q, c, o, bg, vm, H, W))

    args = (means, scales, quats, colors, opac)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)

    # Perturb ALONG the gradient (per-array unit direction): maximizes the
    # directional-derivative signal relative to float32 render noise -- the
    # standard well-conditioned gradient check. The residual ~3% is intrinsic to
    # splatting's hard 1/255-cull / early-termination discontinuities, which FD
    # steps cross; hence the loose bound.
    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))

    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)

    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"{path} FD mismatch: analytic {analytic:.6e} vs numeric {numeric:.6e} (rel {rel:.2e})"
    )


def test_grad_under_jit() -> None:
    n, H, W = 2000, 128, 128
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=3)
    args = (means, scales, quats, colors, opac)
    loss = _losses(H, W)["wmse"]

    def sp(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        return _splax_legacy(m, s, q, c, o, bg, vm, H, W)

    g_eager = jax.grad(loss(sp), argnums=(0, 1, 2, 3, 4))(*args)
    g_jit = jax.jit(jax.grad(loss(sp), argnums=(0, 1, 2, 3, 4)))(*args)
    for a, b in zip(g_eager, g_jit):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_grad_under_vmap_matches_sequential() -> None:
    """Batch-native backward (Phase 6f): grad under vmap over a batched gaussian input
    must match per-sample sequential jax.grad (the batched callables are
    vmap_method='expand_dims'). Full mixed batched/broadcast coverage across every
    diff_wrt selection is in test_warp_grad_batched.py."""
    n, H, W, B = 500, 96, 96, 3
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=2)
    bmeans = means + 0.02 * jax.random.normal(jax.random.key(1), (B, n, 3))

    def loss(m: jax.Array) -> jax.Array:
        return jnp.sum(_splax_tight(m, scales, quats, colors, opac, bg, vm, H, W))

    gv = np.asarray(jax.vmap(jax.grad(loss))(bmeans))
    gs = np.stack([np.asarray(jax.grad(loss)(bmeans[i])) for i in range(B)])
    assert np.allclose(gv, gs, rtol=1e-4, atol=1e-6)


# --- Phase 6e: camera-pose (viewmat) gradients --------------------------------
def _render_diff(
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opac: jax.Array,
    bg: jax.Array,
    vm: jax.Array,
    H: int,
    W: int,
    diff_wrt: tuple[str, ...],
) -> jax.Array:
    return splax.training.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=vm,
        background=bg,
        diff_wrt=diff_wrt,
        **_pk(H, W),
    )[0]


def test_viewmat_finite_difference() -> None:
    """Directional-derivative FD check of the viewmat gradient (the primary
    validation -- gsplat's rasterization exposes no directly comparable viewmat
    grad in this setup). Perturb the 12
    differentiable viewmat entries along the analytic gradient direction; central
    differences match grad.direction to a few percent (loose bound for the same
    hard-cull discontinuities Phase 6a's FD test documents)."""
    n, H, W = 4000, 120, 120
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=11)
    w = jax.random.uniform(jax.random.key(4), (H, W, 3))

    def loss(v: jax.Array) -> jax.Array:
        return jnp.mean(
            w
            * _render_diff(
                means, scales, quats, colors, opac, bg, v, H, W, ("viewmat",)
            )
        )

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


def test_diff_wrt_consistency() -> None:
    """diff_wrt=('gaussians','viewmat') must reproduce the ('gaussians',) gaussian
    grads and the ('viewmat',) camera grad -- the 'both' kernel shares the exact
    vjp helpers, so both agree to tight tolerance (the tiny residual is FMA
    scheduling under the different tile-launch geometry, ~1e-9)."""
    n, H, W = 3000, 110, 110
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=5)
    w = jax.random.uniform(jax.random.key(6), (H, W, 3))

    def gloss(m: jax.Array, diff: tuple[str, ...]) -> jax.Array:
        return jnp.mean(
            w * _render_diff(m, scales, quats, colors, opac, bg, vm, H, W, diff)
        )

    gm_only = jax.grad(gloss)(means, ("gaussians",))
    gm_both = jax.grad(gloss)(means, ("gaussians", "viewmat"))
    assert np.allclose(np.asarray(gm_only), np.asarray(gm_both), rtol=1e-5, atol=1e-6)

    def vloss(v: jax.Array, diff: tuple[str, ...]) -> jax.Array:
        return jnp.mean(
            w * _render_diff(means, scales, quats, colors, opac, bg, v, H, W, diff)
        )

    gv_only = np.asarray(jax.grad(vloss)(vm, ("viewmat",)))
    gv_both = np.asarray(jax.grad(vloss)(vm, ("gaussians", "viewmat")))
    assert np.allclose(gv_only, gv_both, rtol=1e-4, atol=1e-6)


def test_pose_chain_rule_fd() -> None:
    """jax chain rule: parametrize a pose as a 6-vector (axis-angle + translation)
    built into a 4x4 INSIDE jax, then grad w.r.t. the 6-vector. It must be finite
    and match a directional FD -- validates the grad flowing se3 -> 4x4 viewmat ->
    the Warp viewmat backward."""

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
        return jnp.mean(
            w
            * _render_diff(
                means, scales, quats, colors, opac, bg, se3(xi) @ vm, H, W, ("viewmat",)
            )
        )

    g = np.asarray(jax.grad(loss)(xi0))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    d = g / (np.linalg.norm(g) + 1e-12)
    eps = 1e-3
    numeric = (
        float(loss(xi0 + jnp.asarray(d * eps)))
        - float(loss(xi0 - jnp.asarray(d * eps)))
    ) / (2 * eps)
    analytic = float(np.dot(g, d))
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"pose chain-rule FD mismatch: {analytic:.6e} vs {numeric:.6e} (rel {rel:.2e})"
    )


def test_viewmat_grad_under_vmap_matches_sequential() -> None:
    """The viewmat backward is batch-native (Phase 6f): grad under vmap over B distinct
    camera poses recovers per-pose camera gradients matching the sequential loop -- the
    core of scripts/optimize_pose.py --batch."""
    n, H, W, B = 500, 96, 96, 3
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=9)
    vms = jnp.broadcast_to(vm, (B, 4, 4)) + 0.02 * jax.random.normal(
        jax.random.key(2), (B, 4, 4)
    )
    vms = vms.at[:, 3, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(
            _render_diff(means, scales, quats, colors, opac, bg, v, H, W, ("viewmat",))
        )

    gv = np.asarray(jax.vmap(jax.grad(loss))(vms))
    gs = np.stack([np.asarray(jax.grad(loss)(vms[i])) for i in range(B)])
    # The rasterize backward accumulates with atomics, so even the sequential
    # path jitters up to ~2e-4 rel against itself run-to-run; 1e-4 flaked.
    assert np.allclose(gv, gs, rtol=2e-3, atol=1e-4)
