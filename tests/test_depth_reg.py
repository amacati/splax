"""Depth-render + depth-regularization gradient tests (survey T2).

Covers the opt-in expected-depth channel D(p) = Σ wᵢ dᵢ (``splax.rasterize_depth`` /
``splax.render(..., render_depth=True)``) added for COLMAP sparse-point depth
regularization (gsplat ``depth_loss``). Four things are checked:

  1. Forward correctness: for a SINGLE gaussian the depth channel equals dᵢ times the
     accumulated alpha, so D == pvz · A exactly (A = the colour render with unit
     colours over a black background = Σ wᵢ). Empty pixels have depth 0.
  2. Off-path byte-identity: the image returned by the depth path is bit-for-bit the
     plain ``rasterize`` image (the depth channel is a separate accumulator/kernel and
     must not perturb the colour blend).
  3. Finite-difference self-consistency of the depth gradient chain (same style/bound
     as ``test_warp_grad.py``): a depth-only loss and a mixed colour+depth loss both
     match a central-difference directional derivative. A depth-only loss produces a
     ZERO colour gradient (depth is independent of colours) and nonzero geometry
     gradients (the v_depths → project chain plus the depth → blend-weight chain).
  4. grad under jit matches eager; grad under vmap is batch-native (matches the
     per-sample sequential loop).

FD bound follows the existing splat FD tests (8e-2 rel): the hard 1/255-cull and
early-termination discontinuities that FD steps cross are the intrinsic residual.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import splax


class _PK(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


class _Common(_PK):
    viewmat: jax.Array
    background: jax.Array
    block_size: int


def _pk(H: int, W: int) -> _PK:
    return {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }


def _scene(
    n: int, H: int, W: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.split(jax.random.key(seed), 6)
    means = jax.random.normal(key[0], (n, 3)) * 0.5
    scales = jax.random.uniform(key[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(key[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(key[3], (n, 3))
    opac = jax.random.uniform(key[4], (n, 1), minval=0.1, maxval=0.6)
    bg = jax.random.uniform(key[5], (3,))
    vm = jnp.array(
        [[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32
    )
    return means, scales, quats, colors, opac, bg, vm


def test_depth_render_single_gaussian() -> None:
    """Single gaussian: D(p) = pvz · A(p) exactly (A = accumulated alpha), and empty
    pixels are 0. Validates the forward expected-depth accumulator against the colour
    blend it mirrors."""
    H = W = 64
    means = jnp.array([[0.1, -0.05, 0.0]])
    scales = jnp.array([[0.12, 0.12, 0.12]])
    quats = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    opac = jnp.array([[0.9]])
    vm = jnp.array(
        [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 4.0], [0, 0, 0, 1]], jnp.float32
    )
    pvz = float((vm[:3, :3] @ means[0] + vm[:3, 3])[2])  # camera-space depth
    black = jnp.zeros(3)

    # accumulated alpha A = Σ wᵢ : unit colour over a black background.
    img, _ = splax.render(
        means,
        scales,
        quats,
        jnp.ones((1, 3)),
        opac,
        viewmat=vm,
        background=black,
        block_size=16,
        **_pk(H, W),
    )
    A = img[..., 0]
    _img, depth = splax.render(
        means,
        scales,
        quats,
        jnp.array([[0.5, 0.5, 0.5]]),
        opac,
        viewmat=vm,
        background=black,
        render_depth=True,
        block_size=16,
        **_pk(H, W),
    )
    A = np.asarray(A)
    depth = np.asarray(depth)
    # covered region: D == pvz · A. Empty region: both 0.
    assert np.allclose(depth, pvz * A, atol=1e-4), (
        f"max dev {np.abs(depth - pvz * A).max():.2e}"
    )
    assert depth[0, 0] == 0.0 and A[0, 0] == 0.0  # corner is background
    assert depth.max() > 0.5 * pvz  # gaussian actually contributes


def test_offpath_image_byte_identical() -> None:
    """The depth path's image is bit-for-bit the plain rasterize image."""
    n, H, W = 4000, 128, 128
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)
    common: _Common = {"viewmat": vm, "background": bg, "block_size": 16, **_pk(H, W)}
    img_plain, _ = splax.render(means, scales, quats, colors, opac, **common)
    img_depth, _d = splax.render(
        means, scales, quats, colors, opac, render_depth=True, **common
    )
    assert np.array_equal(np.asarray(img_plain), np.asarray(img_depth))


@pytest.mark.parametrize("mode", ["depth_only", "mixed"])
def test_depth_grad_finite_difference(mode: str) -> None:
    """Central-difference directional-derivative check of the depth gradient chain,
    across all five splat params at once. depth-only isolates the v_depths + depth→
    blend-weight chains; mixed adds the colour channel."""
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=7)
    wd = jax.random.uniform(jax.random.key(9), (H, W))
    wc = jax.random.uniform(jax.random.key(10), (H, W, 3))

    def loss(
        m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array
    ) -> jax.Array:
        img, depth = splax.render(
            m,
            s,
            q,
            c,
            o,
            viewmat=vm,
            background=bg,
            render_depth=True,
            block_size=16,
            **_pk(H, W),
        )
        assert depth is not None  # render_depth=True fills the depth slot
        dl = jnp.mean(wd * depth)
        return dl if mode == "depth_only" else jnp.mean(wc * img) + dl

    args = (means, scales, quats, colors, opac)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)

    if mode == "depth_only":
        # depth is independent of colours -> exactly zero colour gradient.
        assert float(jnp.linalg.norm(grads[3])) == 0.0
    # geometry gradients are nonzero (the whole point of the regularizer).
    assert float(jnp.linalg.norm(grads[0])) > 0.0

    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))
    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"{mode} FD mismatch: {analytic:.6e} vs {numeric:.6e} (rel {rel:.2e})"
    )


def test_depth_grad_under_jit() -> None:
    n, H, W = 2000, 96, 96
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=3)
    wd = jax.random.uniform(jax.random.key(4), (H, W))
    args = (means, scales, quats, colors, opac)

    def loss(
        m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array
    ) -> jax.Array:
        _img, depth = splax.render(
            m,
            s,
            q,
            c,
            o,
            viewmat=vm,
            background=bg,
            render_depth=True,
            block_size=16,
            **_pk(H, W),
        )
        assert depth is not None  # render_depth=True fills the depth slot
        return jnp.mean(wd * depth)

    g_eager = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    g_jit = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args)
    for a, b in zip(g_eager, g_jit):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_depth_grad_under_vmap_matches_sequential() -> None:
    """The depth backward is batch-native (shares the batched image_id indexing): grad
    under vmap over a batched gaussian input matches the per-sample sequential grad."""
    n, H, W, B = 500, 96, 96, 3
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=2)
    bmeans = means + 0.02 * jax.random.normal(jax.random.key(1), (B, n, 3))
    wd = jax.random.uniform(jax.random.key(5), (H, W))

    def loss(m: jax.Array) -> jax.Array:
        _img, depth = splax.render(
            m,
            scales,
            quats,
            colors,
            opac,
            viewmat=vm,
            background=bg,
            render_depth=True,
            block_size=16,
            **_pk(H, W),
        )
        assert depth is not None  # render_depth=True fills the depth slot
        return jnp.sum(wd * depth)

    gv = np.asarray(jax.vmap(jax.grad(loss))(bmeans))
    gs = np.stack([np.asarray(jax.grad(loss)(bmeans[i])) for i in range(B)])
    assert np.allclose(gv, gs, rtol=2e-3, atol=1e-5)
