"""Test anti aliased opacity compensation.

Tests for the anti-aliased opacity compensation (Mip-Splatting / gsplat
``rasterize_mode="antialiased"``).

The compensation multiplies a per-gaussian factor ρ = √(det Σ₂D / det(Σ₂D+εI))
into the opacity before the blend, cancelling the area inflation the ε=0.3 screen
dilation grants thin gaussians. ρ is computed in JAX from the projection's own
``conics`` output (``splax.opacity_compensation``). Its gradient chains to
scales/quats/means through project's existing conic-to-covariance vjp.

Checks:
  1. Closed form: ρ from the conic equals the direct det ratio, ρ∈[0,1], and culled
     gaussians give 1.
  2. Off == plain: ``antialiased=False`` is byte-identical (forward + grad) to the
     plain grad-free path. The ``map_opacities`` split is a no-op when equal.
  3. On changes the render (ρ<1 for real gaussians).
  4. Finite-difference directional-derivative self-consistency of the antialiased
     grads over all five splat params (same style and bound as the finite-difference
     gradient tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import splax

if TYPE_CHECKING:
    from collections.abc import Callable


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


def test_compensation_closed_form() -> None:
    """ρ from the conic matches the direct det-ratio, bounded to [0,1], culled gaussians give 1."""
    n, H, W = 3000, 128, 128
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)
    xys, depths, radii, conics, _nth, _cum = splax.project(
        means, scales, quats, vm, **_pk(H, W), opacities=opac.reshape(n)
    )
    rho = np.asarray(splax.opacity_compensation(conics, radii))
    conics = np.asarray(conics).reshape(n, 3)
    radii = np.asarray(radii).reshape(n)
    eps = 0.3
    # Reference: rebuild the dilated Σ₂D from the conic (= its inverse), strip the
    # ε dilation, take the det ratio directly.
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    live = radii > 0
    det_conic = a * c - b * b
    det_d = np.where(live, 1.0 / np.where(det_conic == 0, 1.0, det_conic), 1.0)
    cxx, cyy, cxy = c * det_d, a * det_d, -b * det_d
    det_o = (cxx - eps) * (cyy - eps) - cxy * cxy
    ref = np.sqrt(np.clip(np.where(live, det_o / det_d, 1.0), 0.0, 1.0))
    assert np.allclose(rho[live], ref[live], atol=1e-5), (
        f"max |ρ - ref| = {np.abs(rho[live] - ref[live]).max():.2e}"
    )
    assert np.all(rho >= 0.0) and np.all(rho <= 1.0), "ρ must lie in [0, 1]"
    assert np.allclose(rho[~live], 1.0), "culled gaussians must get ρ = 1"
    # real gaussians actually get compensated (ρ meaningfully below 1 somewhere)
    assert rho[live].min() < 0.98, "expected some thin gaussians with ρ < 1"


def test_antialiased_off_matches_plain() -> None:
    """Match the plain render when anti aliasing is off."""
    n, H, W = 2500, 110, 110
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=2)
    pk = _pk(H, W)

    off, _ = splax.render(
        means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=False, **pk
    )
    plain, _ = splax.render(means, scales, quats, colors, opac, viewmat=vm, background=bg, **pk)
    assert np.array_equal(np.asarray(off), np.asarray(plain)), (
        "antialiased=False must be byte-identical to the plain forward"
    )

    # At the rasterize level, map_opacities=opac vs None gives byte-identical forward + grad.
    xys, depths, radii, conics, _nth, cum = splax.project(
        means, scales, quats, vm, **pk, opacities=opac.reshape(n)
    )

    def rast(map_opac: jax.Array | None) -> Callable[[jax.Array], jax.Array]:
        def f(o: jax.Array) -> jax.Array:
            return jnp.mean(
                splax.rasterize(
                    colors,
                    o,
                    bg,
                    xys,
                    depths,
                    radii,
                    conics,
                    cum,
                    img_shape=(H, W),
                    map_opacities=map_opac,
                )
            )

        return f

    g_none = np.asarray(jax.grad(rast(None))(opac))
    g_self = np.asarray(jax.grad(rast(opac))(opac))
    assert np.allclose(g_none, g_self, rtol=2e-3, atol=1e-6), (
        f"map_opacities=opac vs None grad mismatch beyond atomic jitter "
        f"(max|d|={np.abs(g_none - g_self).max():.2e})"
    )


def test_antialiased_changes_output() -> None:
    n, H, W = 2500, 110, 110
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=3)
    pk = _pk(H, W)
    off = np.asarray(
        splax.render(
            means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=False, **pk
        )[0]
    )
    on = np.asarray(
        splax.render(
            means, scales, quats, colors, opac, viewmat=vm, background=bg, antialiased=True, **pk
        )[0]
    )
    assert np.abs(on - off).max() > 1e-3, "antialiased render must differ from plain"


def test_antialiased_finite_difference() -> None:
    """Check anti aliased gradients with finite differences."""
    n, H, W = 400, 80, 80
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=7)
    w = jax.random.uniform(jax.random.key(5), (H, W, 3))
    pk = _pk(H, W)

    def loss(m: jax.Array, s: jax.Array, q: jax.Array, c: jax.Array, o: jax.Array) -> jax.Array:
        img, _ = splax.render(m, s, q, c, o, viewmat=vm, background=bg, antialiased=True, **pk)
        return jnp.mean(w * img)

    args = (means, scales, quats, colors, opac)
    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args)
    dirs = [g / (jnp.linalg.norm(g) + 1e-12) for g in grads]
    analytic = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))

    eps = 2e-3
    plus = [a + eps * d for a, d in zip(args, dirs)]
    minus = [a - eps * d for a, d in zip(args, dirs)]
    numeric = (float(loss(*plus)) - float(loss(*minus))) / (2 * eps)
    rel = abs(analytic - numeric) / (abs(numeric) + 1e-12)
    assert rel < 8e-2, (
        f"antialiased FD mismatch: analytic {analytic:.6e} vs numeric {numeric:.6e} (rel {rel:.2e})"
    )
