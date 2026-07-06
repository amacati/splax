"""Per-object rigid transforms on the grad-free inference path.

Non-overlapping slices of the splat each follow their own 4x4 world-space
transform, applied on the fly inside the projection kernel. Checked here:

  1. Identity transforms are byte-identical to the plain render, and omitting
     transforms is byte-identical too (the guarded kernel block never runs).
  2. Correctness against a manual reference that pre-transforms the slice's
     means and quats in JAX. The two formulations are mathematically equal but
     round differently, so projection outputs are compared tightly (no radii or
     visibility flips) and images perceptually.
  3. Batching. vmap over the transform stack equals the sequential loop
     bit-exactly, with the splat broadcast.
  4. Multiple objects move independently.
  5. Invalid slices and mismatched shapes raise immediately.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import splax
from splax._project import _project_call


class _KW(TypedDict):
    viewmat: jax.Array
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]


def _scene(
    n: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1), minval=0.1, maxval=0.6)
    return means, scales, quats, colors, opac


def _kw(H: int, W: int) -> _KW:
    vm = jnp.array(
        [[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32
    )
    return {
        "viewmat": vm,
        "background": jnp.zeros(3),
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
    }


def _euler_T(
    rx: float, ry: float, rz: float, t: tuple[float, float, float]
) -> np.ndarray:
    """4x4 rigid transform from XYZ Euler angles in radians plus a translation."""

    def rot(axis: int, a: float) -> np.ndarray:
        ca, sa = np.cos(a), np.sin(a)
        m = np.eye(3, dtype=np.float64)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i] = ca
        m[j, j] = ca
        m[i, j] = -sa if axis != 1 else sa
        m[j, i] = sa if axis != 1 else -sa
        return m

    R = rot(2, rz) @ rot(1, ry) @ rot(0, rx)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t
    return T


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return np.array([w, x, y, z], np.float32)


def _qmul_wxyz(a: jax.Array, b: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _manual_move(
    means: jax.Array, quats: jax.Array, T: np.ndarray, start: int, stop: int
) -> tuple[jax.Array, jax.Array]:
    """Reference transform of a slice, applied to the splat arrays in JAX."""
    R = jnp.asarray(T[:3, :3])
    t = jnp.asarray(T[:3, 3])
    q_obj = jnp.asarray(_rotmat_to_quat_wxyz(T[:3, :3]))
    m2 = means.at[start:stop].set(means[start:stop] @ R.T + t)
    q2 = quats.at[start:stop].set(_qmul_wxyz(q_obj[None], quats[start:stop]))
    return m2, q2


def test_identity_transforms_byte_identical() -> None:
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=1)
    kw = _kw(128, 128)
    plain = np.asarray(splax.inference.render(means, scales, quats, colors, opac, **kw))
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    ident = np.asarray(
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((0, 1000), (2000, 3000)),
        )
    )
    assert np.array_equal(plain, ident)


def test_projection_matches_manual_transform() -> None:
    """Kernel transform vs pre-transformed inputs, same projection outputs.

    The kernel rotates the covariance factor while the reference rotates the
    quaternion, mathematically equal with different rounding. Projected centers,
    depths, and conics must agree tightly, with zero radii or visibility flips.
    """
    n = 4000
    means, scales, quats, _colors, opac = _scene(n, seed=2)
    kw = _kw(128, 128)
    T = _euler_T(0.26, -0.17, 0.52, (0.3, -0.2, 0.1))
    args = (n, kw["img_shape"], kw["f"], kw["c"], 1.0, 0.01)
    tf_ids = jnp.full((n,), -1, jnp.int32).at[:1000].set(0)
    a = _project_call(
        means,
        scales,
        quats,
        kw["viewmat"],
        opac.reshape(n),
        *args,
        jnp.asarray(T)[None],
        tf_ids,
    )
    m2, q2 = _manual_move(means, quats, T, 0, 1000)
    b = _project_call(m2, scales, q2, kw["viewmat"], opac.reshape(n), *args)

    ra, rb = np.asarray(a[2]).ravel(), np.asarray(b[2]).ravel()
    np.testing.assert_array_equal(ra > 0, rb > 0)
    live = ra > 0
    np.testing.assert_allclose(
        np.asarray(a[0])[live], np.asarray(b[0])[live], atol=5e-2
    )
    np.testing.assert_allclose(
        np.asarray(a[1]).ravel()[live], np.asarray(b[1]).ravel()[live], atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(a[3])[live], np.asarray(b[3])[live], atol=1e-3
    )


def test_render_matches_manual_transform() -> None:
    """Full render vs the manual reference, bounded perceptually. Rounding flips
    a handful of pixels at blend cull and order boundaries."""
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=3)
    kw = _kw(128, 128)
    T = _euler_T(0.26, -0.17, 0.52, (0.3, -0.2, 0.1))
    moved = np.asarray(
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(T)[None],
            gaussian_slices=((0, 1000),),
        )
    )
    m2, q2 = _manual_move(means, quats, T, 0, 1000)
    ref = np.asarray(splax.inference.render(m2, scales, q2, colors, opac, **kw))
    mse = float(np.mean((moved - ref) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert psnr > 60, f"kernel vs manual transform PSNR only {psnr:.1f} dB"
    # the transform must actually change the image
    plain = np.asarray(splax.inference.render(means, scales, quats, colors, opac, **kw))
    assert np.abs(moved - plain).max() > 1e-2


def test_vmap_over_transforms_matches_sequential() -> None:
    """One shared splat, per-batch transforms. vmap equals the sequential loop
    bit-exactly, every element moves the slice differently."""
    n, B = 4000, 3
    means, scales, quats, colors, opac = _scene(n, seed=4)
    kw = _kw(96, 96)
    Ts = np.stack(
        [_euler_T(0.0, 0.0, 0.3 * i, (0.05 * i, -0.03 * i, 0.0)) for i in range(B)]
    )
    tfs = jnp.asarray(Ts)[:, None]  # (B, 1, 4, 4)

    def render_tf(tf: jax.Array) -> jax.Array:
        return splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=tf,
            gaussian_slices=((500, 1500),),
        )

    out = np.asarray(jax.vmap(render_tf)(tfs))
    seq = np.stack([np.asarray(render_tf(tfs[i])) for i in range(B)])
    np.testing.assert_array_equal(out, seq)
    # elements genuinely differ
    assert np.abs(out[0] - out[B - 1]).max() > 1e-2


def test_two_objects_move_independently() -> None:
    """Two slices with distinct transforms equal the manual reference applying
    both, and swapping the transforms changes the image."""
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=5)
    kw = _kw(128, 128)
    Ta = _euler_T(0.0, 0.0, 0.4, (0.2, 0.0, 0.0))
    Tb = _euler_T(0.3, 0.0, 0.0, (-0.1, 0.15, 0.0))
    slices = ((0, 800), (2000, 2600))
    both = np.asarray(
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(np.stack([Ta, Tb])),
            gaussian_slices=slices,
        )
    )
    m2, q2 = _manual_move(means, quats, Ta, 0, 800)
    m2, q2 = _manual_move(m2, q2, Tb, 2000, 2600)
    ref = np.asarray(splax.inference.render(m2, scales, q2, colors, opac, **kw))
    mse = float(np.mean((both - ref) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert psnr > 60, f"two-object transform PSNR only {psnr:.1f} dB"

    swapped = np.asarray(
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(np.stack([Tb, Ta])),
            gaussian_slices=slices,
        )
    )
    assert np.abs(both - swapped).max() > 1e-2


def test_invalid_transform_inputs_raise() -> None:
    n = 1000
    means, scales, quats, colors, opac = _scene(n, seed=6)
    kw = _kw(64, 64)
    eye = jnp.eye(4, dtype=jnp.float32)[None]

    with pytest.raises(ValueError, match="together"):
        splax.inference.render(
            means, scales, quats, colors, opac, **kw, gaussian_transforms=eye
        )
    with pytest.raises(ValueError, match="does not match"):
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((0, 100), (200, 300)),
        )
    with pytest.raises(ValueError, match="outside"):
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((900, 1100),),
        )
    with pytest.raises(ValueError, match="overlap"):
        splax.inference.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.broadcast_to(eye[0], (2, 4, 4)),
            gaussian_slices=((0, 500), (400, 600)),
        )
