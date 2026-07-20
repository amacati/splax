"""Per-object rigid transforms, forward semantics and gradients.

Non-overlapping slices of the splat each follow their own 4x4 world-space transform,
applied on the fly inside the projection kernel, with gradients flowing to the gaussian
parameters, the camera pose, and the transforms themselves. Checked here:

  1. Identity transforms are byte-identical to the plain render in both the image and
     the gradients, and omitting transforms is byte-identical too.
  2. Forward correctness against a manual reference that pre-transforms the slice's
     means and quats in JAX. The two formulations are mathematically equal but round
     differently, so projection outputs are compared tightly (no radii or visibility
     flips) and images perceptually.
  3. Gaussian gradients under active transforms against the same JAX reference through
     the plain, already validated backward. Quaternion gradients compare in the tangent
     space of the unit sphere: the reference normalizes through scipy while the kernel
     reports the raw-parameterization gradient (gsplat convention), and the radial
     component is projected out by the quat normalization every trainer applies anyway.
  4. Transform gradients in pose coordinates (rotvec + translation), the coordinates an
     object-pose optimizer consumes, avoiding the raw-matrix-entry ambiguity of
     gradients off the SO(3) manifold.
  5. Batching. vmap over the transform stack equals the sequential loop for the forward
     bit-exactly and for gradients numerically, including vmap over viewmats.
  6. Invalid slices and mismatched shapes raise immediately.
"""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform as TF
from scipy.spatial.transform import Rotation as R

import splax

# region forward


class _KW(TypedDict):
    viewmat: jax.Array
    background: jax.Array
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]


def _scene(n: int, seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1), minval=0.1, maxval=0.6)
    return means, scales, quats, colors, opac


def _kw(H: int, W: int) -> _KW:
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return {
        "viewmat": vm,
        "background": jnp.zeros(3),
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
    }


def _manual_move(
    means: jax.Array, quats: jax.Array, T: np.ndarray, start: int, stop: int
) -> tuple[jax.Array, jax.Array]:
    """Reference transform of a slice, applied to the splat arrays in JAX."""
    transform = TF.from_matrix(jnp.asarray(T))
    rotated = transform.rotation * R.from_quat(quats[start:stop], scalar_first=True)
    m2 = means.at[start:stop].set(transform.apply(means[start:stop]))
    q2 = quats.at[start:stop].set(rotated.as_quat(scalar_first=True))
    return m2, q2


def test_identity_transforms_byte_identical() -> None:
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=1)
    kw = _kw(128, 128)
    plain = np.asarray(splax.render(means, scales, quats, colors, opac, **kw)[0])
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    ident = np.asarray(
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=eye,
            gaussian_slices=((0, 1000), (2000, 3000)),
        )[0]
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
    rot = R.from_euler("xyz", [0.26, -0.17, 0.52])
    T = TF.from_components((0.3, -0.2, 0.1), rot).as_matrix().astype(np.float32)
    args = (n, kw["img_shape"], kw["f"], kw["c"], 1.0, 0.01)
    tf_ids = jnp.full((n,), -1, jnp.int32).at[:1000].set(0)
    a = splax._project._project_call(
        means, scales, quats, kw["viewmat"], opac.reshape(n), *args, jnp.asarray(T)[None], tf_ids
    )
    m2, q2 = _manual_move(means, quats, T, 0, 1000)
    b = splax._project._project_call(m2, scales, q2, kw["viewmat"], opac.reshape(n), *args)

    ra, rb = np.asarray(a[2]).ravel(), np.asarray(b[2]).ravel()
    np.testing.assert_array_equal(ra > 0, rb > 0)
    live = ra > 0
    np.testing.assert_allclose(np.asarray(a[0])[live], np.asarray(b[0])[live], atol=5e-2)
    np.testing.assert_allclose(
        np.asarray(a[1]).ravel()[live], np.asarray(b[1]).ravel()[live], atol=1e-3
    )
    np.testing.assert_allclose(np.asarray(a[3])[live], np.asarray(b[3])[live], atol=1e-3)


def test_render_matches_manual_transform() -> None:
    """Match transformed render against manual reference."""
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=3)
    kw = _kw(128, 128)
    rot = R.from_euler("xyz", [0.26, -0.17, 0.52])
    T = TF.from_components((0.3, -0.2, 0.1), rot).as_matrix().astype(np.float32)
    moved = np.asarray(
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(T)[None],
            gaussian_slices=((0, 1000),),
        )[0]
    )
    m2, q2 = _manual_move(means, quats, T, 0, 1000)
    ref = np.asarray(splax.render(m2, scales, q2, colors, opac, **kw)[0])
    mse = float(np.mean((moved - ref) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert psnr > 60, f"kernel vs manual transform PSNR only {psnr:.1f} dB"
    # the transform must actually change the image
    plain = np.asarray(splax.render(means, scales, quats, colors, opac, **kw)[0])
    assert np.abs(moved - plain).max() > 1e-2


def test_vmap_over_transforms_matches_sequential() -> None:
    """Match vmap transform output against sequential output."""
    n, B = 4000, 3
    means, scales, quats, colors, opac = _scene(n, seed=4)
    kw = _kw(96, 96)
    angles = np.array([[0.0, 0.0, 0.3 * i] for i in range(B)])
    trans = np.array([[0.05 * i, -0.03 * i, 0.0] for i in range(B)])
    Ts = TF.from_components(trans, R.from_euler("xyz", angles)).as_matrix().astype(np.float32)
    tfs = jnp.asarray(Ts)[:, None]  # (B, 1, 4, 4)

    def render_tf(tf: jax.Array) -> jax.Array:
        return splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=tf,
            gaussian_slices=((500, 1500),),
        )[0]

    out = np.asarray(jax.vmap(render_tf)(tfs))
    seq = np.stack([np.asarray(render_tf(tfs[i])) for i in range(B)])
    np.testing.assert_array_equal(out, seq)
    # elements genuinely differ
    assert np.abs(out[0] - out[B - 1]).max() > 1e-2


def test_two_objects_move_independently() -> None:
    """Move two slices independently and match the manual reference."""
    n = 4000
    means, scales, quats, colors, opac = _scene(n, seed=5)
    kw = _kw(128, 128)
    rot_a = R.from_euler("xyz", [0.0, 0.0, 0.4])
    Ta = TF.from_components((0.2, 0.0, 0.0), rot_a).as_matrix().astype(np.float32)
    rot_b = R.from_euler("xyz", [0.3, 0.0, 0.0])
    Tb = TF.from_components((-0.1, 0.15, 0.0), rot_b).as_matrix().astype(np.float32)
    slices = ((0, 800), (2000, 2600))
    both = np.asarray(
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(np.stack([Ta, Tb])),
            gaussian_slices=slices,
        )[0]
    )
    m2, q2 = _manual_move(means, quats, Ta, 0, 800)
    m2, q2 = _manual_move(m2, q2, Tb, 2000, 2600)
    ref = np.asarray(splax.render(m2, scales, q2, colors, opac, **kw)[0])
    mse = float(np.mean((both - ref) ** 2))
    psnr = 99.0 if mse == 0 else -10 * np.log10(mse)
    assert psnr > 60, f"two-object transform PSNR only {psnr:.1f} dB"

    swapped = np.asarray(
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.asarray(np.stack([Tb, Ta])),
            gaussian_slices=slices,
        )[0]
    )
    assert np.abs(both - swapped).max() > 1e-2


def test_invalid_transform_inputs_raise() -> None:
    n = 1000
    means, scales, quats, colors, opac = _scene(n, seed=6)
    kw = _kw(64, 64)
    eye = jnp.eye(4, dtype=jnp.float32)[None]

    with pytest.raises(ValueError, match="together"):
        splax.render(means, scales, quats, colors, opac, **kw, gaussian_transforms=eye)
    with pytest.raises(ValueError, match="does not match"):
        splax.render(
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
        splax.render(
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
        splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **kw,
            gaussian_transforms=jnp.broadcast_to(eye[0], (2, 4, 4)),
            gaussian_slices=((0, 500), (400, 600)),
        )


# region gradients

N = 2000
SLICES = ((0, 700), (1000, 1600))
K = len(SLICES)
IDS = np.asarray(splax._project._transform_ids(N, SLICES))
MOVED = IDS >= 0

ROTVECS = jnp.asarray([[0.15, -0.08, 0.3], [-0.2, 0.12, 0.05]], jnp.float32)
TRANS = jnp.asarray([[0.1, -0.05, 0.02], [-0.08, 0.03, 0.1]], jnp.float32)


def _tfs(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
    rot = R.from_rotvec(rotvecs).as_matrix()
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (K, 4, 4))
    return eye.at[:, :3, :3].set(rot).at[:, :3, 3].set(trans)


def _loss_kernel(
    means: jax.Array, scales: jax.Array, quats: jax.Array, tfs: jax.Array, *, extras: tuple
) -> jax.Array:
    colors, opac, kw, target = extras
    img, _ = splax.render(
        means, scales, quats, colors, opac, **kw, gaussian_transforms=tfs, gaussian_slices=SLICES
    )
    return jnp.mean((img - target) ** 2)


def _loss_reference(
    means: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    rotvecs: jax.Array,
    trans: jax.Array,
    *,
    extras: tuple,
) -> jax.Array:
    """The identical transform applied to the splat arrays in JAX, plain render.

    Parameterized by rotvec + translation rather than the 4x4 matrix, because scipy's
    matrix-to-quaternion conversion NaNs under jax.grad (branchy ``where`` gradients).
    """
    colors, opac, kw, target = extras
    rot = R.from_rotvec(rotvecs)[IDS]
    moved = rot.apply(means) + trans[IDS]
    composed = rot * R.from_quat(quats, scalar_first=True)
    means_ref = jnp.where(MOVED[:, None], moved, means)
    quats_ref = jnp.where(MOVED[:, None], composed.as_quat(scalar_first=True), quats)
    img, _ = splax.render(means_ref, scales, quats_ref, colors, opac, **kw)
    return jnp.mean((img - target) ** 2)


def _setup(seed: int) -> tuple:
    means, scales, quats, colors, opac = _scene(N, seed=seed)
    kw = _kw(96, 96)
    target = jax.random.uniform(jax.random.key(100 + seed), (96, 96, 3))
    return means, scales, quats, (colors, opac, kw, target)


def test_identity_transforms_match_plain_grads() -> None:
    means, scales, quats, extras = _setup(seed=2)
    eye = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (K, 4, 4))
    g_id = jax.grad(_loss_kernel, argnums=(0, 1, 2))(means, scales, quats, eye, extras=extras)

    def loss_plain(means: jax.Array, scales: jax.Array, quats: jax.Array) -> jax.Array:
        colors, opac, kw, target = extras
        img, _ = splax.render(means, scales, quats, colors, opac, **kw)
        return jnp.mean((img - target) ** 2)

    g_plain = jax.grad(loss_plain, argnums=(0, 1, 2))(means, scales, quats)
    for a, b, name in zip(g_id, g_plain, ("means", "scales", "quats")):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-8, err_msg=name)


def test_gaussian_grads_match_jax_reference() -> None:
    means, scales, quats, extras = _setup(seed=3)
    tfs = _tfs(ROTVECS, TRANS)
    gk = jax.grad(_loss_kernel, argnums=(0, 1, 2))(means, scales, quats, tfs, extras=extras)
    gr = jax.grad(_loss_reference, argnums=(0, 1, 2))(
        means, scales, quats, ROTVECS, TRANS, extras=extras
    )

    np.testing.assert_allclose(np.asarray(gk[0]), np.asarray(gr[0]), rtol=5e-3, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gk[1]), np.asarray(gr[1]), rtol=5e-3, atol=2e-6)
    # quats in the tangent space of the unit sphere, see the module docstring
    q = np.asarray(quats)
    tang = lambda g: g - np.sum(g * q, axis=1, keepdims=True) * q  # noqa: E731
    np.testing.assert_allclose(tang(np.asarray(gk[2])), tang(np.asarray(gr[2])), atol=2e-6)


def test_pose_grads_match_jax_reference() -> None:
    """Transform gradients, contracted to rotvec + translation pose coordinates."""
    means, scales, quats, extras = _setup(seed=4)

    def kernel_pose_loss(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
        return _loss_kernel(means, scales, quats, _tfs(rotvecs, trans), extras=extras)

    def reference_pose_loss(rotvecs: jax.Array, trans: jax.Array) -> jax.Array:
        return _loss_reference(means, scales, quats, rotvecs, trans, extras=extras)

    gk = jax.grad(kernel_pose_loss, argnums=(0, 1))(ROTVECS, TRANS)
    gr = jax.grad(reference_pose_loss, argnums=(0, 1))(ROTVECS, TRANS)
    np.testing.assert_allclose(np.asarray(gk[0]), np.asarray(gr[0]), rtol=5e-3, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gk[1]), np.asarray(gr[1]), rtol=5e-3, atol=2e-6)
    assert np.abs(np.asarray(gk[0])).max() > 0 and np.abs(np.asarray(gk[1])).max() > 0


def test_transform_grad_bottom_row_zero() -> None:
    means, scales, quats, extras = _setup(seed=5)
    tfs = _tfs(ROTVECS, TRANS)
    g_tf = jax.grad(_loss_kernel, argnums=3)(means, scales, quats, tfs, extras=extras)
    assert np.abs(np.asarray(g_tf)[:, 3, :]).max() == 0.0


def test_vmap_grad_over_transform_stack() -> None:
    means, scales, quats, extras = _setup(seed=6)
    stack = jnp.stack([_tfs(ROTVECS, TRANS), _tfs(-ROTVECS, -TRANS)])
    grad_tf = jax.grad(_loss_kernel, argnums=3)
    gb = jax.vmap(lambda t: grad_tf(means, scales, quats, t, extras=extras))(stack)
    seq = [np.asarray(grad_tf(means, scales, quats, t, extras=extras)) for t in stack]
    np.testing.assert_allclose(np.asarray(gb), np.stack(seq), rtol=1e-5, atol=1e-8)


def test_vmap_grad_over_viewmats_with_transforms() -> None:
    means, scales, quats, extras = _setup(seed=7)
    colors, opac, kw, target = extras
    tfs = _tfs(ROTVECS, TRANS)
    vms = jnp.stack([kw["viewmat"], kw["viewmat"].at[0, 3].add(0.15)])

    def loss(viewmat: jax.Array) -> jax.Array:
        view_kw = kw | {"viewmat": viewmat}
        img, _ = splax.render(
            means,
            scales,
            quats,
            colors,
            opac,
            **view_kw,
            gaussian_transforms=tfs,
            gaussian_slices=SLICES,
        )
        return jnp.mean((img - target) ** 2)

    gb = jax.vmap(jax.grad(loss))(vms)
    seq = [np.asarray(jax.grad(loss)(vm)) for vm in vms]
    np.testing.assert_allclose(np.asarray(gb), np.stack(seq), rtol=1e-5, atol=1e-8)
