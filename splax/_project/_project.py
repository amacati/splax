"""Differentiable projection stage.

``project`` composes the Warp projection kernels from ``splax._project._kernels`` with a
``jax.custom_vjp``, so ``jax.grad`` flows through it with respect to the gaussian parameters and the
camera pose. The forward rule reads each input's static perturbed bit, so the backward rule branches
in Python and launches only the kernels the requested gradients need.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.core
import jax.numpy as jnp

from splax._project._kernels import (
    _project_bwd_gaussians_ffi,
    _project_bwd_joint_ffi,
    _project_bwd_viewmat_ffi,
    _project_ffi,
)

# region public API


def project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    opacities: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float] | None = None,
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project gaussians to screen space with the opacity-aware tile intersection.

    The projection is differentiable through ``jax.custom_vjp``, and gradient selection follows
    ``jax.grad`` and its argnums. Perturbing the viewmat runs only the camera-pose accumulator,
    perturbing the means, scales, or quaternions runs only the gaussian-gradient kernel, and both
    together run the joint kernel. Without gradients the primal runs exactly as the forward-only
    path. Opacities feed the integer tile counts and carry no gradient through projection, their
    gradient flows through rasterization instead.

    Args:
        mean3ds: Gaussian centers, shape ``(N, 3)``.
        scales: Per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        viewmat: World-to-camera matrix, shape ``(4, 4)``.
        opacities: Gaussian opacities, one entry per gaussian.
        img_shape: Image size as ``(height, width)`` in pixels.
        f: Focal lengths ``(fx, fy)`` in pixels.
        c: Principal point ``(cx, cy)`` in pixels, defaulting to the image center.
        glob_scale: Global factor applied to all scales.
        clip_thresh: Near-plane clipping threshold.

    Returns:
        Tuple of the screen-space centers, depths, radii, conics, per-gaussian tile counts, and
        their inclusive prefix sum ``cum_tiles_hit``.
    """
    n = mean3ds.shape[0]
    if c is None:
        c = (img_shape[1] / 2, img_shape[0] / 2)
    return _project_differentiable(
        mean3ds,
        scales,
        quats,
        viewmat,
        opacities.reshape(n),
        int(n),
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
    )


def opacity_compensation(conics: jax.Array, radii: jax.Array, eps: float = 0.3) -> jax.Array:
    """Compute the Mip-Splatting anti-aliased opacity compensation factor per gaussian.

    The factor is ``sqrt(det(cov2d) / det(cov2d + eps I))``, the determinant ratio of the undilated
    2d covariance over the eps-dilated one that projection already applies. Multiplying it into the
    opacity before the blend cancels the artificial area inflation the dilation grants thin
    gaussians.

    The factor is computed in closed form from the projection's conics, so its gradient flows back
    to the scales, quaternions, and means through the existing projection vjp without any Warp
    kernel change. For a conic with entries ``a``, ``b``, and ``c`` the ratio is
    ``1 - eps (a + c) + eps^2 (a c - b^2)``, clipped to ``[0, 1]`` for float safety. Culled
    gaussians with non-positive radii get a factor of 1.

    Args:
        conics: Inverse 2d covariances from projection, shape ``(N, 3)``.
        radii: Screen-space radii, non-positive for culled gaussians.
        eps: Screen-space dilation the projection applies, in squared pixels.

    Returns:
        Per-gaussian compensation factor, shape ``(N,)``.
    """
    c = conics.reshape(-1, 3)
    a = c[:, 0]
    b = c[:, 1]
    cc = c[:, 2]
    rho2 = 1.0 - eps * (a + cc) + (eps * eps) * (a * cc - b * b)
    rho = jnp.sqrt(jnp.clip(rho2, 0.0, 1.0))
    valid = radii.reshape(-1) > 0
    return jnp.where(valid, rho, 1.0)


# region custom vjp


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def _project_differentiable(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return _project_call(
        mean3ds, scales, quats, viewmat, opac, n, img_shape, f, c, glob_scale, clip_thresh
    )


def _project_fwd_rule(
    mean3ds: jax.custom_derivatives.CustomVJPPrimal,
    scales: jax.custom_derivatives.CustomVJPPrimal,
    quats: jax.custom_derivatives.CustomVJPPrimal,
    viewmat: jax.custom_derivatives.CustomVJPPrimal,
    opac: jax.custom_derivatives.CustomVJPPrimal,
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple]:
    # Under symbolic_zeros the five differentiable args arrive as CustomVJPPrimal
    # with a value and a static perturbed bit. The perturbation pattern is encoded
    # as () or None in the residuals. Both are pytree structure, not leaves, so
    # they stay concrete under jit and the backward rule branches in Python.
    # opacities are never differentiated through project, they only drive integer
    # tile counts. The opacity gradient flows through rasterize.
    m, s, q = mean3ds.value, scales.value, quats.value
    vm, op = viewmat.value, opac.value
    pert_gaussians = () if (mean3ds.perturbed or scales.perturbed or quats.perturbed) else None
    pert_viewmat = () if viewmat.perturbed else None
    out = _project_call(m, s, q, vm, op, n, img_shape, f, c, glob_scale, clip_thresh)
    _xys, _depths, radii, conics, _nth, _cum = out
    return out, (m, s, q, vm, radii, conics, pert_gaussians, pert_viewmat)


def _materialize(ct: jax.Array | jax.custom_derivatives.SymbolicZero) -> jax.Array:
    # symbolic_zeros hands the backward rule SymbolicZero objects for cotangents
    # not involved in differentiation, e.g. the depth channel under a plain image
    # loss. The Warp kernels need concrete arrays, so those become dense zeros of
    # the cotangent's own shape and dtype, correct under batching too. XLA folds
    # the zeros away.
    if isinstance(ct, jax.custom_derivatives.SymbolicZero):
        aval = cast("jax.core.ShapedArray", ct.aval)
        return jnp.zeros(aval.shape, aval.dtype)
    return cast("jax.Array", ct)


def _project_bwd_rule(
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    residuals: tuple,
    cotangents: tuple,
) -> tuple[jax.Array | None, ...]:
    # The perturbation pattern was recorded statically in the forward rule, so this
    # branch is pure Python and only the needed backward kernels launch.
    m, s, q, vm, radii, conics, pert_gaussians, pert_viewmat = residuals
    v_xys, v_depths, _v_radii, v_conics, _v_nth, _v_cum = cotangents
    r = radii.reshape(n).astype(jnp.int32)
    vx = _materialize(v_xys)
    vd = _materialize(v_depths).reshape(-1)
    vc = _materialize(v_conics)
    v_mean: jax.Array | None = None
    v_scale: jax.Array | None = None
    v_quat: jax.Array | None = None
    v_viewmat: jax.Array | None = None
    fx, fy = float(f[0]), float(f[1])
    if pert_gaussians is not None and pert_viewmat is not None:
        v_mean, v_scale, v_quat, v_viewmat = _project_bwd_joint_ffi(
            m,
            s,
            q,
            vm,
            r,
            conics,
            vx,
            vd,
            vc,
            int(n),
            fx,
            fy,
            float(glob_scale),
            output_dims={"v_mean3d": n, "v_scale": n, "v_quat": n, "v_viewmat": (4, 4)},
        )
    elif pert_viewmat is not None:
        (v_viewmat,) = _project_bwd_viewmat_ffi(
            m,
            s,
            q,
            vm,
            r,
            conics,
            vx,
            vd,
            vc,
            int(n),
            fx,
            fy,
            float(glob_scale),
            output_dims=(4, 4),
        )
    elif pert_gaussians is not None:
        v_mean, v_scale, v_quat = _project_bwd_gaussians_ffi(
            m, s, q, vm, r, conics, vx, vd, vc, int(n), fx, fy, float(glob_scale), output_dims=n
        )
    return (v_mean, v_scale, v_quat, v_viewmat, None)


_project_differentiable.defvjp(_project_fwd_rule, _project_bwd_rule, symbolic_zeros=True)


def _project_call(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    gaussian_transforms: jax.Array | None = None,
    transform_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # gaussian_transforms and transform_ids enable the rigid transforms of the
    # grad-free inference path. The differentiable path never passes them, its
    # backward recomputes geometry from the untransformed residuals. Without
    # transforms the kernel takes dummy operands and skips the transform block.
    H, W = img_shape
    if gaussian_transforms is None:
        gaussian_transforms = jnp.zeros((1, 4, 4), jnp.float32)
        transform_ids = jnp.full((1,), -1, jnp.int32)
        num_transforms = 1
        has_transforms = False
    else:
        assert transform_ids is not None  # built alongside the transforms
        num_transforms = gaussian_transforms.shape[-3]
        has_transforms = True
    xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit = _project_ffi(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        gaussian_transforms,
        transform_ids,
        int(n),
        int(num_transforms),
        bool(has_transforms),
        int(H),
        int(W),
        float(f[0]),
        float(f[1]),
        float(c[0]),
        float(c[1]),
        float(glob_scale),
        float(clip_thresh),
        output_dims=n,
    )
    depths = depths.reshape(n, 1)
    radii = radii.reshape(n, 1)
    num_tiles_hit = num_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    cum_tiles_hit = cum_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    return xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit
