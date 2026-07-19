"""gsplat reference adapters for the splax parity tests.

gsplat (https://github.com/nerfstudio-project/gsplat) is the external CUDA/torch
reference the splax Warp kernels are cross-checked against. This module wraps
gsplat's ``fully_fused_projection`` and ``rasterization`` so they take and return
the same quantities as ``splax.project`` / ``splax.render``, converting through
numpy at the torch/jax boundary.

Convention differences, documented once here (they hold for every test):

  - Framework bridge: gsplat is torch. Inputs arrive as jax/numpy arrays. We move
    them to CUDA torch tensors and return results as numpy (``.detach().cpu()``).
    numpy (host) rather than dlpack keeps the bridge device- and version-robust.
  - viewmat: both use a world-to-camera, OpenCV-convention (+z forward, +y down)
    4x4 matrix, so it passes through unchanged. gsplat wants a camera batch axis,
    so we pass ``viewmat[None]`` (C=1) and squeeze it back out.
  - Quaternion order: gsplat and splax both store quats scalar-first (wxyz), so no
    reordering is needed. gsplat normalizes internally. We pass them as given.
  - Intrinsics: gsplat takes a 3x3 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] rather
    than splax's separate ``f=(fx, fy)`` / ``c=(cx, cy)``.
  - glob_scale: splax scales the covariance by ``glob_scale``. gsplat has no such
    argument, so we fold it in as ``scales * glob_scale``.
  - clip_thresh maps to gsplat's ``near_plane`` (min camera-space z).
  - eps2d: both dilate the projected 2D covariance by 0.3 px before inversion, so
    the returned conics (inverse-2D-cov upper triangle a, b, c) are directly
    comparable. gsplat's default ``eps2d=0.3`` is passed explicitly.
  - Opacities / colors: both take opacities in [0, 1] and linear RGB colors in
    [0, 1] applied directly (we pass ``sh_degree=None`` so colors are used as-is,
    no SH evaluation). splax opacities are (N, 1). gsplat wants (N,), so we ravel.
"""

from __future__ import annotations

import gsplat
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# torch and gsplat are required test dependencies. A missing install fails
# collection loudly instead of skipping the parity tests.
import torch

_PROBE = None


def _probe() -> tuple[bool, str]:
    """Cached (ok, reason) for whether gsplat can actually run a projection.

    gsplat loads (or JIT-compiles) a CUDA extension. The import can succeed while the
    extension is unavailable (no toolkit / prebuilt mismatch), surfacing only on the
    first real call. We probe a tiny projection once and cache the verdict.
    """
    global _PROBE
    if _PROBE is None:
        try:
            project(
                jnp.zeros((1, 3)),
                jnp.ones((1, 3)) * 0.05,
                jnp.array([[1.0, 0.0, 0.0, 0.0]]),
                jnp.eye(4).at[2, 3].set(5.0),
                img_shape=(16, 16),
                f=(16.0, 16.0),
                c=(8, 8),
                glob_scale=1.0,
                clip_thresh=0.01,
            )
            _PROBE = (True, "")
        except Exception as e:  # noqa: BLE001 - any failure means "unavailable"
            _PROBE = (False, repr(e))
    return _PROBE


def require_working(allow_module_level: bool = False) -> None:
    """Fail unless the gsplat CUDA extension actually runs a projection.

    gsplat is a required test reference. A broken extension fails the parity
    tests loudly instead of skipping them, so environment problems get fixed
    rather than hidden.
    """
    del allow_module_level  # failing needs no module-level special case
    ok, reason = _probe()
    if not ok:
        pytest.fail(f"gsplat CUDA reference unavailable, fix the env: {reason}")


def _np(x: jax.Array | np.ndarray) -> np.ndarray:
    # np.array (copy) rather than asarray: jax arrays are read-only, and torch warns
    # when wrapping a non-writable buffer.
    return np.array(x, dtype=np.float32)


def _ft(a: jax.Array | np.ndarray) -> torch.Tensor:
    """Convert a jax/numpy array to a float32 CUDA torch tensor."""
    return torch.as_tensor(_np(a), dtype=torch.float32, device="cuda")


def _K(f: tuple[float, float], c: tuple[float, float]) -> np.ndarray:
    fx, fy = f
    cx, cy = c
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float32)


def project(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    viewmat: jax.Array | np.ndarray,
    *,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gsplat ``fully_fused_projection`` in splax.project's terms.

    Returns numpy (radii, means2d, depths, conics), each aligned to the N input
    gaussians (camera axis squeezed). ``radii`` is gsplat's per-axis pixel radius
    (int). Use ``(radii > 0).any(-1)`` as the visibility mask.
    """
    H, W = img_shape
    radii, means2d, depths, conics, _comp = gsplat.fully_fused_projection(
        _ft(means),
        None,
        _ft(quats),
        _ft(scales) * float(glob_scale),
        _ft(viewmat)[None],
        _ft(_K(f, c))[None],
        W,
        H,
        eps2d=0.3,
        near_plane=float(clip_thresh),
        packed=False,
        calc_compensations=False,
    )
    radii_n, means2d_n, depths_n, conics_n = (
        x[0].detach().cpu().numpy() for x in (radii, means2d, depths, conics)
    )
    return radii_n, means2d_n, depths_n, conics_n


def render(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> np.ndarray:
    """Gsplat ``rasterization`` in splax.render's terms. Returns numpy (H, W, 3)."""
    H, W = img_shape
    out, alpha, _meta = gsplat.rasterization(
        _ft(means),
        _ft(quats),
        _ft(scales) * float(glob_scale),
        _ft(opacities).reshape(-1),
        _ft(colors),
        _ft(viewmat)[None],
        _ft(_K(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    # gsplat returns colors composited over black plus the accumulated alpha. Put it
    # on the requested background exactly as splax.render does (composite over bg).
    img = out[0] + (1.0 - alpha[0]) * _ft(background).reshape(3)
    return img.detach().cpu().numpy()


def viewmat_grad(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    target: np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> np.ndarray:
    """Gsplat gradient of ``mean((render(viewmat) - target) ** 2)`` wrt the viewmat.

    Returns the (4, 4) numpy gradient.
    """
    H, W = img_shape
    viewmat_t = _ft(viewmat).requires_grad_(True)
    out, alpha, _meta = gsplat.rasterization(
        _ft(means),
        _ft(quats),
        _ft(scales) * float(glob_scale),
        _ft(opacities).reshape(-1),
        _ft(colors),
        viewmat_t[None],
        _ft(_K(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    img = out[0] + (1.0 - alpha[0]) * _ft(background).reshape(3)
    ((img - _ft(target)) ** 2).mean().backward()
    return viewmat_t.grad.detach().cpu().numpy()


def grad(
    means: jax.Array | np.ndarray,
    scales: jax.Array | np.ndarray,
    quats: jax.Array | np.ndarray,
    colors: jax.Array | np.ndarray,
    opacities: jax.Array | np.ndarray,
    *,
    viewmat: jax.Array | np.ndarray,
    background: jax.Array | np.ndarray,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gsplat grads with respect to (means, scales, quats, colors, opacities).

    ``weight is None`` gives loss = sum(image), otherwise loss = mean(weight * image**2),
    the same two scalar losses the splax grad-parity test uses. Returns a tuple of
    five numpy grad arrays aligned to the splax inputs (opacities grad reshaped to
    the (N, 1) input layout).
    """
    H, W = img_shape
    n = _np(means).shape[0]
    means_t = _ft(means).requires_grad_(True)
    scales_t = _ft(scales).requires_grad_(True)
    quats_t = _ft(quats).requires_grad_(True)
    colors_t = _ft(colors).requires_grad_(True)
    opac_t = _ft(opacities).reshape(-1).requires_grad_(True)

    out, alpha, _meta = gsplat.rasterization(
        means_t,
        quats_t,
        scales_t * float(glob_scale),
        opac_t,
        colors_t,
        _ft(viewmat)[None],
        _ft(_K(f, c))[None],
        W,
        H,
        near_plane=float(clip_thresh),
        eps2d=0.3,
        render_mode="RGB",
    )
    img = out[0] + (1.0 - alpha[0]) * _ft(background).reshape(3)
    loss = img.sum() if weight is None else (_ft(weight) * img**2).mean()
    loss.backward()

    # gsplat folds glob_scale into scales*glob_scale, so d/d(scales) already carries
    # the factor. This matches splax which scales inside the kernel.
    grads = (means_t, scales_t, quats_t, colors_t)
    out_g = [x.grad.detach().cpu().numpy() for x in grads]
    out_g.append(opac_t.grad.detach().cpu().numpy().reshape(n, 1))
    return tuple(out_g)
