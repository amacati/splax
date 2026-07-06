"""Warp projection stage, forward and backward.

The forward is a single Warp kernel wrapped as a JAX FFI call. It projects each
gaussian to screen space and counts the tiles its opacity-aware tight ellipse
touches, followed by an inclusive prefix scan for cum_tiles_hit, all on the
XLA-provided CUDA stream. The backward is one jax.custom_vjp with symbolic_zeros.
The forward rule reads each input's static perturbed bit, so the backward rule
branches in Python and launches only the kernels the requested gradients need.
Gradient selection is therefore pure jax.grad / argnums.
"""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental.ffi import jax_callable, JaxCallableGraphMode

from splax._intersect import (
    ALPHA_THRESHOLD,
    BLOCK_WIDTH,
    GAUSSIAN_EXTEND_SQ,
    _ellipse_setup,
    _ellipse_tile_count,
)


@wp.func
def _quat_to_rotmat(q: wp.vec4) -> wp.mat33:
    # quats are stored scalar-first (w, x, y, z)
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


@wp.kernel
def _project_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: wp.int32,
    num_transforms: wp.int32,
    has_transforms: wp.int32,
    sel_means: wp.int32,
    sel_scales: wp.int32,
    sel_quats: wp.int32,
    sel_view: wp.int32,
    sel_opac: wp.int32,
    sel_transforms: wp.int32,
    img_h: wp.int32,
    img_w: wp.int32,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    glob_scale: wp.float32,
    clip_thresh: wp.float32,
    # outputs
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    num_tiles_hit: wp.array[wp.int32],
):
    # Launched over B*N flat threads, no host loop. bid is the batch element, gid
    # the gaussian within the shared scene. Each input array is either batched
    # (leading dim B*N or 4B, selector 1, indexed at the flat idx) or broadcast
    # (leading dim N or 4, selector 0, indexed at gid or row 0). Outputs are always
    # full batch. For B=1 this is exactly the unbatched path.
    idx = wp.tid()
    n = num_gaussians
    bid = idx // n
    gid = idx % n
    m_idx = wp.where(sel_means != 0, idx, gid)
    s_idx = wp.where(sel_scales != 0, idx, gid)
    q_idx = wp.where(sel_quats != 0, idx, gid)
    o_idx = wp.where(sel_opac != 0, idx, gid)
    vb = wp.where(sel_view != 0, bid, 0) * 4  # row offset into (4B, 4) viewmat

    radii[idx] = 0
    num_tiles_hit[idx] = 0
    xys[idx] = wp.vec2(0.0, 0.0)
    depths[idx] = 0.0
    conics[idx] = wp.vec3(0.0, 0.0, 0.0)

    mean = means3d[m_idx]

    # Optional rigid transforms tied to gaussians. transform_ids maps each
    # gaussian to the transform it follows, -1 leaves it static. The transform
    # stack is either broadcast (K, 4, 4) or batched (B*K, 4, 4), selected like
    # the viewmat, so a batched render moves the same gaussians differently in
    # every batch element. The mean moves in world space here, the covariance
    # follows via M below. With has_transforms == 0 nothing in this block
    # executes and the kernel matches the plain path exactly.
    R_tf = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    moved = wp.int32(0)
    if has_transforms != 0:
        tf_id = transform_ids[gid]
        if tf_id >= 0:
            tf_idx = wp.where(sel_transforms != 0, bid * num_transforms + tf_id, tf_id)
            R_tf = wp.mat33(
                gaussian_transforms[tf_idx, 0, 0],
                gaussian_transforms[tf_idx, 0, 1],
                gaussian_transforms[tf_idx, 0, 2],
                gaussian_transforms[tf_idx, 1, 0],
                gaussian_transforms[tf_idx, 1, 1],
                gaussian_transforms[tf_idx, 1, 2],
                gaussian_transforms[tf_idx, 2, 0],
                gaussian_transforms[tf_idx, 2, 1],
                gaussian_transforms[tf_idx, 2, 2],
            )
            t_tf = wp.vec3(
                gaussian_transforms[tf_idx, 0, 3],
                gaussian_transforms[tf_idx, 1, 3],
                gaussian_transforms[tf_idx, 2, 3],
            )
            mean = R_tf * mean + t_tf
            moved = wp.int32(1)

    mx = mean[0]
    my = mean[1]
    mz = mean[2]

    # near-plane clip on p_view = viewmat @ mean (row-major 4x3)
    pvx = (
        viewmat[vb + 0, 0] * mx
        + viewmat[vb + 0, 1] * my
        + viewmat[vb + 0, 2] * mz
        + viewmat[vb + 0, 3]
    )
    pvy = (
        viewmat[vb + 1, 0] * mx
        + viewmat[vb + 1, 1] * my
        + viewmat[vb + 1, 2] * mz
        + viewmat[vb + 1, 3]
    )
    pvz = (
        viewmat[vb + 2, 0] * mx
        + viewmat[vb + 2, 1] * my
        + viewmat[vb + 2, 2] * mz
        + viewmat[vb + 2, 3]
    )
    if pvz <= clip_thresh:
        return

    # world covariance V3 = M M^T with M = R diag(glob_scale * scale)
    R = _quat_to_rotmat(quats[q_idx])
    s = scales[s_idx]
    sx = glob_scale * s[0]
    sy = glob_scale * s[1]
    sz = glob_scale * s[2]
    M = wp.mat33(
        R[0, 0] * sx,
        R[0, 1] * sy,
        R[0, 2] * sz,
        R[1, 0] * sx,
        R[1, 1] * sy,
        R[1, 2] * sz,
        R[2, 0] * sx,
        R[2, 1] * sy,
        R[2, 2] * sz,
    )
    if moved != 0:
        # rotate the covariance factor, V3 becomes R_tf V3 R_tf^T
        M = R_tf * M
    V3 = M * wp.transpose(M)

    # EWA projection of the covariance
    W = wp.mat33(
        viewmat[vb + 0, 0],
        viewmat[vb + 0, 1],
        viewmat[vb + 0, 2],
        viewmat[vb + 1, 0],
        viewmat[vb + 1, 1],
        viewmat[vb + 1, 2],
        viewmat[vb + 2, 0],
        viewmat[vb + 2, 1],
        viewmat[vb + 2, 2],
    )
    tan_fovx = 0.5 * wp.float32(img_w) / fx
    tan_fovy = 0.5 * wp.float32(img_h) / fy
    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    tx = pvx
    ty = pvy
    tz = pvz
    tx = tz * wp.min(lim_x, wp.max(-lim_x, tx / tz))
    ty = tz * wp.min(lim_y, wp.max(-lim_y, ty / tz))
    rz = 1.0 / tz
    rz2 = rz * rz
    J = wp.mat33(
        fx * rz,
        0.0,
        -fx * tx * rz2,
        0.0,
        fy * rz,
        -fy * ty * rz2,
        0.0,
        0.0,
        0.0,
    )
    T = J * W
    cov = T * V3 * wp.transpose(T)
    # 0.3 px screen-space dilation, the standard 3DGS low-pass guard
    cxx = cov[0, 0] + 0.3
    cxy = cov[0, 1]
    cyy = cov[1, 1] + 0.3

    det = cxx * cyy - cxy * cxy
    if det == 0.0:
        return
    inv_det = 1.0 / det
    conic = wp.vec3(cyy * inv_det, -cxy * inv_det, cxx * inv_det)

    # pixel center from the unclamped p_view
    rw = 1.0 / (pvz + 1e-6)
    center_x = (pvx * rw) * fx + cx
    center_y = (pvy * rw) * fy + cy

    tb_x = (img_w + BLOCK_WIDTH - 1) / BLOCK_WIDTH
    tb_y = (img_h + BLOCK_WIDTH - 1) / BLOCK_WIDTH

    # Opacity-aware tight tile intersection. The rasterize key emission
    # walks the identical ellipse via the shared _intersect helpers, so the counted
    # and emitted tile totals match exactly.
    opac = opacities[o_idx]
    if opac < ALPHA_THRESHOLD:
        return  # alpha < 1/255 everywhere, contributes nothing
    t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
    ext = wp.sqrt(t)
    radius_x = wp.ceil(ext * wp.sqrt(cxx))
    radius_y = wp.ceil(ext * wp.sqrt(cyy))
    if radius_x <= 0.0 and radius_y <= 0.0:
        return
    if (
        center_x + radius_x <= 0.0
        or center_x - radius_x >= wp.float32(img_w)
        or center_y + radius_y <= 0.0
        or center_y - radius_y >= wp.float32(img_h)
    ):
        return
    setup = _ellipse_setup(
        conic[0], conic[1], conic[2], t, center_x, center_y, tb_x, tb_y
    )
    count = _ellipse_tile_count(setup)
    if count <= 0:
        return
    num_tiles_hit[idx] = count
    depths[idx] = pvz
    radii[idx] = wp.int32(wp.max(radius_x, radius_y))
    xys[idx] = wp.vec2(center_x, center_y)
    conics[idx] = conic


def _project_launch(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: int,
    num_transforms: int,
    has_transforms: int,
    img_h: int,
    img_w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    glob_scale: float,
    clip_thresh: float,
    # outputs
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    num_tiles_hit: wp.array[wp.int32],
    cum_tiles_hit: wp.array[wp.int32],
) -> None:
    # N is passed statically because jax.vmap hides the batch axis from this
    # wrapper. B is recovered from an output shape, always full batch under
    # expand_dims. Each input is batched (leading dim above base, selector 1) or
    # broadcast (equal to base, selector 0).
    n = num_gaussians
    total = xys.shape[0]  # B*N
    sel_means = 1 if means3d.shape[0] > n else 0
    sel_scales = 1 if scales.shape[0] > n else 0
    sel_quats = 1 if quats.shape[0] > n else 0
    sel_view = 1 if viewmat.shape[0] > 4 else 0
    sel_opac = 1 if opacities.shape[0] > n else 0
    sel_transforms = 1 if gaussian_transforms.shape[0] > num_transforms else 0
    wp.launch(
        _project_kernel,
        dim=total,
        inputs=[
            means3d,
            scales,
            quats,
            viewmat,
            opacities,
            gaussian_transforms,
            transform_ids,
            n,
            num_transforms,
            has_transforms,
            sel_means,
            sel_scales,
            sel_quats,
            sel_view,
            sel_opac,
            sel_transforms,
            img_h,
            img_w,
            fx,
            fy,
            cx,
            cy,
            glob_scale,
            clip_thresh,
        ],
        outputs=[xys, depths, radii, conics, num_tiles_hit],
    )
    # One global inclusive prefix sum over the flattened B*N tile counts, so all
    # images' intersections are laid out contiguously for a single global sort.
    wp.utils.array_scan(num_tiles_hit, cum_tiles_hit, inclusive=True)


# graph_mode=WARP keeps array_scan's host-side temp management out of the JAX graph
# capture. Warp captures and replays a CUDA graph keyed on buffer addresses and
# re-captures when a batch size or address changes.
# vmap_method="expand_dims" makes batching native. Under jax.vmap every operand
# gains a leading batch axis that warp's FFI callback collapses into the leading
# array dim, and the callable launches once over B*N. A non-vmapped call keeps
# base rank and reduces to the unbatched path exactly.
_project_ffi = jax_callable(
    _project_launch,
    num_outputs=6,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)


# Backward pass. All vjp math lives in shared wp.func helpers so the three kernel
# variants (gaussians only, viewmat only, joint) carry no duplicated math. Each
# variant composes the helpers it needs and writes the grads it owns.
#
# cov3d is recomputed in the backward rather than saved. It is deterministic,
# bit-identical, and saves a 6N residual. Blur compensation is dropped (zero
# cotangent in the render path). The EWA J is rebuilt from the unclamped
# camera-space position, the standard gsplat approximation.
#
# Viewmat gradient, derived from the forward and validated by finite differences.
# With t = W mean + trans, v_p the total gradient wrt t and v_T the gradient wrt
# T = J W, only the top 12 viewmat entries are differentiable and
#     v_trans = v_p
#     v_R     = outer(v_p, mean) + J^T v_T
# The outer product is the dt/dW term. J^T v_T holds J fixed, J's own W dependence
# flows through v_p via t.
#
# Every gaussian contributes to one shared 12-float v_viewmat per image. Each
# block reduces its threads' contributions with wp.tile_sum and thread 0 issues
# one atomic per entry per block. Projection has uniform per-gaussian work and no
# early termination, so the block barrier is amortised and the reduction beats
# plain per-thread atomics 20 to 110x. The rasterize backward is the opposite
# case, see _rasterize.py.

VIEW_BLOCK = wp.constant(256)  # threads per block for the tile_sum viewmat reduce


@wp.struct
class _Geom:
    W: wp.mat33  # upper-left 3x3 of the row-major viewmat (camera rotation)
    R: wp.mat33  # quaternion rotation
    M: wp.mat33  # R diag(glob_scale * scale)
    V: wp.mat33  # world covariance M M^T
    J: wp.mat33  # EWA jacobian, rebuilt from the unclamped position
    T: wp.mat33  # J W
    tx: wp.float32  # camera-space position (unclamped)
    ty: wp.float32
    tz: wp.float32
    rz2: wp.float32
    rz3: wp.float32
    sx: wp.float32  # glob_scale * scale
    sy: wp.float32
    sz: wp.float32


@wp.func
def _recompute_geom(
    mean: wp.vec3,
    quat: wp.vec4,
    scale: wp.vec3,
    W: wp.mat33,
    trans: wp.vec3,
    glob_scale: wp.float32,
    fx: wp.float32,
    fy: wp.float32,
) -> _Geom:
    g = _Geom()
    g.W = W
    tx = W[0, 0] * mean[0] + W[0, 1] * mean[1] + W[0, 2] * mean[2] + trans[0]
    ty = W[1, 0] * mean[0] + W[1, 1] * mean[1] + W[1, 2] * mean[2] + trans[1]
    tz = W[2, 0] * mean[0] + W[2, 1] * mean[1] + W[2, 2] * mean[2] + trans[2]
    rz = 1.0 / tz
    rz2 = rz * rz
    g.tx = tx
    g.ty = ty
    g.tz = tz
    g.rz2 = rz2
    g.rz3 = rz2 * rz
    J = wp.mat33(
        fx * rz,
        0.0,
        -fx * tx * rz2,
        0.0,
        fy * rz,
        -fy * ty * rz2,
        0.0,
        0.0,
        0.0,
    )
    g.J = J
    g.T = J * W
    R = _quat_to_rotmat(quat)
    g.R = R
    sx = glob_scale * scale[0]
    sy = glob_scale * scale[1]
    sz = glob_scale * scale[2]
    g.sx = sx
    g.sy = sy
    g.sz = sz
    M = wp.mat33(
        R[0, 0] * sx,
        R[0, 1] * sy,
        R[0, 2] * sz,
        R[1, 0] * sx,
        R[1, 1] * sy,
        R[1, 2] * sz,
        R[2, 0] * sx,
        R[2, 1] * sy,
        R[2, 2] * sz,
    )
    g.M = M
    g.V = M * wp.transpose(M)
    return g


@wp.func
def _vcov2d_from_conic(conic: wp.vec3, v_conic: wp.vec3) -> wp.vec3:
    # conic to cov2d vjp, v_cov2d = -X G X with X the conic and G the cotangent
    cx = conic[0]
    cy = conic[1]
    cz = conic[2]
    g00 = v_conic[0]
    g01 = 0.5 * v_conic[1]
    g11 = v_conic[2]
    XG00 = cx * g00 + cy * g01
    XG01 = cx * g01 + cy * g11
    XG10 = cy * g00 + cz * g01
    XG11 = cy * g01 + cz * g11
    S00 = XG00 * cx + XG01 * cy
    S01 = XG00 * cy + XG01 * cz
    S10 = XG10 * cx + XG11 * cy
    S11 = XG10 * cy + XG11 * cz
    return wp.vec3(-S00, -(S10 + S01), -S11)


@wp.func
def _proj_vjp(
    g: _Geom,
    fx: wp.float32,
    fy: wp.float32,
    v_xy: wp.vec2,
    v_depth: wp.float32,
    vcov2d: wp.vec3,
):
    # Returns (v_p, v_T, v_V). v_p is the gradient wrt the camera-space position
    # (pixel + depth + EWA terms), used by the world-mean grad and the viewmat
    # translation grad. v_T is the gradient wrt T = J W, used by the viewmat
    # rotation grad. v_V is the gradient wrt the world covariance, used by
    # scale and quat.
    tx = g.tx
    ty = g.ty
    tz = g.tz
    rz2 = g.rz2
    rz3 = g.rz3
    rw = 1.0 / (tz + 1e-6)
    vpx = fx * v_xy[0]
    vpy = fy * v_xy[1]
    vvx = vpx * rw
    vvy = vpy * rw
    vvz = -(vpx * tx + vpy * ty) * rw * rw
    # the depth cotangent adds onto the z component of the position grad
    vvz = vvz + v_depth
    v_cov = wp.mat33(
        vcov2d[0],
        0.5 * vcov2d[1],
        0.0,
        0.5 * vcov2d[1],
        vcov2d[2],
        0.0,
        0.0,
        0.0,
        0.0,
    )
    Tt = wp.transpose(g.T)
    v_V = Tt * v_cov * g.T
    v_T = v_cov * g.T * g.V + wp.transpose(v_cov) * g.T * g.V
    v_J = v_T * wp.transpose(g.W)
    v_t_x = -fx * rz2 * v_J[0, 2]
    v_t_y = -fy * rz2 * v_J[1, 2]
    v_t_z = (
        -fx * rz2 * v_J[0, 0]
        + 2.0 * fx * tx * rz3 * v_J[0, 2]
        - fy * rz2 * v_J[1, 1]
        + 2.0 * fy * ty * rz3 * v_J[1, 2]
    )
    v_p = wp.vec3(vvx + v_t_x, vvy + v_t_y, vvz + v_t_z)
    return v_p, v_T, v_V


@wp.func
def _scale_quat_vjp(g: _Geom, quat: wp.vec4, v_V: wp.mat33, glob_scale: wp.float32):
    # Returns (v_scale, v_quat)
    vc0 = v_V[0, 0]
    vc1 = v_V[0, 1] + v_V[1, 0]
    vc2 = v_V[0, 2] + v_V[2, 0]
    vc3 = v_V[1, 1]
    vc4 = v_V[1, 2] + v_V[2, 1]
    vc5 = v_V[2, 2]
    v_Vc = wp.mat33(
        vc0,
        0.5 * vc1,
        0.5 * vc2,
        0.5 * vc1,
        vc3,
        0.5 * vc4,
        0.5 * vc2,
        0.5 * vc4,
        vc5,
    )
    v_M = (v_Vc * g.M) * 2.0
    R = g.R
    v_scale = wp.vec3(
        (R[0, 0] * v_M[0, 0] + R[1, 0] * v_M[1, 0] + R[2, 0] * v_M[2, 0]) * glob_scale,
        (R[0, 1] * v_M[0, 1] + R[1, 1] * v_M[1, 1] + R[2, 1] * v_M[2, 1]) * glob_scale,
        (R[0, 2] * v_M[0, 2] + R[1, 2] * v_M[1, 2] + R[2, 2] * v_M[2, 2]) * glob_scale,
    )
    v_R = wp.mat33(
        v_M[0, 0] * g.sx,
        v_M[0, 1] * g.sy,
        v_M[0, 2] * g.sz,
        v_M[1, 0] * g.sx,
        v_M[1, 1] * g.sy,
        v_M[1, 2] * g.sz,
        v_M[2, 0] * g.sx,
        v_M[2, 1] * g.sy,
        v_M[2, 2] * g.sz,
    )
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    vq_w = 2.0 * (
        x * (v_R[2, 1] - v_R[1, 2])
        + y * (v_R[0, 2] - v_R[2, 0])
        + z * (v_R[1, 0] - v_R[0, 1])
    )
    vq_x = 2.0 * (
        -2.0 * x * (v_R[1, 1] + v_R[2, 2])
        + y * (v_R[1, 0] + v_R[0, 1])
        + z * (v_R[2, 0] + v_R[0, 2])
        + w * (v_R[2, 1] - v_R[1, 2])
    )
    vq_y = 2.0 * (
        x * (v_R[1, 0] + v_R[0, 1])
        - 2.0 * y * (v_R[0, 0] + v_R[2, 2])
        + z * (v_R[2, 1] + v_R[1, 2])
        + w * (v_R[0, 2] - v_R[2, 0])
    )
    vq_z = 2.0 * (
        x * (v_R[2, 0] + v_R[0, 2])
        + y * (v_R[2, 1] + v_R[1, 2])
        - 2.0 * z * (v_R[0, 0] + v_R[1, 1])
        + w * (v_R[1, 0] - v_R[0, 1])
    )
    return v_scale, wp.vec4(vq_w, vq_x, vq_y, vq_z)


@wp.func
def _load_W_trans(viewmat: wp.array2d[wp.float32], vb: wp.int32):
    # Upper-left 3x3 W and translation from the viewmat row block starting at vb
    W = wp.mat33(
        viewmat[vb + 0, 0],
        viewmat[vb + 0, 1],
        viewmat[vb + 0, 2],
        viewmat[vb + 1, 0],
        viewmat[vb + 1, 1],
        viewmat[vb + 1, 2],
        viewmat[vb + 2, 0],
        viewmat[vb + 2, 1],
        viewmat[vb + 2, 2],
    )
    trans = wp.vec3(viewmat[vb + 0, 3], viewmat[vb + 1, 3], viewmat[vb + 2, 3])
    return W, trans


@wp.func
def _view_grad(g: _Geom, mean: wp.vec3, v_p: wp.vec3, v_T: wp.mat33):
    # v_R = outer(v_p, mean) + J^T v_T and v_trans = v_p, see header derivation
    v_R = wp.outer(v_p, mean) + wp.transpose(g.J) * v_T
    return v_R, v_p


# Gaussian-grad kernel. Launched over B*N flat threads like the forward. Gaussian
# grads are written per view at the flat idx. When the gaussian inputs were
# broadcast under vmap, JAX reduces the per-view grads over the batch axis.
@wp.kernel
def _project_bwd_gaussians_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy_in: wp.array[wp.vec2],
    v_depth_in: wp.array[wp.float32],
    v_conic_in: wp.array[wp.vec3],
    num_gaussians: wp.int32,
    sel_means: wp.int32,
    sel_scales: wp.int32,
    sel_quats: wp.int32,
    sel_view: wp.int32,
    sel_radii: wp.int32,
    sel_conics: wp.int32,
    sel_vxy: wp.int32,
    sel_vdepth: wp.int32,
    sel_vconic: wp.int32,
    fx: wp.float32,
    fy: wp.float32,
    glob_scale: wp.float32,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
):
    idx = wp.tid()
    n = num_gaussians
    bid = idx // n
    gid = idx % n
    # Every operand is either batched (own leading dim B*N, read at idx) or
    # broadcast (leading dim N, read at gid). A cotangent can arrive broadcast even
    # when the geometry is batched. The depth cotangent is zero for an image loss,
    # so JAX hands it back at size N, and indexing it at idx would read OOB.
    r_idx = wp.where(sel_radii != 0, idx, gid)
    if radii[r_idx] <= 0:
        return  # culled gaussian keeps its pre-zeroed grad
    m_idx = wp.where(sel_means != 0, idx, gid)
    s_idx = wp.where(sel_scales != 0, idx, gid)
    q_idx = wp.where(sel_quats != 0, idx, gid)
    c_idx = wp.where(sel_conics != 0, idx, gid)
    vx_idx = wp.where(sel_vxy != 0, idx, gid)
    vd_idx = wp.where(sel_vdepth != 0, idx, gid)
    vc_idx = wp.where(sel_vconic != 0, idx, gid)
    vb = wp.where(sel_view != 0, bid, 0) * 4
    mean = means3d[m_idx]
    W, trans = _load_W_trans(viewmat, vb)
    g = _recompute_geom(mean, quats[q_idx], scales[s_idx], W, trans, glob_scale, fx, fy)
    vcov2d = _vcov2d_from_conic(conics[c_idx], v_conic_in[vc_idx])
    v_p, v_T, v_V = _proj_vjp(g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
    v_mean3d[idx] = wp.transpose(g.W) * v_p
    vs, vq = _scale_quat_vjp(g, quats[q_idx], v_V, glob_scale)
    v_scale[idx] = vs
    v_quat[idx] = vq


# Viewmat kernel. Skips the whole scale/quat/cov3d grad chain and the gaussian
# grad arrays, accumulating only v_viewmat. The accumulator is per image. Under
# vmap each gaussian contributes to its own image's 12-float row block only, so
# per-view camera grads come out independent.
@wp.kernel
def _project_bwd_viewmat_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy_in: wp.array[wp.vec2],
    v_depth_in: wp.array[wp.float32],
    v_conic_in: wp.array[wp.vec3],
    num_gaussians: wp.int32,
    blocks_per_image: wp.int32,
    sel_means: wp.int32,
    sel_scales: wp.int32,
    sel_quats: wp.int32,
    sel_view: wp.int32,
    sel_radii: wp.int32,
    sel_conics: wp.int32,
    sel_vxy: wp.int32,
    sel_vdepth: wp.int32,
    sel_vconic: wp.int32,
    fx: wp.float32,
    fy: wp.float32,
    glob_scale: wp.float32,
    v_viewmat: wp.array2d[wp.float32],
):
    # launch_tiled over B*blocks_per_image blocks. Each block belongs to one image,
    # so the block-collective tile_sum never crosses an image boundary. Threads
    # outside N or culled contribute zero but must still take part in the tile_sum.
    blk, tr = wp.tid()
    n = num_gaussians
    image_id = blk // blocks_per_image
    local_block = blk % blocks_per_image
    gid = local_block * VIEW_BLOCK + tr
    idx = image_id * n + gid
    v_R = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_t = wp.vec3(0.0, 0.0, 0.0)
    r_idx = wp.where(sel_radii != 0, idx, gid)
    if gid < n and radii[r_idx] > 0:
        m_idx = wp.where(sel_means != 0, idx, gid)
        s_idx = wp.where(sel_scales != 0, idx, gid)
        q_idx = wp.where(sel_quats != 0, idx, gid)
        c_idx = wp.where(sel_conics != 0, idx, gid)
        vx_idx = wp.where(sel_vxy != 0, idx, gid)
        vd_idx = wp.where(sel_vdepth != 0, idx, gid)
        vc_idx = wp.where(sel_vconic != 0, idx, gid)
        vb = wp.where(sel_view != 0, image_id, 0) * 4
        mean = means3d[m_idx]
        W, trans = _load_W_trans(viewmat, vb)
        g = _recompute_geom(
            mean, quats[q_idx], scales[s_idx], W, trans, glob_scale, fx, fy
        )
        vcov2d = _vcov2d_from_conic(conics[c_idx], v_conic_in[vc_idx])
        v_p, v_T, _v_V = _proj_vjp(
            g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d
        )
        v_R, v_t = _view_grad(g, mean, v_p, v_T)
    ob = image_id * 4
    for i in range(3):
        for j in range(3):
            s = wp.tile_sum(wp.tile(v_R[i, j]))
            if tr == 0:
                wp.atomic_add(v_viewmat, ob + i, j, wp.tile_extract(s, 0))
        st = wp.tile_sum(wp.tile(v_t[i]))
        if tr == 0:
            wp.atomic_add(v_viewmat, ob + i, 3, wp.tile_extract(st, 0))


# Joint kernel, gaussians plus viewmat. Per-image block layout like the viewmat
# kernel so the tile_sum stays within one image. The gaussian grads compose the
# helpers in the same order as the gaussians-only kernel, so they are
# bit-identical to it.
@wp.kernel
def _project_bwd_joint_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy_in: wp.array[wp.vec2],
    v_depth_in: wp.array[wp.float32],
    v_conic_in: wp.array[wp.vec3],
    num_gaussians: wp.int32,
    blocks_per_image: wp.int32,
    sel_means: wp.int32,
    sel_scales: wp.int32,
    sel_quats: wp.int32,
    sel_view: wp.int32,
    sel_radii: wp.int32,
    sel_conics: wp.int32,
    sel_vxy: wp.int32,
    sel_vdepth: wp.int32,
    sel_vconic: wp.int32,
    fx: wp.float32,
    fy: wp.float32,
    glob_scale: wp.float32,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
    v_viewmat: wp.array2d[wp.float32],
):
    blk, tr = wp.tid()
    n = num_gaussians
    image_id = blk // blocks_per_image
    local_block = blk % blocks_per_image
    gid = local_block * VIEW_BLOCK + tr
    idx = image_id * n + gid
    v_R = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_t = wp.vec3(0.0, 0.0, 0.0)
    r_idx = wp.where(sel_radii != 0, idx, gid)
    if gid < n and radii[r_idx] > 0:
        m_idx = wp.where(sel_means != 0, idx, gid)
        s_idx = wp.where(sel_scales != 0, idx, gid)
        q_idx = wp.where(sel_quats != 0, idx, gid)
        c_idx = wp.where(sel_conics != 0, idx, gid)
        vx_idx = wp.where(sel_vxy != 0, idx, gid)
        vd_idx = wp.where(sel_vdepth != 0, idx, gid)
        vc_idx = wp.where(sel_vconic != 0, idx, gid)
        vb = wp.where(sel_view != 0, image_id, 0) * 4
        mean = means3d[m_idx]
        W, trans = _load_W_trans(viewmat, vb)
        g = _recompute_geom(
            mean, quats[q_idx], scales[s_idx], W, trans, glob_scale, fx, fy
        )
        vcov2d = _vcov2d_from_conic(conics[c_idx], v_conic_in[vc_idx])
        v_p, v_T, v_V = _proj_vjp(
            g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d
        )
        v_mean3d[idx] = wp.transpose(g.W) * v_p
        vs, vq = _scale_quat_vjp(g, quats[q_idx], v_V, glob_scale)
        v_scale[idx] = vs
        v_quat[idx] = vq
        v_R, v_t = _view_grad(g, mean, v_p, v_T)
    ob = image_id * 4
    for i in range(3):
        for j in range(3):
            s = wp.tile_sum(wp.tile(v_R[i, j]))
            if tr == 0:
                wp.atomic_add(v_viewmat, ob + i, j, wp.tile_extract(s, 0))
        st = wp.tile_sum(wp.tile(v_t[i]))
        if tr == 0:
            wp.atomic_add(v_viewmat, ob + i, 3, wp.tile_extract(st, 0))


_BWD_BLOCK = int(VIEW_BLOCK)


def _bwd_selectors(
    n: int,
    viewmat: "wp.array | wp.array2d[wp.float32]",
    means3d: wp.array,
    scales: wp.array,
    quats: wp.array,
    radii: wp.array,
    conics: wp.array,
    v_xy: wp.array,
    v_depth: wp.array,
    v_conic: wp.array,
) -> tuple[int, ...]:
    # Each operand is independently batched (own leading dim B*N or 4B, read at
    # the flat idx) or broadcast (N or 4, read at gid). One selector per operand,
    # indexed in-kernel exactly like the forward. This is required for
    # correctness, not just robustness. A cotangent can arrive broadcast even when
    # the geometry is fully batched, e.g. the depth cotangent of an image loss.
    def sel(a: "wp.array | wp.array2d[wp.float32]", base: int) -> int:
        # every operand is a real wp.array at runtime, array2d is a stub-only alias
        return 1 if cast(wp.array, a).shape[0] > base else 0

    return (
        sel(means3d, n),
        sel(scales, n),
        sel(quats, n),
        sel(viewmat, 4),
        sel(radii, n),
        sel(conics, n),
        sel(v_xy, n),
        sel(v_depth, n),
        sel(v_conic, n),
    )


def _project_bwd_gaussians_launch(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy: wp.array[wp.vec2],
    v_depth: wp.array[wp.float32],
    v_conic: wp.array[wp.vec3],
    num_gaussians: int,
    fx: float,
    fy: float,
    glob_scale: float,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
) -> None:
    n = num_gaussians
    B = v_mean3d.shape[0] // n
    sels = _bwd_selectors(
        n, viewmat, means3d, scales, quats, radii, conics, v_xy, v_depth, v_conic
    )
    v_mean3d.zero_()
    v_scale.zero_()
    v_quat.zero_()
    wp.launch(
        _project_bwd_gaussians_kernel,
        dim=B * n,
        inputs=[
            means3d,
            scales,
            quats,
            viewmat,
            radii,
            conics,
            v_xy,
            v_depth,
            v_conic,
            n,
            *sels,
            fx,
            fy,
            glob_scale,
        ],
        outputs=[v_mean3d, v_scale, v_quat],
        device=means3d.device,
    )


def _project_bwd_viewmat_launch(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy: wp.array[wp.vec2],
    v_depth: wp.array[wp.float32],
    v_conic: wp.array[wp.vec3],
    num_gaussians: int,
    fx: float,
    fy: float,
    glob_scale: float,
    v_viewmat: wp.array2d[wp.float32],
) -> None:
    n = num_gaussians
    B = v_viewmat.shape[0] // 4
    sels = _bwd_selectors(
        n, viewmat, means3d, scales, quats, radii, conics, v_xy, v_depth, v_conic
    )
    v_viewmat.zero_()
    blocks_per_image = (n + _BWD_BLOCK - 1) // _BWD_BLOCK
    wp.launch_tiled(
        _project_bwd_viewmat_kernel,
        dim=[B * blocks_per_image],
        inputs=[
            means3d,
            scales,
            quats,
            viewmat,
            radii,
            conics,
            v_xy,
            v_depth,
            v_conic,
            n,
            blocks_per_image,
            *sels,
            fx,
            fy,
            glob_scale,
        ],
        outputs=[v_viewmat],
        block_dim=_BWD_BLOCK,
        device=means3d.device,
    )


def _project_bwd_joint_launch(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy: wp.array[wp.vec2],
    v_depth: wp.array[wp.float32],
    v_conic: wp.array[wp.vec3],
    num_gaussians: int,
    fx: float,
    fy: float,
    glob_scale: float,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
    v_viewmat: wp.array2d[wp.float32],
) -> None:
    n = num_gaussians
    B = v_mean3d.shape[0] // n
    sels = _bwd_selectors(
        n, viewmat, means3d, scales, quats, radii, conics, v_xy, v_depth, v_conic
    )
    v_mean3d.zero_()
    v_scale.zero_()
    v_quat.zero_()
    v_viewmat.zero_()
    blocks_per_image = (n + _BWD_BLOCK - 1) // _BWD_BLOCK
    wp.launch_tiled(
        _project_bwd_joint_kernel,
        dim=[B * blocks_per_image],
        inputs=[
            means3d,
            scales,
            quats,
            viewmat,
            radii,
            conics,
            v_xy,
            v_depth,
            v_conic,
            n,
            blocks_per_image,
            *sels,
            fx,
            fy,
            glob_scale,
        ],
        outputs=[v_mean3d, v_scale, v_quat, v_viewmat],
        block_dim=_BWD_BLOCK,
        device=means3d.device,
    )


# Batch-native backward, exactly like the forward. Gaussian grads come out per
# view and JAX reduces broadcast inputs over the batch axis. The viewmat grad is
# a per-image accumulator.
_project_bwd_gaussians_ffi = jax_callable(
    _project_bwd_gaussians_launch,
    num_outputs=3,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)
_project_bwd_viewmat_ffi = jax_callable(
    _project_bwd_viewmat_launch,
    num_outputs=1,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)
_project_bwd_joint_ffi = jax_callable(
    _project_bwd_joint_launch,
    num_outputs=4,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)


_ProjOut = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


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
) -> _ProjOut:
    # gaussian_transforms and transform_ids enable the rigid transforms of the
    # grad-free inference path. The differentiable path never passes them, its
    # backward recomputes geometry from the untransformed residuals. Without
    # transforms the kernel takes dummy operands and skips the transform block.
    H, W = img_shape
    if gaussian_transforms is None:
        gaussian_transforms = jnp.zeros((1, 4, 4), jnp.float32)
        transform_ids = jnp.full((1,), -1, jnp.int32)
        num_transforms = 1
        has_transforms = 0
    else:
        assert transform_ids is not None  # built alongside the transforms
        num_transforms = gaussian_transforms.shape[-3]
        has_transforms = 1
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
        int(has_transforms),
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


def _project_fwd_rule(
    mean3ds: "jax.custom_derivatives.CustomVJPPrimal",
    scales: "jax.custom_derivatives.CustomVJPPrimal",
    quats: "jax.custom_derivatives.CustomVJPPrimal",
    viewmat: "jax.custom_derivatives.CustomVJPPrimal",
    opac: "jax.custom_derivatives.CustomVJPPrimal",
    n: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
) -> tuple[_ProjOut, tuple]:
    # Under symbolic_zeros the five differentiable args arrive as CustomVJPPrimal
    # with a value and a static perturbed bit. The perturbation pattern is encoded
    # as () or None in the residuals. Both are pytree structure, not leaves, so
    # they stay concrete under jit and the backward rule branches in Python.
    # opacities are never differentiated through project, they only drive integer
    # tile counts. The opacity gradient flows through rasterize.
    m, s, q = mean3ds.value, scales.value, quats.value
    vm, op = viewmat.value, opac.value
    pert_gaussians = (
        () if (mean3ds.perturbed or scales.perturbed or quats.perturbed) else None
    )
    pert_viewmat = () if viewmat.perturbed else None
    out = _project_call(m, s, q, vm, op, n, img_shape, f, c, glob_scale, clip_thresh)
    _xys, _depths, radii, conics, _nth, _cum = out
    return out, (m, s, q, vm, radii, conics, pert_gaussians, pert_viewmat)


def _materialize(ct: object) -> jax.Array:
    # symbolic_zeros hands the backward rule SymbolicZero objects for cotangents
    # not involved in differentiation, e.g. the depth channel under a plain image
    # loss. The Warp kernels need concrete arrays, so those become dense zeros of
    # the cotangent's own shape and dtype, correct under batching too. XLA folds
    # the zeros away.
    if isinstance(ct, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros(ct.aval.shape, ct.aval.dtype)
    return cast(jax.Array, ct)


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
            output_dims=n,
        )
    return (v_mean, v_scale, v_quat, v_viewmat, None)


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
) -> _ProjOut:
    return _project_call(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        n,
        img_shape,
        f,
        c,
        glob_scale,
        clip_thresh,
    )


_project_differentiable.defvjp(
    _project_fwd_rule, _project_bwd_rule, symbolic_zeros=True
)


def opacity_compensation(
    conics: jax.Array, radii: jax.Array, eps: float = 0.3
) -> jax.Array:
    """Mip-Splatting anti-aliased opacity compensation factor per gaussian.

    The factor is sqrt(det(cov2d) / det(cov2d + eps I)), the det ratio of the
    undilated 2D covariance over the eps-dilated one that projection already
    applies. Multiplying it into the opacity before the blend cancels the
    artificial area inflation the dilation grants thin gaussians (Mip-Splatting,
    Yu et al., CVPR 2024).

    Computed in closed form from the projection's conics output, so its gradient
    flows back to scales, quats, and means through project's existing vjp with no
    Warp kernel change. With (a, b, c) the conic the ratio is
    1 - eps (a + c) + eps^2 (a c - b^2), clipped to [0, 1] for float safety.
    Culled gaussians (radii <= 0) get factor 1.
    """
    c = conics.reshape(-1, 3)
    a = c[:, 0]
    b = c[:, 1]
    cc = c[:, 2]
    rho2 = 1.0 - eps * (a + cc) + (eps * eps) * (a * cc - b * b)
    rho = jnp.sqrt(jnp.clip(rho2, 0.0, 1.0))
    valid = radii.reshape(-1) > 0
    return jnp.where(valid, rho, 1.0)


def project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    opacities: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
) -> _ProjOut:
    """Project gaussians to screen space with the opacity-aware tile intersection.

    Returns (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit), the
    standard 3DGS projection outputs.

    Differentiable via jax.custom_vjp. Gradient selection is pure jax.grad and
    argnums. Perturbing the viewmat runs only the camera-pose accumulator,
    perturbing means, scales, or quats runs only the gaussian-grad kernel, and
    both together run the joint kernel. Without gradients the primal runs exactly
    as the forward-only path. opacities feed the integer tile counts and carry no
    gradient through project, their gradient flows through rasterize.
    """
    n = mean3ds.shape[0]
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
