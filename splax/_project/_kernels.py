"""Warp projection kernels and their JAX FFI callables.

The forward kernel projects each gaussian to screen space and counts the tiles its opacity-aware
ellipse touches. The backward kernel computes every gradient for the projection in a single pass,
skipping the transform read and reduction when no transforms are active. Each kernel is wrapped into
a JAX FFI callable that is used to define the custom_vjps in _project.py.
"""

from __future__ import annotations

from typing import cast

import warp as wp
from warp import JaxCallableGraphMode, jax_callable

from splax._batching import nested_vmap
from splax._intersect import (
    ALPHA_THRESHOLD,
    BLOCK_WIDTH,
    GAUSSIAN_EXTEND_SQ,
    ellipse_setup,
    ellipse_tile_count,
)

wp.set_module_options({"fast_math": True})  # fastmath significantly accelerates the kernels

VIEW_BLOCK = wp.constant(256)  # threads per block for the tile_sum viewmat reduce


# region forward kernels


def _project_warp(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: int,
    num_transforms: int,
    has_transforms: bool,
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
    """Launch the kernel behind the ffi, deriving the batch selectors from runtime shapes.

    Native batch handling in jax requires jax.vmap with vmap_method="expand_dims". Warp's FFI
    callback, however, collapses batches into the leading array dimension. We detect this and
    configure the kernel to account for it, enabling one single kernel launch over all batches.
    """
    # N is passed statically because jax.vmap hides the batch axis from this wrapper. B is recovered
    # from an output shape. Inputs arrive flattened, so batched inputs have shape  (B1 * ... * N).
    # We thus check if the input is batched by comparing its leading dim to N. If batched, the flat
    # idx is used to read the input, otherwise the gaussian idx is used.
    n = num_gaussians
    total = xys.shape[0]  # B1 * ... * N
    sel_means = means3d.shape[0] > n
    sel_scales = scales.shape[0] > n
    sel_quats = quats.shape[0] > n
    sel_view = viewmat.shape[0] > 4
    sel_opac = opacities.shape[0] > n
    sel_transforms = gaussian_transforms.shape[0] > num_transforms
    inputs = [
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
    ]
    outputs = [xys, depths, radii, conics, num_tiles_hit]
    wp.launch(_project_kernel, dim=total, inputs=inputs, outputs=outputs)
    # One global inclusive prefix sum over the flattened tile counts, so all images' intersections
    # are laid out contiguously for a single global sort. array_scan requires host-side temp
    # management, so cannot be captured in the JAX graph.
    wp.utils.array_scan(num_tiles_hit, cum_tiles_hit, inclusive=True)


_project_ffi = nested_vmap(
    jax_callable(
        _project_warp,
        num_outputs=6,
        graph_mode=JaxCallableGraphMode.WARP,
        vmap_method="expand_dims",  # native batch handling in one single kernel launch
    ),
    n_arrays=7,
    name="project",
)


@wp.kernel
def _project_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: wp.int32,
    num_transforms: wp.int32,
    has_transforms: wp.bool,
    sel_means: wp.bool,
    sel_scales: wp.bool,
    sel_quats: wp.bool,
    sel_view: wp.bool,
    sel_opac: wp.bool,
    sel_transforms: wp.bool,
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
    idx = wp.tid()
    n = num_gaussians
    bid = idx // n  # batch id used to select the viewmat and transform slice
    gid = idx % n  # gaussian id. Used as index when the input is broadcast
    m_idx = wp.where(sel_means, idx, gid)
    s_idx = wp.where(sel_scales, idx, gid)
    q_idx = wp.where(sel_quats, idx, gid)
    o_idx = wp.where(sel_opac, idx, gid)
    vb = wp.where(sel_view, bid, 0) * 4  # row offset into (4B, 4) viewmat

    xys[idx] = wp.vec2(0.0, 0.0)
    depths[idx] = 0.0
    radii[idx] = 0
    conics[idx] = wp.vec3(0.0, 0.0, 0.0)
    num_tiles_hit[idx] = 0

    mean = means3d[m_idx]

    # Optional rigid transforms for the splats that moves them in world space
    R_tf = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    moved = wp.bool(False)
    if has_transforms:  # Transforms are optional, skip if not present
        tf_id = transform_ids[gid]
        if tf_id >= 0:  # -1 means skipping the transform for this particular gaussian
            # Transforms are either broadcast (K, 4, 4) or batched (B*K, 4, 4). If batched, we need
            # to load the transform for the correct gaussian and batch element
            tf_idx = wp.where(sel_transforms, bid * num_transforms + tf_id, tf_id)
            R_tf, t_tf = _load_transform(gaussian_transforms, tf_idx)
            mean = R_tf * mean + t_tf
            moved = wp.bool(True)

    # camera-space position p_view = viewmat @ mean (upper 3x4 block), then near-plane clip
    W, trans = _load_W_trans(viewmat, vb)
    p_view = W * mean + trans
    if p_view[2] <= clip_thresh:
        return

    # world covariance V3 = M M^T with M = R diag(glob_scale * scale)
    M = wp.quat_to_matrix(quats[q_idx]) * wp.diag(glob_scale * scales[s_idx])
    if moved:  # If an external transform was applied, account for the additional rotation
        M = R_tf * M
    V3 = M * wp.transpose(M)

    # EWA projection of the covariance
    tan_fovx = 0.5 * wp.float32(img_w) / fx
    tan_fovy = 0.5 * wp.float32(img_h) / fy
    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    tx = p_view[0]
    ty = p_view[1]
    tz = p_view[2]
    tx = tz * wp.min(lim_x, wp.max(-lim_x, tx / tz))
    ty = tz * wp.min(lim_y, wp.max(-lim_y, ty / tz))
    rz = 1.0 / tz
    rz2 = rz * rz
    J = wp.mat33(fx * rz, 0.0, -fx * tx * rz2, 0.0, fy * rz, -fy * ty * rz2, 0.0, 0.0, 0.0)
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
    rw = 1.0 / (p_view[2] + 1e-6)
    center_x = (p_view[0] * rw) * fx + cx
    center_y = (p_view[1] * rw) * fy + cy

    tb_x = (img_w + BLOCK_WIDTH - 1) / BLOCK_WIDTH
    tb_y = (img_h + BLOCK_WIDTH - 1) / BLOCK_WIDTH

    # Opacity-aware tight tile intersection. Uses the _intersect helpers shared with the rasterizer,
    # so the counted and emitted tile totals match exactly.
    opac = opacities[o_idx]
    if opac < ALPHA_THRESHOLD:
        return  # alpha < 1/255 everywhere, contributes nothing
    t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
    ext = wp.sqrt(t)
    radius_x = wp.ceil(ext * wp.sqrt(cxx))
    radius_y = wp.ceil(ext * wp.sqrt(cyy))
    if radius_x <= 0.0 and radius_y <= 0.0:
        return  # Extend is less than one pixel in both axes
    if (
        center_x + radius_x <= 0.0
        or center_x - radius_x >= wp.float32(img_w)
        or center_y + radius_y <= 0.0
        or center_y - radius_y >= wp.float32(img_h)
    ):
        return  # Gaussian is fully offscreen, no tiles hit
    setup = ellipse_setup(conic[0], conic[1], conic[2], t, center_x, center_y, tb_x, tb_y)
    count = ellipse_tile_count(setup)
    if count <= 0:
        return
    num_tiles_hit[idx] = count
    depths[idx] = p_view[2]
    radii[idx] = wp.int32(wp.max(radius_x, radius_y))
    xys[idx] = wp.vec2(center_x, center_y)
    conics[idx] = conic


# region backward kernels

# All vjp math lives in shared wp.func helpers so the three kernel variants
# (gaussians only, viewmat only, joint) carry no duplicated math. Each variant
# composes the helpers it needs and writes the grads it owns.
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
# case, see _rasterize.


def _project_bwd_warp(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy: wp.array[wp.vec2],
    v_depth: wp.array[wp.float32],
    v_conic: wp.array[wp.vec3],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: int,
    num_transforms: int,
    has_transforms: bool,
    fx: float,
    fy: float,
    glob_scale: float,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
    v_viewmat: wp.array2d[wp.float32],
    v_transforms: wp.array3d[wp.float32],
) -> None:
    """Launch the single projection backward, deriving batch selectors from runtime shapes."""
    n = num_gaussians
    B = v_mean3d.shape[0] // n
    sels = _bwd_selectors(n, viewmat, means3d, scales, quats, radii, conics, v_xy, v_depth, v_conic)
    sel_transforms = gaussian_transforms.shape[0] > num_transforms
    v_mean3d.zero_()
    v_scale.zero_()
    v_quat.zero_()
    v_viewmat.zero_()
    v_transforms.zero_()
    blocks_per_image = (n + VIEW_BLOCK - 1) // VIEW_BLOCK
    wp.launch_tiled(
        _project_bwd_kernel,
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
            gaussian_transforms,
            transform_ids,
            n,
            num_transforms,
            has_transforms,
            blocks_per_image,
            *sels,
            sel_transforms,
            fx,
            fy,
            glob_scale,
        ],
        outputs=[v_mean3d, v_scale, v_quat, v_viewmat, v_transforms],
        block_dim=VIEW_BLOCK,
        device=means3d.device,
    )


# Batch-native backward, exactly like the forward. Gaussian grads come out per
# view and JAX reduces broadcast inputs over the batch axis. The viewmat and
# transform grads are per-image accumulators.
_project_bwd_ffi = nested_vmap(
    jax_callable(
        _project_bwd_warp,
        num_outputs=5,
        graph_mode=JaxCallableGraphMode.WARP,
        vmap_method="expand_dims",
    ),
    n_arrays=11,
    name="project_bwd",
)


def _bwd_selectors(
    n: int,
    viewmat: wp.array | wp.array2d[wp.float32],
    means3d: wp.array,
    scales: wp.array,
    quats: wp.array,
    radii: wp.array,
    conics: wp.array,
    v_xy: wp.array,
    v_depth: wp.array,
    v_conic: wp.array,
) -> tuple[bool, ...]:
    # Each operand is independently batched (own leading dim B*N or 4B, read at
    # the flat idx) or broadcast (N or 4, read at gid). One selector per operand,
    # indexed in-kernel exactly like the forward. This is required for
    # correctness, not just robustness. A cotangent can arrive broadcast even when
    # the geometry is fully batched, e.g. the depth cotangent of an image loss.
    def sel(a: wp.array | wp.array2d[wp.float32], base: int) -> bool:
        # every operand is a real wp.array at runtime, array2d is a stub-only alias
        return cast("wp.array", a).shape[0] > base

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


# The projection backward, computing every gradient set in one pass so gradient
# selection is jax.grad's job, not the kernel's. With has_transforms false it
# skips the transform read, apply, and reduction and reduces to the plain gaussian
# plus viewmat grads. The recompute must apply the same transforms as the forward,
# or every gradient is taken at the wrong geometry.
#
# With mean' = R_tf mean + t_tf and M' = R_tf M the chain rules are
#     v_mean = R_tf^T v_mean'          v_M = R_tf^T v_M'
#     v_R_tf = outer(v_mean', mean) + v_M' M^T
#     v_t_tf = v_mean'
# where v_mean' = W^T v_p is the gradient at the transformed world mean and
# v_M' = 2 sym(v_V) M' the one at the rotated covariance factor. Transform grads
# reduce like the viewmat: slices are contiguous, so almost every block sees one
# transform id and folds its 12 contributions with one tile_sum plus one atomic
# per entry. Mixed blocks at slice boundaries and the final partial block fall
# back to per-thread atomics. Plain per-thread atomics were measured to double
# the backward at 50% coverage with a single transform (one contended
# accumulator), the block reduction removes that.
@wp.kernel
def _project_bwd_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.quat],
    viewmat: wp.array2d[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    v_xy_in: wp.array[wp.vec2],
    v_depth_in: wp.array[wp.float32],
    v_conic_in: wp.array[wp.vec3],
    gaussian_transforms: wp.array3d[wp.float32],
    transform_ids: wp.array[wp.int32],
    num_gaussians: wp.int32,
    num_transforms: wp.int32,
    has_transforms: wp.bool,
    blocks_per_image: wp.int32,
    sel_means: wp.bool,
    sel_scales: wp.bool,
    sel_quats: wp.bool,
    sel_view: wp.bool,
    sel_radii: wp.bool,
    sel_conics: wp.bool,
    sel_vxy: wp.bool,
    sel_vdepth: wp.bool,
    sel_vconic: wp.bool,
    sel_transforms: wp.bool,
    fx: wp.float32,
    fy: wp.float32,
    glob_scale: wp.float32,
    v_mean3d: wp.array[wp.vec3],
    v_scale: wp.array[wp.vec3],
    v_quat: wp.array[wp.vec4],
    v_viewmat: wp.array2d[wp.float32],
    v_transforms: wp.array3d[wp.float32],
):
    blk, tr = wp.tid()
    n = num_gaussians
    image_id = blk // blocks_per_image
    local_block = blk % blocks_per_image
    gid = local_block * VIEW_BLOCK + tr
    idx = image_id * n + gid
    v_R = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_t = wp.vec3(0.0, 0.0, 0.0)
    v_R_tf = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_t_tf = wp.vec3(0.0, 0.0, 0.0)
    moved = wp.bool(False)
    tf_id = wp.int32(-1)
    if gid < n and has_transforms:
        tf_id = transform_ids[gid]  # read even when culled, for the uniformity vote
    r_idx = wp.where(sel_radii, idx, gid)
    if gid < n and radii[r_idx] > 0:
        m_idx = wp.where(sel_means, idx, gid)
        s_idx = wp.where(sel_scales, idx, gid)
        q_idx = wp.where(sel_quats, idx, gid)
        c_idx = wp.where(sel_conics, idx, gid)
        vx_idx = wp.where(sel_vxy, idx, gid)
        vd_idx = wp.where(sel_vdepth, idx, gid)
        vc_idx = wp.where(sel_vconic, idx, gid)
        vb = wp.where(sel_view, image_id, 0) * 4
        mean_local = means3d[m_idx]
        mean = mean_local
        R_tf = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if tf_id >= 0:
            tf_idx = wp.where(sel_transforms, image_id * num_transforms + tf_id, tf_id)
            R_tf, t_tf = _load_transform(gaussian_transforms, tf_idx)
            mean = R_tf * mean + t_tf
            moved = wp.bool(True)
        W, trans = _load_W_trans(viewmat, vb)
        # Recomputed geometry, deterministic and bit-identical to the forward. p is the unclamped
        # camera-space position, s the glob_scale-applied scale, M = R diag(s) the covariance factor
        # (rotated by any active transform), V = M M^T the world covariance, J the EWA jacobian
        # rebuilt from the unclamped p, and T = J W.
        p = W * mean + trans
        R = wp.quat_to_matrix(quats[q_idx])
        s = glob_scale * scales[s_idx]
        M = R * wp.diag(s)
        if moved:
            M = R_tf * M
        V = M * wp.transpose(M)
        rz = 1.0 / p[2]
        rz2 = rz * rz
        J = wp.mat33(fx * rz, 0.0, -fx * p[0] * rz2, 0.0, fy * rz, -fy * p[1] * rz2, 0.0, 0.0, 0.0)
        T = J * W
        vcov2d = _vcov2d_from_conic(conics[c_idx], v_conic_in[vc_idx])
        v_p, v_T, v_V = _proj_vjp(p, T, W, V, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
        v_mean_world = wp.transpose(W) * v_p
        v_M_world = _v_M_from_vV(v_V, M)
        v_mean_out = v_mean_world
        v_M = v_M_world
        if moved:
            v_mean_out = wp.transpose(R_tf) * v_mean_world
            v_M = wp.transpose(R_tf) * v_M_world
        v_mean3d[idx] = v_mean_out
        vs, vq = _scale_quat_vjp(R, s, quats[q_idx], v_M, glob_scale)
        v_scale[idx] = vs
        v_quat[idx] = vq
        v_R, v_t = _view_grad(J, mean, v_p, v_T)
        if moved:
            M_local = wp.transpose(R_tf) * M
            v_R_tf = wp.outer(v_mean_world, mean_local) + v_M_world * wp.transpose(M_local)
            v_t_tf = v_mean_world
    ob = image_id * 4
    # viewmat grad reduces in every block, the transform grad only when transforms are active. Both
    # branches on has_transforms are uniform across the block, so the tile ops stay collective.
    uniform = wp.bool(False)
    mask = wp.float32(0.0)
    ob_tf = wp.int32(0)
    tmin = wp.int32(-1)
    tmax = wp.int32(-1)
    if has_transforms:
        tmin = wp.tile_extract(wp.tile_min(wp.tile(tf_id)), 0)
        tmax = wp.tile_extract(wp.tile_max(wp.tile(tf_id)), 0)
        uniform = tmin == tmax and tmin >= 0
        mask = wp.where(uniform, 1.0, 0.0)  # masked so the tile ops run unconditionally
        ob_tf = image_id * num_transforms + wp.max(tmin, 0)
    for i in range(3):
        for j in range(3):
            sr = wp.tile_sum(wp.tile(v_R[i, j]))
            if tr == 0:
                wp.atomic_add(v_viewmat, ob + i, j, wp.tile_extract(sr, 0))
            if has_transforms:
                stf = wp.tile_sum(wp.tile(mask * v_R_tf[i, j]))
                if tr == 0 and uniform:
                    wp.atomic_add(v_transforms, ob_tf, i, j, wp.tile_extract(stf, 0))
        st = wp.tile_sum(wp.tile(v_t[i]))
        if tr == 0:
            wp.atomic_add(v_viewmat, ob + i, 3, wp.tile_extract(st, 0))
        if has_transforms:
            sttf = wp.tile_sum(wp.tile(mask * v_t_tf[i]))
            if tr == 0 and uniform:
                wp.atomic_add(v_transforms, ob_tf, i, 3, wp.tile_extract(sttf, 0))
    # mixed block at a slice boundary, per-thread fallback
    if has_transforms and moved and tmin != tmax:
        out_tf = image_id * num_transforms + tf_id
        for i in range(3):
            for j in range(3):
                wp.atomic_add(v_transforms, out_tf, i, j, v_R_tf[i, j])
            wp.atomic_add(v_transforms, out_tf, i, 3, v_t_tf[i])


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
    p: wp.vec3,
    T: wp.mat33,
    W: wp.mat33,
    V: wp.mat33,
    fx: wp.float32,
    fy: wp.float32,
    v_xy: wp.vec2,
    v_depth: wp.float32,
    vcov2d: wp.vec3,
) -> tuple[wp.vec3, wp.mat33, wp.mat33]:
    # Returns (v_p, v_T, v_V). v_p is the gradient wrt the camera-space position
    # (pixel + depth + EWA terms), used by the world-mean grad and the viewmat
    # translation grad. v_T is the gradient wrt T = J W, used by the viewmat
    # rotation grad. v_V is the gradient wrt the world covariance, used by
    # scale and quat. p is the unclamped camera-space position.
    tx = p[0]
    ty = p[1]
    tz = p[2]
    rz = 1.0 / tz
    rz2 = rz * rz
    rz3 = rz2 * rz
    rw = 1.0 / (tz + 1e-6)
    vpx = fx * v_xy[0]
    vpy = fy * v_xy[1]
    vvx = vpx * rw
    vvy = vpy * rw
    vvz = -(vpx * tx + vpy * ty) * rw * rw
    # the depth cotangent adds onto the z component of the position grad
    vvz = vvz + v_depth
    v_cov = wp.mat33(
        vcov2d[0], 0.5 * vcov2d[1], 0.0, 0.5 * vcov2d[1], vcov2d[2], 0.0, 0.0, 0.0, 0.0
    )
    Tt = wp.transpose(T)
    v_V = Tt * v_cov * T
    v_T = v_cov * T * V + wp.transpose(v_cov) * T * V
    v_J = v_T * wp.transpose(W)
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
def _v_M_from_vV(v_V: wp.mat33, M: wp.mat33) -> wp.mat33:
    # Symmetrized covariance cotangent, then v_M = 2 sym(v_V) M for V = M M^T
    vc0 = v_V[0, 0]
    vc1 = v_V[0, 1] + v_V[1, 0]
    vc2 = v_V[0, 2] + v_V[2, 0]
    vc3 = v_V[1, 1]
    vc4 = v_V[1, 2] + v_V[2, 1]
    vc5 = v_V[2, 2]
    v_Vc = wp.mat33(vc0, 0.5 * vc1, 0.5 * vc2, 0.5 * vc1, vc3, 0.5 * vc4, 0.5 * vc2, 0.5 * vc4, vc5)
    return (v_Vc * M) * 2.0


@wp.func
def _scale_quat_vjp(
    R: wp.mat33, s: wp.vec3, quat: wp.quat, v_M: wp.mat33, glob_scale: wp.float32
) -> tuple[wp.vec3, wp.vec4]:
    # Returns (v_scale, v_quat). v_M is the cotangent of the local covariance factor M = R diag(s)
    # with s = glob_scale * scale, already rotated back into the local frame when a rigid transform
    # moved the gaussian. v_quat is the analytic adjoint of the unit-quaternion rotation map,
    # computed directly here rather than through wp.quat_to_matrix's adjoint so it keeps the radial
    # component. R is the quaternion rotation.
    v_scale = wp.vec3(
        (R[0, 0] * v_M[0, 0] + R[1, 0] * v_M[1, 0] + R[2, 0] * v_M[2, 0]) * glob_scale,
        (R[0, 1] * v_M[0, 1] + R[1, 1] * v_M[1, 1] + R[2, 1] * v_M[2, 1]) * glob_scale,
        (R[0, 2] * v_M[0, 2] + R[1, 2] * v_M[1, 2] + R[2, 2] * v_M[2, 2]) * glob_scale,
    )
    v_R = wp.mat33(
        v_M[0, 0] * s[0],
        v_M[0, 1] * s[1],
        v_M[0, 2] * s[2],
        v_M[1, 0] * s[0],
        v_M[1, 1] * s[1],
        v_M[1, 2] * s[2],
        v_M[2, 0] * s[0],
        v_M[2, 1] * s[1],
        v_M[2, 2] * s[2],
    )
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    vq_w = 2.0 * (
        x * (v_R[2, 1] - v_R[1, 2]) + y * (v_R[0, 2] - v_R[2, 0]) + z * (v_R[1, 0] - v_R[0, 1])
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
    return v_scale, wp.vec4(vq_x, vq_y, vq_z, vq_w)


@wp.func
def _load_transform(transforms: wp.array3d[wp.float32], idx: wp.int32) -> tuple[wp.mat33, wp.vec3]:
    # Rotation and translation of the (idx, 4, 4) rigid transform, shared by the
    # forward and the transform-aware backward
    R = wp.mat33(
        transforms[idx, 0, 0],
        transforms[idx, 0, 1],
        transforms[idx, 0, 2],
        transforms[idx, 1, 0],
        transforms[idx, 1, 1],
        transforms[idx, 1, 2],
        transforms[idx, 2, 0],
        transforms[idx, 2, 1],
        transforms[idx, 2, 2],
    )
    t = wp.vec3(transforms[idx, 0, 3], transforms[idx, 1, 3], transforms[idx, 2, 3])
    return R, t


@wp.func
def _load_W_trans(viewmat: wp.array2d[wp.float32], vb: wp.int32) -> tuple[wp.mat33, wp.vec3]:
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
def _view_grad(J: wp.mat33, mean: wp.vec3, v_p: wp.vec3, v_T: wp.mat33) -> tuple[wp.mat33, wp.vec3]:
    # v_R = outer(v_p, mean) + J^T v_T and v_trans = v_p, see header derivation
    v_R = wp.outer(v_p, mean) + wp.transpose(J) * v_T
    return v_R, v_p
