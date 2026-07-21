"""Warp projection kernels and their JAX FFI callables.

The forward kernel projects each gaussian to screen space and counts the tiles its opacity-aware
ellipse touches. The backward kernel computes every gradient for the projection in a single pass,
skipping the transform read and reduction when no transforms are active. Each kernel is wrapped into
a JAX FFI callable that is used to define the custom_vjps in _project.py.
"""

from __future__ import annotations

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

    # Apply the optional rigid transform, project to camera space, and clip on the near plane. The
    # dummy transform_ids has length 1, so only index it when transforms are present.
    tf_id = wp.int32(-1)
    if has_transforms:
        tf_id = transform_ids[gid]
    R_tf, moved, mean = _apply_transform(
        gaussian_transforms, bid, num_transforms, sel_transforms, tf_id, mean
    )
    W, p_view, M = _project_geom(
        mean, quats[q_idx], scales[s_idx], glob_scale, viewmat, vb, R_tf, moved
    )
    if p_view[2] <= clip_thresh:
        return
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

# cov3d is recomputed in the backward instead of cached, saving memory on the order of 6N. The EWA
# Jacobian is rebuilt from the unclamped camera-space position, the standard gsplat approximation.


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
    # Selection analogous to the forward kernel to detect broadcast vs batched inputs
    sel_means = means3d.shape[0] > n
    sel_scales = scales.shape[0] > n
    sel_quats = quats.shape[0] > n
    sel_view = viewmat.shape[0] > 4
    sel_radii = radii.shape[0] > n
    sel_conics = conics.shape[0] > n
    sel_vxy = v_xy.shape[0] > n
    sel_vdepth = v_depth.shape[0] > n
    sel_vconic = v_conic.shape[0] > n
    sel_transforms = gaussian_transforms.shape[0] > num_transforms
    v_mean3d.zero_()
    v_scale.zero_()
    v_quat.zero_()
    v_viewmat.zero_()
    v_transforms.zero_()
    blocks_per_image = (n + VIEW_BLOCK - 1) // VIEW_BLOCK
    # Performance note: Every gaussian contributes to one shared v_viewmat per image. Each block
    # reduces its threads' contributions with wp.tile_sum and thread 0 issues one atomic per entry
    # per block. Projection has uniform per-gaussian work and no early termination, so the block
    # barrier is amortised and the reduction beats plain per-thread atomics.
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
            sel_means,
            sel_scales,
            sel_quats,
            sel_view,
            sel_radii,
            sel_conics,
            sel_vxy,
            sel_vdepth,
            sel_vconic,
            sel_transforms,
            fx,
            fy,
            glob_scale,
        ],
        outputs=[v_mean3d, v_scale, v_quat, v_viewmat, v_transforms],
        block_dim=VIEW_BLOCK,
        device=means3d.device,
    )


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
    if gid < n and radii[r_idx] > 0:  # Skip culled gaussians
        m_idx = wp.where(sel_means, idx, gid)
        s_idx = wp.where(sel_scales, idx, gid)
        q_idx = wp.where(sel_quats, idx, gid)
        c_idx = wp.where(sel_conics, idx, gid)
        vx_idx = wp.where(sel_vxy, idx, gid)
        vd_idx = wp.where(sel_vdepth, idx, gid)
        vc_idx = wp.where(sel_vconic, idx, gid)
        vb = wp.where(sel_view, image_id, 0) * 4
        mean_local = means3d[m_idx]
        # Recomputed geometry, identical to the forward. _apply_transform and _project_geom are the
        # shared forward/backward geometry. The EWA jacobian J is rebuilt here from the unclamped p
        # (the gsplat approximation), and R, s are recomputed for the scale/quat vjp (CSE'd with the
        # M build inside _project_geom).
        R_tf, moved, mean = _apply_transform(
            gaussian_transforms, image_id, num_transforms, sel_transforms, tf_id, mean_local
        )
        W, p, M = _project_geom(
            mean, quats[q_idx], scales[s_idx], glob_scale, viewmat, vb, R_tf, moved
        )
        V = M * wp.transpose(M)
        R = wp.quat_to_matrix(quats[q_idx])
        s = glob_scale * scales[s_idx]
        rz = 1.0 / p[2]
        rz2 = rz * rz
        J = wp.mat33(fx * rz, 0.0, -fx * p[0] * rz2, 0.0, fy * rz, -fy * p[1] * rz2, 0.0, 0.0, 0.0)
        T = J * W
        vcov2d = _vcov2d_from_conic(conics[c_idx], v_conic_in[vc_idx])
        v_p, v_T, v_V = _proj_vjp(p, T, W, V, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
        v_mean_world = wp.transpose(W) * v_p
        v_M_world = (v_V + wp.transpose(v_V)) * M
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
    # conic-to-cov2d vjp: v_cov2d = -X G X, X the conic matrix, G the cotangent (off-diagonal
    # halved, b sits in both symmetric slots). X G X is symmetric, repacked to (a, b, c) with the
    # two off-diagonals summed into b.
    X = wp.mat22(conic[0], conic[1], conic[1], conic[2])
    G = wp.mat22(v_conic[0], 0.5 * v_conic[1], 0.5 * v_conic[1], v_conic[2])
    S = X * G * X
    return wp.vec3(-S[0, 0], -(S[0, 1] + S[1, 0]), -S[1, 1])


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
    v_V = wp.transpose(T) * v_cov * T
    v_T = 2.0 * v_cov * T * V  # v_cov is symmetric, so v_cov T V + v_cov^T T V = 2 v_cov T V
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
    v_R = v_M * wp.diag(s)  # column-scale by s, the vjp of M = R diag(s) wrt R
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
def _load_affine(m: wp.array2d[wp.float32], r0: wp.int32) -> tuple[wp.mat33, wp.vec3]:
    # Rotation (upper-left 3x3) and translation (column 3) of the affine block in rows [r0, r0+3) of
    # a row-major (rows, 4) array. Shared by the viewmat (row block at vb) and the rigid transforms
    # (a (4, 4) block sliced in-kernel from the (K, 4, 4) stack, read at r0 = 0).
    R = wp.mat33(
        m[r0 + 0, 0],
        m[r0 + 0, 1],
        m[r0 + 0, 2],
        m[r0 + 1, 0],
        m[r0 + 1, 1],
        m[r0 + 1, 2],
        m[r0 + 2, 0],
        m[r0 + 2, 1],
        m[r0 + 2, 2],
    )
    t = wp.vec3(m[r0 + 0, 3], m[r0 + 1, 3], m[r0 + 2, 3])
    return R, t


@wp.func
def _apply_transform(
    transforms: wp.array3d[wp.float32],
    image_id: wp.int32,
    num_transforms: wp.int32,
    sel_transforms: wp.bool,
    tf_id: wp.int32,
    mean: wp.vec3,
) -> tuple[wp.mat33, wp.bool, wp.vec3]:
    # Apply the optional rigid transform tf_id to the mean in world space (tf_id < 0 leaves it
    # static). Returns the transform rotation (identity if static), whether it moved, and the
    # possibly-moved mean. The covariance factor follows R_tf downstream via the moved flag.
    R_tf = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    moved = wp.bool(False)
    if tf_id >= 0:
        tf_idx = wp.where(sel_transforms, image_id * num_transforms + tf_id, tf_id)
        R_tf, t_tf = _load_affine(transforms[tf_idx], 0)
        mean = R_tf * mean + t_tf
        moved = wp.bool(True)
    return R_tf, moved, mean


@wp.func
def _project_geom(
    mean: wp.vec3,
    quat: wp.quat,
    scale: wp.vec3,
    glob_scale: wp.float32,
    viewmat: wp.array2d[wp.float32],
    vb: wp.int32,
    R_tf: wp.mat33,
    moved: wp.bool,
) -> tuple[wp.mat33, wp.vec3, wp.mat33]:
    # Geometry shared by the forward and backward: the camera rotation W and the unclamped
    # camera-space position p from the viewmat, and the covariance factor M = R diag(glob_scale
    # scale), rotated by an active transform. Returns (W, p, M); the caller forms V = M M^T.
    W, trans = _load_affine(viewmat, vb)
    p = W * mean + trans
    M = wp.quat_to_matrix(quat) * wp.diag(glob_scale * scale)
    if moved:
        M = R_tf * M
    return W, p, M


@wp.func
def _view_grad(J: wp.mat33, mean: wp.vec3, v_p: wp.vec3, v_T: wp.mat33) -> tuple[wp.mat33, wp.vec3]:
    # v_R = outer(v_p, mean) + J^T v_T and v_trans = v_p, see header derivation
    v_R = wp.outer(v_p, mean) + wp.transpose(J) * v_T
    return v_R, v_p
