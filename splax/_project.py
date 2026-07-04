"""Warp port of the reference CUDA projection stage (forward + backward).

The forward ports the reference CUDA ``project_gaussians_fwd`` kernel (and its
device helpers) faithfully into a single Warp kernel, wrapped as a JAX FFI call
via warp.jax_experimental.ffi.jax_callable. The callable launches the projection
kernel and then an inclusive prefix scan for cum_tiles_hit, all on the
XLA-provided CUDA stream. The backward section provides ``jax.custom_vjp`` rules
with three kernel variants selected by ``diff_wrt`` (gaussians / viewmat / both).

The public ``project`` returns (xys, depths, radii, conics, num_tiles_hit,
cum_tiles_hit): the standard 3DGS projection outputs.
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental.ffi import jax_callable, JaxCallableGraphMode

wp.init()


@wp.func
def _quat_to_rotmat(q: wp.vec4) -> wp.mat33:
    # quats are stored as (w, x, y, z) -> q[0]=w, q[1]=x, q[2]=y, q[3]=z
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    # standard row-major rotation matrix (matches glm column-major helper)
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


# --- SNUGBOX + AccuTile opacity-aware tile intersection (survey O6) -----------
# Faithful port of gsplat's SpeedySplat path (ProjectionEWA3DGSFused.cu:186-224,
# IntersectTile.cu:38-260). Instead of one isotropic 3-sigma radius and its
# axis-aligned tile bbox, low-opacity/anisotropic gaussians get a tight ellipse:
#   * opacity-aware isocontour level  t = min(EXTEND^2, 2*ln(opacity/ALPHA))
#     (a gaussian's alpha drops below 1/255 outside this level, so tiles past it
#     are invisible); per-axis radii radius_{x,y} = ceil(sqrt(t)*sqrt(cov_{xx,yy})).
#   * AccuTile walk: instead of emitting every tile in the padded bbox, walk the
#     ellipse column by column and emit only the tiles its boundary spans.
# Both the projection (tile COUNT -> num_tiles_hit) and the rasterize key-emission
# kernel run the *same* setup + column walk (shared _accutile_setup / _accutile_col)
# so the counted and emitted tile totals agree bit-for-bit -- required, or the
# per-gaussian sort-buffer offsets (cum_tiles_hit) corrupt.
GAUSSIAN_EXTEND_SQ = wp.constant(3.33 * 3.33)  # (max ellipse extent in sigma)^2
ALPHA_THRESHOLD = wp.constant(1.0 / 255.0)


@wp.struct
class AccuSetup:
    valid: wp.int32  # 1 if the ellipse's tile rectangle is non-empty
    A: wp.float32  # conic (inverse 2d covariance upper triangle)
    B: wp.float32
    C: wp.float32
    disc: wp.float32  # B*B - A*C  (= -det(conic) < 0 for a real ellipse)
    t: wp.float32  # opacity-aware isocontour level
    px: wp.float32  # ellipse center (image px), UN-swapped
    py: wp.float32
    # bbox_* / rect_* are stored with the walk's outer axis in component [0]:
    # for an x-major walk (isY=0) that's (x, y); for a y-major walk they are swapped.
    bbox_min: wp.vec2
    bbox_max: wp.vec2
    bbox_argmin: wp.vec2
    bbox_argmax: wp.vec2
    rect_min: wp.vec2i
    rect_max: wp.vec2i
    isY: wp.int32  # 1 if the walk marches over the y tiles (shorter span outer)


@wp.func
def _ellipse_intersection(
    A: wp.float32,
    B: wp.float32,
    C: wp.float32,
    disc: wp.float32,
    t: wp.float32,
    px: wp.float32,
    py: wp.float32,
    isY: wp.int32,
    coord: wp.float32,
) -> wp.vec2:
    # gsplat accutile_ellipse_intersection: where the boundary line u=coord meets
    # the ellipse, giving the [lower, upper] extent of the cross axis at that line.
    if isY != 0:
        p_u = py
        p_v = px
        coeff = A
    else:
        p_u = px
        p_v = py
        coeff = C
    h = coord - p_u
    sqrt_term = wp.sqrt(disc * h * h + t * coeff)
    return wp.vec2(
        (-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v
    )


@wp.func
def _accutile_setup(
    A: wp.float32,
    B: wp.float32,
    C: wp.float32,
    t: wp.float32,
    px: wp.float32,
    py: wp.float32,
    tile_size: wp.int32,
    tile_width: wp.int32,
    tile_height: wp.int32,
) -> AccuSetup:
    # SNUGBOX tight AABB of the ellipse + tile rectangle, then pick the shorter tile
    # span as the walk's outer axis (isY). Faithful to IntersectTile.cu:210-252.
    s = AccuSetup()
    s.valid = wp.int32(0)
    s.A = A
    s.B = B
    s.C = C
    s.t = t
    s.px = px
    s.py = py
    disc = B * B - A * C
    s.disc = disc
    neg_t_over_disc = -t / disc
    x_extent = wp.sqrt(neg_t_over_disc * C)
    y_extent = wp.sqrt(neg_t_over_disc * A)
    bbox_min = wp.vec2(px - x_extent, py - y_extent)
    bbox_max = wp.vec2(px + x_extent, py + y_extent)
    Bx_over_C = B * x_extent / C
    By_over_A = B * y_extent / A
    bbox_argmin = wp.vec2(py + Bx_over_C, px + By_over_A)
    bbox_argmax = wp.vec2(py - Bx_over_C, px - By_over_A)
    ts = wp.float32(tile_size)
    rminx = wp.max(0, wp.min(tile_width, wp.int32(bbox_min[0] / ts)))
    rminy = wp.max(0, wp.min(tile_height, wp.int32(bbox_min[1] / ts)))
    rmaxx = wp.max(0, wp.min(tile_width, wp.int32(bbox_max[0] / ts + 1.0)))
    rmaxy = wp.max(0, wp.min(tile_height, wp.int32(bbox_max[1] / ts + 1.0)))
    x_span = rmaxx - rminx
    y_span = rmaxy - rminy
    if y_span * x_span == 0:
        return s
    isY = wp.int32(0)
    if y_span < x_span:
        isY = wp.int32(1)
    s.isY = isY
    if isY != 0:
        s.rect_min = wp.vec2i(rminy, rminx)
        s.rect_max = wp.vec2i(rmaxy, rmaxx)
        s.bbox_min = wp.vec2(bbox_min[1], bbox_min[0])
        s.bbox_max = wp.vec2(bbox_max[1], bbox_max[0])
        s.bbox_argmin = wp.vec2(bbox_argmin[1], bbox_argmin[0])
        s.bbox_argmax = wp.vec2(bbox_argmax[1], bbox_argmax[0])
    else:
        s.rect_min = wp.vec2i(rminx, rminy)
        s.rect_max = wp.vec2i(rmaxx, rmaxy)
        s.bbox_min = bbox_min
        s.bbox_max = bbox_max
        s.bbox_argmin = bbox_argmin
        s.bbox_argmax = bbox_argmax
    s.valid = wp.int32(1)
    return s


@wp.func
def _accutile_first_imin(s: AccuSetup, block: wp.float32) -> wp.vec2:
    # Cross-axis extent carried into the first column (gsplat's intersect_min_line
    # init): the boundary at the rect's leading line, or a degenerate default when
    # that line lies outside the ellipse bbox.
    min_line0 = wp.float32(s.rect_min[0]) * block
    if s.bbox_min[0] <= min_line0:
        return _ellipse_intersection(
            s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, min_line0
        )
    return wp.vec2(s.bbox_max[1], s.bbox_min[1])


@wp.func
def _accutile_col(
    u: wp.int32, s: AccuSetup, block: wp.float32, I_min: wp.vec2
) -> wp.vec4:
    # One outer column of the AccuTile walk: returns (min_tile_v, max_tile_v, I_max)
    # where I_max feeds the next column as its I_min (gsplat's rolling
    # intersect_min_line = intersect_max_line). Cross-axis tile range [min_v, max_v).
    min_line = wp.float32(u) * block
    max_line = min_line + block
    if max_line <= s.bbox_max[0]:
        I_max = _ellipse_intersection(
            s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, max_line
        )
    else:
        I_max = I_min
    if (min_line <= s.bbox_argmin[1]) and (s.bbox_argmin[1] < max_line):
        ellipse_min = s.bbox_min[1]
    else:
        ellipse_min = wp.min(I_min[0], I_max[0])
    if (min_line <= s.bbox_argmax[1]) and (s.bbox_argmax[1] < max_line):
        ellipse_max = s.bbox_max[1]
    else:
        ellipse_max = wp.max(I_min[1], I_max[1])
    min_v = wp.max(s.rect_min[1], wp.min(s.rect_max[1], wp.int32(ellipse_min / block)))
    max_v = wp.min(
        s.rect_max[1], wp.max(s.rect_min[1], wp.int32(ellipse_max / block + 1.0))
    )
    return wp.vec4(wp.float32(min_v), wp.float32(max_v), I_max[0], I_max[1])


@wp.func
def _accutile_count(s: AccuSetup, block: wp.float32) -> wp.int32:
    # Total tiles the ellipse touches -> num_tiles_hit (gsplat first_pass count).
    if s.valid == 0:
        return wp.int32(0)
    I_min = _accutile_first_imin(s, block)
    count = wp.int32(0)
    for u in range(s.rect_min[0], s.rect_max[0]):
        r = _accutile_col(u, s, block, I_min)
        count = count + (wp.int32(r[1]) - wp.int32(r[0]))
        I_min = wp.vec2(r[2], r[3])
    return count


@wp.kernel
def _project_kernel(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    num_gaussians: wp.int32,
    sel_means: wp.int32,
    sel_scales: wp.int32,
    sel_quats: wp.int32,
    sel_view: wp.int32,
    sel_opac: wp.int32,
    has_opac: wp.int32,
    img_h: wp.int32,
    img_w: wp.int32,
    fx: wp.float32,
    fy: wp.float32,
    cx: wp.float32,
    cy: wp.float32,
    glob_scale: wp.float32,
    clip_thresh: wp.float32,
    block_width: wp.int32,
    # outputs
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    num_tiles_hit: wp.array[wp.int32],
):
    # Native batch: launched over B*N flat threads (gsplat-style, no host loop).
    # bid = batch element, gid = gaussian within the (shared) scene. Each per-input
    # array is either "batched" (leading dim B*N / 4B, selector=1, index by the flat
    # position) or "broadcast" (leading dim N / 4, selector=0, index by gid / row 0).
    # Outputs are always full batch (B*N), written at the flat idx. For B=1 this is
    # exactly the unbatched path (bid=0, gid=idx, all selectors 0).
    idx = wp.tid()
    n = num_gaussians
    bid = idx // n
    gid = idx % n
    m_idx = wp.where(sel_means != 0, idx, gid)
    s_idx = wp.where(sel_scales != 0, idx, gid)
    q_idx = wp.where(sel_quats != 0, idx, gid)
    o_idx = wp.where(sel_opac != 0, idx, gid)
    vb = wp.where(sel_view != 0, bid, 0) * 4  # row offset into (4B,4) viewmat

    # defaults (match CUDA: radii/num_tiles_hit zeroed; others left for culled)
    radii[idx] = 0
    num_tiles_hit[idx] = 0
    xys[idx] = wp.vec2(0.0, 0.0)
    depths[idx] = 0.0
    conics[idx] = wp.vec3(0.0, 0.0, 0.0)

    mean = means3d[m_idx]
    mx = mean[0]
    my = mean[1]
    mz = mean[2]

    # clip_near_plane: p_view = viewmat @ mean (row-major 4x3)
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

    # scale_rot_to_cov3d
    R = _quat_to_rotmat(quats[q_idx])
    s = scales[s_idx]
    sx = glob_scale * s[0]
    sy = glob_scale * s[1]
    sz = glob_scale * s[2]
    # M = R @ diag(sx, sy, sz)
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
    V3 = M * wp.transpose(M)  # 3d covariance

    # project_cov3d_ewa
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
    # t = W @ mean + translation == p_view
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
    c00 = cov[0, 0]
    c11 = cov[1, 1]
    c01 = cov[0, 1]
    cxx = c00 + 0.3
    cxy = c01
    cyy = c11 + 0.3

    # compute_cov2d_bounds
    det = cxx * cyy - cxy * cxy
    if det == 0.0:
        return
    inv_det = 1.0 / det
    conic = wp.vec3(cyy * inv_det, -cxy * inv_det, cxx * inv_det)

    # project_pix (uses unclamped p_view)
    rw = 1.0 / (pvz + 1e-6)
    center_x = (pvx * rw) * fx + cx
    center_y = (pvy * rw) * fy + cy

    bw = wp.float32(block_width)
    tb_x = (img_w + block_width - 1) / block_width
    tb_y = (img_h + block_width - 1) / block_width

    if has_opac != 0:
        # SNUGBOX + AccuTile (survey O6): opacity-aware tight ellipse. Only the
        # tile COUNT and radii change here; the emission kernel walks the identical
        # ellipse (shared _accutile_* funcs) so counts match exactly.
        opac = opacities[o_idx]
        if opac < ALPHA_THRESHOLD:
            return  # sub-threshold: alpha < 1/255 everywhere -> contributes nothing
        t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
        ext = wp.sqrt(t)
        radius_x = wp.ceil(ext * wp.sqrt(cxx))
        radius_y = wp.ceil(ext * wp.sqrt(cyy))
        if radius_x <= 0.0 and radius_y <= 0.0:
            return
        # off-image reject (gsplat ProjectionEWA3DGSFused.cu:215)
        if (
            center_x + radius_x <= 0.0
            or center_x - radius_x >= wp.float32(img_w)
            or center_y + radius_y <= 0.0
            or center_y - radius_y >= wp.float32(img_h)
        ):
            return
        setup = _accutile_setup(
            conic[0], conic[1], conic[2], t, center_x, center_y, block_width, tb_x, tb_y
        )
        count = _accutile_count(setup, bw)
        if count <= 0:
            return
        num_tiles_hit[idx] = count
        depths[idx] = pvz
        radii[idx] = wp.int32(wp.max(radius_x, radius_y))
        xys[idx] = wp.vec2(center_x, center_y)
        conics[idx] = conic
        return

    # legacy isotropic 3-sigma bbox path (opacities not supplied)
    b = 0.5 * (cxx + cyy)
    v1 = b + wp.sqrt(wp.max(0.1, b * b - det))
    v2 = b - wp.sqrt(wp.max(0.1, b * b - det))
    radius = wp.ceil(3.0 * wp.sqrt(wp.max(v1, v2)))

    # get_tile_bbox -> get_bbox (clamped to tile bounds)
    tc_x = center_x / bw
    tc_y = center_y / bw
    tr = radius / bw
    tmin_x = wp.min(wp.max(0, wp.int32(tc_x - tr)), tb_x)
    tmax_x = wp.min(wp.max(0, wp.int32(tc_x + tr + 1.0)), tb_x)
    tmin_y = wp.min(wp.max(0, wp.int32(tc_y - tr)), tb_y)
    tmax_y = wp.min(wp.max(0, wp.int32(tc_y + tr + 1.0)), tb_y)
    tile_area = (tmax_x - tmin_x) * (tmax_y - tmin_y)
    if tile_area <= 0:
        return

    num_tiles_hit[idx] = tile_area
    depths[idx] = pvz
    radii[idx] = wp.int32(radius)
    xys[idx] = wp.vec2(center_x, center_y)
    conics[idx] = conic


def _project_launch(
    means3d: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    quats: wp.array[wp.vec4],
    viewmat: wp.array2d[wp.float32],
    opacities: wp.array[wp.float32],
    num_gaussians: int,
    has_opac: int,
    img_h: int,
    img_w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
    # outputs
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    num_tiles_hit: wp.array[wp.int32],
    cum_tiles_hit: wp.array[wp.int32],
) -> None:
    # num_gaussians (N) is passed statically because jax.vmap hides the batch axis
    # from the Python wrapper (means3d.shape[0] would read B*N here). B is recovered
    # from an output shape (always full batch under expand_dims: B*N). Each input is
    # batched (leading dim > base -> selector 1) or broadcast (== base -> selector 0).
    n = num_gaussians
    total = xys.shape[0]  # B*N
    sel_means = 1 if means3d.shape[0] > n else 0
    sel_scales = 1 if scales.shape[0] > n else 0
    sel_quats = 1 if quats.shape[0] > n else 0
    sel_view = 1 if viewmat.shape[0] > 4 else 0
    sel_opac = 1 if opacities.shape[0] > n else 0
    wp.launch(
        _project_kernel,
        dim=total,
        inputs=[
            means3d,
            scales,
            quats,
            viewmat,
            opacities,
            n,
            sel_means,
            sel_scales,
            sel_quats,
            sel_view,
            sel_opac,
            has_opac,
            img_h,
            img_w,
            fx,
            fy,
            cx,
            cy,
            glob_scale,
            clip_thresh,
            block_width,
        ],
        outputs=[xys, depths, radii, conics, num_tiles_hit],
    )
    # One global inclusive prefix sum over the flattened B*N tile counts, so all
    # images' intersections are laid out contiguously (matches gsplat's global
    # cumsum + single sort). num_tiles_hit[idx-1] gives each gaussian's global write
    # offset; the total is cum_tiles_hit[-1].
    wp.utils.array_scan(num_tiles_hit, cum_tiles_hit, inclusive=True)


# graph_mode=WARP: Warp captures/replays a CUDA graph keyed on buffer addresses,
# which keeps array_scan (host-side temp management) out of the JAX graph capture;
# a re-capture occurs when a batch size / address changes. WARP_STAGED_EX is the
# documented fallback if XLA donation ever makes addresses truly unstable.
# vmap_method="expand_dims": under jax.vmap every operand gains a leading batch
# axis (size B if mapped, 1 if broadcast); warp's FFI callback collapses it into
# the leading array dim (means -> (B*N,), viewmat -> (4B,4), outputs -> (B*N,)).
# The callable launches natively over B*N and indexes per-input batched/broadcast.
# For a non-vmapped call every array keeps base rank, so this reduces to the
# unbatched path exactly.
_project_ffi = jax_callable(
    _project_launch,
    num_outputs=6,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)


# --- Backward pass -------------------------------------------------------------
# Faithful port of the reference CUDA project_gaussians_bwd and its
# device helpers (project_cov3d_ewa_vjp, scale_rot_to_cov3d_vjp, quat/helper vjps),
# with camera-pose (viewmat) gradients ported from gsplat's
# ProjectionEWA3DGSFused.cu backward (posW2C_VJP / covarW2C_VJP).
#
# ALL vjp math lives in the shared ``wp.func`` helpers below so the three kernel
# variants -- gaussians-only, viewmat-only, both -- carry ZERO duplicated math;
# each variant just composes the helpers it needs and writes the grads it owns.
#
#   * ``_recompute_geom``  : re-derives the forward geometry (W, camera-space t,
#     J, T=JW, quat-rotation R, M=R diag(s), world covariance V=M Mᵀ). cov3d is
#     recomputed here rather than saved (deterministic, bit-identical, saves a 6N
#     residual). Shared by every variant.
#   * ``_vcov2d_from_conic`` : cov2d_to_conic_vjp (v_cov2d = -X G X).
#   * ``_vp_vT_vV`` : project_pix_vjp + depth + project_cov3d_ewa_vjp, returning
#     v_p (grad wrt the camera-space position t, used by BOTH the world-mean grad
#     and the viewmat translation grad), v_T (grad wrt T=JW, used by the viewmat
#     rotation grad) and v_V (grad wrt world covariance, used by scale/quat).
#   * ``_scale_quat_vjp`` : scale_rot_to_cov3d_vjp + quat_to_rotmat_vjp.
#
# Divergences from the reference CUDA backward: cov3d recomputed not saved; blur compensation dropped
# (zero cotangent in the render path); the EWA J is rebuilt from the UNCLAMPED
# camera-space t (standard gsplat approximation; the fov clamp is the identity for
# on-screen gaussians). Backward is path-agnostic (tight and legacy share
# xys/depths/conics).
#
# Viewmat gradient math (derived from splax's own forward, cross-checked against
# gsplat and validated by finite differences -- the reference CUDA backward has no viewmat grads):
#   t = W·mean + trans, with W the upper-left 3x3 of the row-major viewmat and
#   trans its column 3, rows 0-2. Only these 12 entries are differentiable (the
#   last row is the constant [0,0,0,1]). Let v_p be the total grad wrt t (project
#   pixel + depth + EWA-through-J) and v_T the grad wrt T=JW. Then
#       v_trans = v_p
#       v_R     = outer(v_p, mean) + Jᵀ · v_T
#   ( outer(v_p, mean) is the ∂t/∂W term; Jᵀ·v_T is the ∂(T=JW)/∂W term holding J
#   fixed -- J's own W-dependence flows through v_p via t ). v_viewmat[0:3,0:3]=v_R,
#   v_viewmat[0:3,3]=v_trans, v_viewmat[3,:]=0.
#
# Accumulation: every gaussian contributes to ONE shared 12-float v_viewmat. Two
# strategies are provided and benchmarked (scripts/optimize_pose.py --accum-bench):
#   * plain atomics -- each thread issues 12 wp.atomic_add onto the same 12 global
#     addresses (worst-case N-way contention);
#   * block tile_sum -- each block reduces its threads' 12 contributions with
#     wp.tile_sum and thread 0 issues one atomic per entry per block (~N/BLOCK
#     atomics). Projection has uniform per-gaussian work and NO early termination,
#     so the block barrier is amortised and the reduction wins 20-110x; tile_sum
#     is the shipped default (unlike the rasterize backward, where early exit
#     makes the barrier a loss -- see _rasterize.py).


VIEW_BLOCK = wp.constant(256)  # threads per block for the tile_sum v_viewmat reduce


@wp.struct
class _Geom:
    W: wp.mat33  # upper-left 3x3 of the row-major viewmat (== camera rotation)
    R: wp.mat33  # quaternion rotation
    M: wp.mat33  # R diag(glob_scale*scale)
    V: wp.mat33  # world covariance M Mᵀ
    J: wp.mat33  # EWA jacobian (rebuilt from unclamped t)
    T: wp.mat33  # J W
    tx: wp.float32  # camera-space position (unclamped)
    ty: wp.float32
    tz: wp.float32
    rz2: wp.float32
    rz3: wp.float32
    sx: wp.float32  # glob_scale*scale
    sy: wp.float32
    sz: wp.float32


@wp.struct
class _VP:
    v_p: wp.vec3  # grad wrt camera-space position t (pix + depth + EWA)
    v_T: wp.mat33  # grad wrt T = J W
    v_V: wp.mat33  # grad wrt world covariance V


@wp.struct
class _SQ:
    v_scale: wp.vec3
    v_quat: wp.vec4


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
    # cov2d_to_conic_vjp: v_cov2d = -X G X (X = conic, G = v_conic upper triangle).
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
def _vp_vT_vV(
    g: _Geom,
    fx: wp.float32,
    fy: wp.float32,
    v_xy: wp.vec2,
    v_depth: wp.float32,
    vcov2d: wp.vec3,
) -> _VP:
    out = _VP()
    tx = g.tx
    ty = g.ty
    tz = g.tz
    rz2 = g.rz2
    rz3 = g.rz3
    # project_pix_vjp (center xy -> camera-space position).
    rw = 1.0 / (tz + 1e-6)
    vpx = fx * v_xy[0]
    vpy = fy * v_xy[1]
    vvx = vpx * rw
    vvy = vpy * rw
    vvz = -(vpx * tx + vpy * ty) * rw * rw
    # depth cotangent adds straight onto the z component of the position grad.
    vvz = vvz + v_depth
    # project_cov3d_ewa_vjp: cov = T V Tᵀ, T = J W.
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
    out.v_V = Tt * v_cov * g.T  # df/dV (symmetric)
    v_T = v_cov * g.T * g.V + wp.transpose(v_cov) * g.T * g.V
    out.v_T = v_T
    v_J = v_T * wp.transpose(g.W)
    v_t_x = -fx * rz2 * v_J[0, 2]
    v_t_y = -fy * rz2 * v_J[1, 2]
    v_t_z = (
        -fx * rz2 * v_J[0, 0]
        + 2.0 * fx * tx * rz3 * v_J[0, 2]
        - fy * rz2 * v_J[1, 1]
        + 2.0 * fy * ty * rz3 * v_J[1, 2]
    )
    out.v_p = wp.vec3(vvx + v_t_x, vvy + v_t_y, vvz + v_t_z)
    return out


@wp.func
def _scale_quat_vjp(
    g: _Geom, quat: wp.vec4, v_V: wp.mat33, glob_scale: wp.float32
) -> _SQ:
    out = _SQ()
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
    out.v_scale = wp.vec3(
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
    out.v_quat = wp.vec4(vq_w, vq_x, vq_y, vq_z)
    return out


@wp.func
def _load_W_trans(viewmat: wp.array2d[wp.float32], vb: wp.int32):
    # Load the upper-left 3x3 W and translation from the viewmat row-block starting
    # at row ``vb`` (vb = image_id*4 under batching; 0 unbatched / broadcast viewmat).
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
def _view_grad(g: _Geom, mean: wp.vec3, vp: _VP):
    # v_R = outer(v_p, mean) + Jᵀ v_T ; v_trans = v_p  (see header derivation).
    v_R = wp.outer(vp.v_p, mean) + wp.transpose(g.J) * vp.v_T
    return v_R, vp.v_p


# --- Kernel variant 1: gaussians only (default) --------------------------------
# Launched over B*N flat threads (mirrors the forward _project_kernel). bid decodes
# the image, gid the gaussian; per-input selectors index the batched (B*N / 4B) or
# broadcast (N / 4) residual arrays. The gaussian grads are written per-view at the
# flat idx (full batch B*N); when the gaussian inputs were broadcast under vmap, JAX
# reduces those per-view grads over the batch axis (the vjp of a broadcast sums).
# For B=1 every selector is 0 and idx==gid: the plain single-image path.
@wp.kernel
def _project_bwd_kernel(
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
    # Every operand is either batched (own leading dim B*N -> read at idx) or broadcast
    # (leading dim N -> read at gid). radii/conics are forward geometry; v_xy/v_depth/
    # v_conic are cotangents. A cotangent can arrive BROADCAST even when geometry is
    # batched -- the depth cotangent is zero for an image loss (depth only drives the
    # integer sort), so JAX hands it back at size N; indexing it at idx would read OOB.
    r_idx = wp.where(sel_radii != 0, idx, gid)
    if radii[r_idx] <= 0:
        return  # culled gaussian: leave pre-zeroed 0 grad
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
    vp = _vp_vT_vV(g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
    v_mean3d[idx] = wp.transpose(g.W) * vp.v_p
    sq = _scale_quat_vjp(g, quats[q_idx], vp.v_V, glob_scale)
    v_scale[idx] = sq.v_scale
    v_quat[idx] = sq.v_quat


# --- Kernel variant 2: viewmat only --------------------------------------------
# Skips the whole scale/quat/cov3d grad chain and the five gaussian grad arrays;
# accumulates only v_viewmat. Two accumulation strategies (atomic / tile_sum).
# The accumulator is per-image: under vmap the viewmat output is (4B,4) and
# image_id's 12-float slot is the row-block image_id*4. Each gaussian contributes
# to its own image's slot only, so the per-view camera grads come out independent
# (vmap over poses -> per-pose grads). For B=1 the slot is the (4,4) matrix.
@wp.kernel
def _project_bwd_view_atomic_kernel(
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
    v_viewmat: wp.array2d[wp.float32],
):
    idx = wp.tid()
    n = num_gaussians
    bid = idx // n
    gid = idx % n
    r_idx = wp.where(sel_radii != 0, idx, gid)
    if radii[r_idx] <= 0:
        return
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
    vp = _vp_vT_vV(g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
    v_R, v_t = _view_grad(g, mean, vp)
    ob = bid * 4  # row-block of this image's 12-float accumulator
    for i in range(3):
        wp.atomic_add(v_viewmat, ob + i, 0, v_R[i, 0])
        wp.atomic_add(v_viewmat, ob + i, 1, v_R[i, 1])
        wp.atomic_add(v_viewmat, ob + i, 2, v_R[i, 2])
        wp.atomic_add(v_viewmat, ob + i, 3, v_t[i])


@wp.kernel
def _project_bwd_view_tile_kernel(
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
    # launch_tiled over B*blocks_per_image blocks: each block belongs to ONE image
    # (image_id = block // blocks_per_image), so the block-collective tile_sum never
    # crosses an image boundary. Threads outside N / culled contribute 0 but MUST
    # still take part in the tile_sum. Thread 0 issues one atomic per entry into
    # the image's row-block (blocks_per_image = ceil(N/256) per image).
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
        vp = _vp_vT_vV(g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
        v_R, v_t = _view_grad(g, mean, vp)
    ob = image_id * 4
    for i in range(3):
        for j in range(3):
            s = wp.tile_sum(wp.tile(v_R[i, j]))
            if tr == 0:
                wp.atomic_add(v_viewmat, ob + i, j, wp.tile_extract(s, 0))
        st = wp.tile_sum(wp.tile(v_t[i]))
        if tr == 0:
            wp.atomic_add(v_viewmat, ob + i, 3, wp.tile_extract(st, 0))


# --- Kernel variant 3: both (gaussians + viewmat) -----------------------------
@wp.kernel
def _project_bwd_both_kernel(
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
    # Per-image block layout (like _project_bwd_view_tile_kernel) so the viewmat
    # tile_sum stays within one image; gaussian grads written per-view at the flat
    # idx (full batch B*N) exactly as _project_bwd_kernel.
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
        vp = _vp_vT_vV(g, fx, fy, v_xy_in[vx_idx], v_depth_in[vd_idx], vcov2d)
        # gaussian grads: IDENTICAL composition/order to _project_bwd_kernel, so the
        # gaussian grads are bit-for-bit the same as diff_wrt=("gaussians",).
        v_mean3d[idx] = wp.transpose(g.W) * vp.v_p
        sq = _scale_quat_vjp(g, quats[q_idx], vp.v_V, glob_scale)
        v_scale[idx] = sq.v_scale
        v_quat[idx] = sq.v_quat
        v_R, v_t = _view_grad(g, mean, vp)
    ob = image_id * 4
    for i in range(3):
        for j in range(3):
            s = wp.tile_sum(wp.tile(v_R[i, j]))
            if tr == 0:
                wp.atomic_add(v_viewmat, ob + i, j, wp.tile_extract(s, 0))
        st = wp.tile_sum(wp.tile(v_t[i]))
        if tr == 0:
            wp.atomic_add(v_viewmat, ob + i, 3, wp.tile_extract(st, 0))


# --- Launch wrappers ----------------------------------------------------------
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
    # B is recovered by the caller from an OUTPUT leading dim: under expand_dims JAX
    # always prepends the full batch B to every output, so an output is the only
    # reliable B signal. Each operand is independently batched (own leading dim
    # B*N / 4B -> read at the flat idx) or broadcast (N / 4 -> read at gid), so we
    # pass one selector per operand and index it in-kernel exactly like the forward.
    #
    # This is required for CORRECTNESS, not just robustness: a cotangent can arrive
    # BROADCAST even when the geometry is fully batched. The depth cotangent is a
    # zero for any image loss (depth only drives the integer tile sort, which is
    # non-differentiable), so JAX materializes it at size N; indexing it at the flat
    # B*N idx would read out of bounds and corrupt v_p.z -> the third viewmat row.
    def sel(a: "wp.array | wp.array2d[wp.float32]", base: int) -> int:
        # every operand is a real wp.array at runtime; array2d is a stub-only alias.
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


def _project_bwd_launch(
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
        _project_bwd_kernel,
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


# Module switch: tile_sum (default, benchmarked winner) vs plain atomics. Flipped
# by scripts/optimize_pose.py --accum-bench to time both variants.
_VIEW_ACCUM_TILE = True


def _project_bwd_view_launch(
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
    if _VIEW_ACCUM_TILE:
        blocks_per_image = (n + _BWD_BLOCK - 1) // _BWD_BLOCK
        wp.launch_tiled(
            _project_bwd_view_tile_kernel,
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
    else:
        wp.launch(
            _project_bwd_view_atomic_kernel,
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
            outputs=[v_viewmat],
            device=means3d.device,
        )


def _project_bwd_both_launch(
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
        _project_bwd_both_kernel,
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


# vmap_method="expand_dims": batch-native backward. Under jax.vmap the launch
# recovers B and indexes per-input batched/broadcast, exactly like the forward.
# Gaussian grads are produced per-view (leading B); JAX reduces broadcast
# (shared-gaussian) inputs over the batch axis (the vjp of a broadcast sums). The
# viewmat grad is a per-image accumulator (independent per pose). For B=1 (no vmap)
# every kernel runs the plain single-image path.
# graph_mode=WARP: no host readback / data-dependent shapes.
_project_bwd_ffi = jax_callable(
    _project_bwd_launch,
    num_outputs=3,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)
_project_bwd_view_ffi = jax_callable(
    _project_bwd_view_launch,
    num_outputs=1,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)
_project_bwd_both_ffi = jax_callable(
    _project_bwd_both_launch,
    num_outputs=4,
    graph_mode=JaxCallableGraphMode.WARP,
    vmap_method="expand_dims",
)


def _project_call(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    n: int,
    has_opac: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    H, W = img_shape
    xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit = _project_ffi(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        int(n),
        int(has_opac),
        int(H),
        int(W),
        float(f[0]),
        float(f[1]),
        float(c[0]),
        float(c[1]),
        float(glob_scale),
        float(clip_thresh),
        int(block_width),
        output_dims=n,
    )
    depths = depths.reshape(n, 1)
    radii = radii.reshape(n, 1)
    num_tiles_hit = num_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    cum_tiles_hit = cum_tiles_hit.reshape(n, 1).astype(jnp.uint32)
    return xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit


# The forward + residuals are identical across the three diff_wrt variants; only the
# backward rule (which cotangents it produces, which kernel it runs) differs. A
# custom_vjp bwd cannot see which cotangents the caller wants, so we build one
# custom_vjp function object per diff_wrt selection and dispatch in project().
def _project_fwd_rule(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    opac: jax.Array,
    n: int,
    has_opac: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    out = _project_call(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        n,
        has_opac,
        img_shape,
        f,
        c,
        glob_scale,
        clip_thresh,
        block_width,
    )
    _xys, _depths, radii, conics, _nth, _cum = out
    residuals = (mean3ds, scales, quats, viewmat, radii, conics)
    return out, residuals


def _bwd_common_inputs(
    residuals: tuple[jax.Array, ...], cotangents: tuple[jax.Array, ...], n: int
) -> tuple[jax.Array, ...]:
    mean3ds, scales, quats, viewmat, radii, conics = residuals
    v_xys, v_depths, _v_radii, v_conics, _v_nth, _v_cum = cotangents
    return (
        mean3ds,
        scales,
        quats,
        viewmat,
        radii.reshape(n).astype(jnp.int32),
        conics,
        v_xys,
        v_depths.reshape(n),
        v_conics,
    )


def _make_diff_variant(bwd_rule: Callable) -> Callable:
    @partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
    def variant(
        mean3ds: jax.Array,
        scales: jax.Array,
        quats: jax.Array,
        viewmat: jax.Array,
        opac: jax.Array,
        n: int,
        has_opac: int,
        img_shape: tuple[int, int],
        f: tuple[float, float],
        c: tuple[float, float],
        glob_scale: float,
        clip_thresh: float,
        block_width: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return _project_call(
            mean3ds,
            scales,
            quats,
            viewmat,
            opac,
            n,
            has_opac,
            img_shape,
            f,
            c,
            glob_scale,
            clip_thresh,
            block_width,
        )

    variant.defvjp(_project_fwd_rule, bwd_rule)
    return variant


def _project_diff_bwd(
    n: int,
    has_opac: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
    m, s, q, vm, r, cn, vx, vd, vc = _bwd_common_inputs(residuals, cotangents, n)
    v_mean, v_scale, v_quat = _project_bwd_ffi(
        m,
        s,
        q,
        vm,
        r,
        cn,
        vx,
        vd,
        vc,
        int(n),
        float(f[0]),
        float(f[1]),
        float(glob_scale),
        output_dims=n,
    )
    return (v_mean, v_scale, v_quat, None, None)


def _project_diff_view_bwd(
    n: int,
    has_opac: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
    m, s, q, vm, r, cn, vx, vd, vc = _bwd_common_inputs(residuals, cotangents, n)
    (v_viewmat,) = _project_bwd_view_ffi(
        m,
        s,
        q,
        vm,
        r,
        cn,
        vx,
        vd,
        vc,
        int(n),
        float(f[0]),
        float(f[1]),
        float(glob_scale),
        output_dims=(4, 4),
    )
    # Only viewmat carries a gradient; gaussian inputs get None (they are constants
    # for pose-only optimization -- the whole point of the split).
    return (None, None, None, v_viewmat, None)


def _project_diff_both_bwd(
    n: int,
    has_opac: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
) -> tuple[jax.Array | None, ...]:
    m, s, q, vm, r, cn, vx, vd, vc = _bwd_common_inputs(residuals, cotangents, n)
    v_mean, v_scale, v_quat, v_viewmat = _project_bwd_both_ffi(
        m,
        s,
        q,
        vm,
        r,
        cn,
        vx,
        vd,
        vc,
        int(n),
        float(f[0]),
        float(f[1]),
        float(glob_scale),
        output_dims={"v_mean3d": n, "v_scale": n, "v_quat": n, "v_viewmat": (4, 4)},
    )
    return (v_mean, v_scale, v_quat, v_viewmat, None)


_DIFF_VARIANTS = {
    ("gaussians",): _make_diff_variant(_project_diff_bwd),
    ("viewmat",): _make_diff_variant(_project_diff_view_bwd),
    ("gaussians", "viewmat"): _make_diff_variant(_project_diff_both_bwd),
}


def _normalize_diff_wrt(diff_wrt: tuple[str, ...]) -> tuple[str, ...]:
    key = tuple(diff_wrt)
    if key not in _DIFF_VARIANTS:
        key = tuple(sorted(key))  # accept ("viewmat", "gaussians") order too
    if key not in _DIFF_VARIANTS:
        raise ValueError(
            f"diff_wrt must be one of ('gaussians',), ('viewmat',), "
            f"('gaussians','viewmat'); got {diff_wrt!r}"
        )
    return key


def opacity_compensation(
    conics: jax.Array, radii: jax.Array, eps: float = 0.3
) -> jax.Array:
    """Mip-Splatting anti-aliased opacity compensation factor ρ, per gaussian.

    ρ = √(det(Σ₂D) / det(Σ₂D + εI)) is the det-ratio of the *undilated* 2D
    covariance over the ε-dilated one that projection already applies (``+0.3``
    screen-space blur, cxx=c00+ε …). Multiplying ρ (≤ 1) into the opacity before
    the blend cancels the artificial area inflation the dilation grants thin
    gaussians, so they stop being rewarded for hiding under the blur
    (Mip-Splatting, Yu et al., CVPR 2024; gsplat ``rasterize_mode="antialiased"``).

    Computed here from the projection's own ``conics`` output — the conic is the
    inverse of the ε-dilated Σ₂D, so with (a,b,c)=conic the det ratio has the
    exact closed form (no matrix inverse, no division):

        ρ² = 1 − ε·(a + c) + ε²·(a·c − b²)

    (derivation: conic = inv(Σ₂D+εI) ⇒ a+c = tr/det_d, a·c−b² = 1/det_d, and
    det(Σ₂D) = det_d − ε·tr + ε²; substitute). Because ρ is a smooth function of
    ``conics`` — a differentiable projection output — its gradient flows back to
    scales/quats/means through project's existing conic→covariance vjp
    (``_vcov2d_from_conic``) with **no** change to any Warp kernel; the projection
    forward FFI stays byte-identical. Clipped to [0, 1] for fp safety; culled
    gaussians (radii ≤ 0, conic = 0) get ρ = 1 (the formula already yields 1 at
    conic = 0, so this is a no-op guard).
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
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
    opacities: jax.Array | None = None,
    diff_wrt: tuple[str, ...] = ("gaussians",),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Warp projection returning the standard 3DGS projection outputs.

    Returns (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit).

    Differentiable via jax.custom_vjp. ``diff_wrt`` selects the backward variant:
    ``("gaussians",)`` (default; grads w.r.t. means/scales/quats, viewmat treated
    as constant), ``("viewmat",)`` (only the camera-pose grad; the gaussian grad
    chains and their atomics are skipped -- post-training pose optimization pays
    only for the camera gradient), or
    ``("gaussians", "viewmat")`` (both). Each selection is wired to its own
    custom_vjp object and matching Warp kernel, because a custom_vjp backward rule
    cannot observe which cotangents the caller wants. radii/num_tiles_hit/
    cum_tiles_hit are integer (non-diff); the depth/xy/conic cotangents drive the
    backward. When no gradient is requested the primal runs exactly as the
    forward-only path (custom_vjp only intercepts differentiation).

    ``opacities`` is an optional extension over the legacy projection: when supplied it
    enables the opacity-aware SNUGBOX + AccuTile tight tile intersection (survey
    O6), yielding per-axis radii and an ellipse-walk tile count that both drop
    sub-1/255 tails. Default ``None`` reproduces the legacy isotropic 3-sigma bbox
    behavior exactly, so the public signature stays compatible. opacities is not
    differentiated through project (it only affects integer tile counts); the
    opacity gradient flows through rasterize.
    """
    n = mean3ds.shape[0]
    if opacities is None:
        # dummy broadcast operand; has_opac=0 selects the legacy path in-kernel.
        opac = jnp.zeros((n,), jnp.float32)
        has_opac = 0
    else:
        opac = opacities.reshape(n)
        has_opac = 1
    variant = _DIFF_VARIANTS[_normalize_diff_wrt(diff_wrt)]
    return variant(
        mean3ds,
        scales,
        quats,
        viewmat,
        opac,
        int(n),
        int(has_opac),
        img_shape,
        f,
        c,
        float(glob_scale),
        float(clip_thresh),
        int(block_width),
    )
