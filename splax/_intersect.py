"""Opacity-aware tight tile intersection shared by projection and rasterization.

Ports gsplat's SpeedySplat path, its SNUGBOX and AccuTile stages. Instead of an isotropic 3-sigma
bounding box, each gaussian gets a tight ellipse at the opacity-aware isocontour level. Outside that
level the gaussian's alpha drops below ``1/255``, so tiles past it are invisible. The AccuTile walk
marches the ellipse column by column and emits only the tiles its boundary spans. Projection counts
tiles with the same walk rasterization uses to emit sort keys, so the counted and emitted totals
agree.
"""

import warp as wp

wp.init()

wp.set_module_options({"fast_math": True})  # fastmath significantly accelerates the kernels

BLOCK_WIDTH = wp.constant(16)  # Tile shapes must be static compile-time constants
BLOCK_SIZE = wp.constant(256)  # One 16x16 pixel tile is processed by one 256-thread block
GAUSSIAN_EXTEND_SQ = wp.constant(3.33 * 3.33)  # squared max ellipse extent in sigma
ALPHA_THRESHOLD = wp.constant(1.0 / 255.0)


@wp.struct
class _Ellipse:
    """Opacity-aware ellipse and its tile walk state, from gsplat's SNUGBOX and AccuTile."""

    A: wp.float32  # conic, the inverse 2d covariance upper triangle
    B: wp.float32
    C: wp.float32
    disc: wp.float32  # B*B - A*C, negative for a real ellipse
    t: wp.float32  # opacity-aware isocontour level
    px: wp.float32  # ellipse center in image pixels, un-swapped
    py: wp.float32
    # bbox_* and rect_* store the walk's outer axis in component 0. An x-major walk keeps x then y,
    # and a y-major walk swaps them.
    bbox_min: wp.vec2
    bbox_max: wp.vec2
    bbox_argmin: wp.vec2
    bbox_argmax: wp.vec2
    rect_min: wp.vec2i
    rect_max: wp.vec2i
    valid: wp.bool  # True if the ellipse's tile rectangle is non-empty
    isY: wp.bool  # True if the walk marches over the y tiles with the shorter span outer


@wp.func
def ellipse_init(
    A: wp.float32,
    B: wp.float32,
    C: wp.float32,
    t: wp.float32,
    px: wp.float32,
    py: wp.float32,
    tile_width: wp.int32,
    tile_height: wp.int32,
) -> _Ellipse:
    """Initialize the ellipse and its tile walk state.

    We compute the tight axis-aligned bounding box of the ellipse plus its tile rectangle, and then
    pick the shorter tile span as the walk's outer axis.
    """
    s = _Ellipse()
    s.valid = wp.bool(False)
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
    ts = wp.float32(BLOCK_WIDTH)
    rminx = wp.max(0, wp.min(tile_width, wp.int32(bbox_min[0] / ts)))
    rminy = wp.max(0, wp.min(tile_height, wp.int32(bbox_min[1] / ts)))
    rmaxx = wp.max(0, wp.min(tile_width, wp.int32(bbox_max[0] / ts + 1.0)))
    rmaxy = wp.max(0, wp.min(tile_height, wp.int32(bbox_max[1] / ts + 1.0)))
    x_span = rmaxx - rminx
    y_span = rmaxy - rminy
    if y_span * x_span == 0:
        return s
    isY = y_span < x_span
    s.isY = isY
    if isY:
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
    s.valid = wp.bool(True)
    return s


@wp.func
def _ellipse_intersect(
    A: wp.float32,
    B: wp.float32,
    C: wp.float32,
    disc: wp.float32,
    t: wp.float32,
    px: wp.float32,
    py: wp.float32,
    isY: wp.bool,
    coord: wp.float32,
) -> wp.vec2:
    """Compute the cross-axis extent of the ellipse at a given line along the outer axis."""
    if isY:
        p_u = py
        p_v = px
        coeff = A
    else:
        p_u = px
        p_v = py
        coeff = C
    h = coord - p_u
    sqrt_term = wp.sqrt(disc * h * h + t * coeff)
    return wp.vec2((-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v)


@wp.func
def ellipse_init_span(s: _Ellipse) -> wp.vec2:
    """Cross-axis extent of the ellipse at the leading line of the walk."""
    min_line0 = wp.float32(s.rect_min[0]) * wp.float32(BLOCK_WIDTH)
    # Return a degenerate default when the line lies outside the bbox.
    if s.bbox_min[0] <= min_line0:
        return _ellipse_intersect(s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, min_line0)
    return wp.vec2(s.bbox_max[1], s.bbox_min[1])


@wp.func
def ellipse_column_tile_range(u: wp.int32, s: _Ellipse, I_min: wp.vec2) -> wp.vec4:
    """Cross-axis tile range of the ellipse at one outer tile column."""
    # One outer column of the walk. Returns min_tile_v, max_tile_v, and I_max, where I_max feeds the
    # next column as its I_min following gsplat's rolling intersect lines. The cross-axis tile range
    # is [min_v, max_v).
    block = wp.float32(BLOCK_WIDTH)
    min_line = wp.float32(u) * block
    max_line = min_line + block
    if max_line <= s.bbox_max[0]:
        I_max = _ellipse_intersect(s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, max_line)
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
    max_v = wp.min(s.rect_max[1], wp.max(s.rect_min[1], wp.int32(ellipse_max / block + 1.0)))
    return wp.vec4(wp.float32(min_v), wp.float32(max_v), I_max[0], I_max[1])


@wp.func
def ellipse_tile_count(s: _Ellipse) -> wp.int32:
    # Total tiles the ellipse touches, written to num_tiles_hit by projection.
    if not s.valid:
        return wp.int32(0)
    I_min = ellipse_init_span(s)
    count = wp.int32(0)
    for u in range(s.rect_min[0], s.rect_max[0]):
        r = ellipse_column_tile_range(u, s, I_min)
        count = count + (wp.int32(r[1]) - wp.int32(r[0]))
        I_min = wp.vec2(r[2], r[3])
    return count
