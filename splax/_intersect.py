"""Tile intersection, sort keys, and bin edges shared by projection and rasterization.

The opacity-aware tight tile intersection ports gsplat's SpeedySplat path, its SNUGBOX and AccuTile
stages. Instead of an isotropic 3-sigma bounding box, each gaussian gets a tight ellipse at the
opacity-aware isocontour level. Outside that level the gaussian's alpha drops below ``1/255``, so
tiles past it are invisible. The AccuTile walk marches the ellipse column by column and emits only
the tiles its boundary spans. Projection counts tiles with the same walk rasterization uses to emit
sort keys, so the counted and emitted totals agree bit for bit.

Sort keys come in two widths. When the image and tile ids leave at least 16 low bits, the whole key
packs into one non-negative int32 holding the image id, the tile id, and the quantized depth. This
halves the radix sort passes and quarters the bytes moved, dropping the sort stage by 2.7 to 3.1x on
large frames. Otherwise a 64 bit key carrying the image and tile ids above the depth bits is the
automatic fallback.
"""

import warp as wp

wp.init()

# Compile with approximate transcendentals and fp contraction, matching the CUDA reference's
# -use_fast_math build flag.
wp.set_module_options({"fast_math": True})

# One 16x16 pixel tile is processed by one 256-thread block. Tile shapes must be static, so these
# are compile-time constants and the only supported geometry.
BLOCK_WIDTH = wp.constant(16)
BLOCK_SIZE = wp.constant(256)

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
def _ellipse_intersection(
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
def ellipse_setup(
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
def ellipse_init_span(s: _Ellipse) -> wp.vec2:
    """Cross-axis extent of the ellipse at the leading line of the walk."""
    min_line0 = wp.float32(s.rect_min[0]) * wp.float32(BLOCK_WIDTH)
    # Return a degenerate default when the line lies outside the bbox.
    if s.bbox_min[0] <= min_line0:
        return _ellipse_intersection(s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, min_line0)
    return wp.vec2(s.bbox_max[1], s.bbox_min[1])


@wp.func
def ellipse_column(u: wp.int32, s: _Ellipse, I_min: wp.vec2) -> wp.vec4:
    """Cross-axis tile range of the ellipse at one outer tile column."""
    # One outer column of the walk. Returns min_tile_v, max_tile_v, and I_max, where I_max feeds the
    # next column as its I_min following gsplat's rolling intersect lines. The cross-axis tile range
    # is [min_v, max_v).
    block = wp.float32(BLOCK_WIDTH)
    min_line = wp.float32(u) * block
    max_line = min_line + block
    if max_line <= s.bbox_max[0]:
        I_max = _ellipse_intersection(s.A, s.B, s.C, s.disc, s.t, s.px, s.py, s.isY, max_line)
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
        r = ellipse_column(u, s, I_min)
        count = count + (wp.int32(r[1]) - wp.int32(r[0]))
        I_min = wp.vec2(r[2], r[3])
    return count


# Persistent grow-only scratch cache. The pipeline needs three buffers whose sizes depend on the
# data-dependent intersection count:
#   isect_ids / gaussian_ids  radix sort key and value ping-pong buffers. Warp's radix_sort_pairs
#     mandates a 2*count backing array because it drives a cub::DoubleBuffer internally.
#   tile_bins  per-image per-tile bin edges of length B*num_tiles.
# Re-allocating from the mempool every frame costs a cold ~14 ms re-malloc of ~3 GB at B=8, 1M
# gaussians, and 1080p. One set of buffers is kept per device, keyed on the static shape signature
# over B, N, and num_tiles. Callers invoke this from jitted functions, so the signature is fixed per
# compiled executable and repeated calls reuse stable buffers. A signature change drops everything,
# so a big config's scratch never lingers into a smaller one. Within one signature the sort buffers
# track the running max intersection count with 1.25x headroom.
# The sort buffers are fully overwritten in their valid prefix every frame, so there is no stale
# data hazard. tile_bins is the exception. The bin-edge kernel only touches bins that own
# intersections, so the used prefix must be zeroed each frame.
_SCRATCH_HEADROOM = 1.25
_scratch_cache: dict = {}

# Recorded per-kernel launches, keyed on (kernel, device). Each entry is a
# (wp.Launch, state) pair, where state holds the dim followed by one entry per
# argument, arrays as (ptr, shape) and scalars by value. wp.launch resolves the
# device, module, and hooks and repacks every argument on each call, which costs
# 5 to 16 us of host time per launch on kernels with many arguments. A recorded
# wp.Launch replays in about 2 us, and diffing against the state repacks only
# the arguments that changed. Repacking all arguments through set_params instead
# costs 50 to 60 us more per frame over the five pipeline launches, so the diff
# is load-bearing. Purged together with the scratch cache so a recorded launch
# never keeps a freed scratch buffer alive through its retained argument list.
_launch_cache: dict = {}


def _cached_launch(
    kernel: wp.Kernel, dim: int, args: list, device: wp.Device | None, block_dim: int = 0
) -> None:
    """Launch a kernel through its recorded per-device launch object.

    Records the launch on first use and afterwards replays it, repacking only the
    arguments whose recorded state changed. FFI callbacks are serialized by
    Warp's callback lock, so the state needs no locking of its own.
    """
    entry = _launch_cache.get((kernel, str(device)))
    if entry is None:
        if block_dim:
            launch = wp.launch_tiled(
                kernel, dim=[dim], inputs=args, block_dim=block_dim, device=device, record_cmd=True
            )
        else:
            launch = wp.launch(kernel, dim=dim, inputs=args, device=device, record_cmd=True)
        state = [dim] + [(a.ptr, a.shape) if isinstance(a, wp.array) else a for a in args]
        _launch_cache[(kernel, str(device))] = (launch, state)
        launch.launch()
        return
    launch, state = entry
    if state[0] != dim:
        state[0] = dim
        launch.set_dim([dim, block_dim] if block_dim else dim)
    for i, a in enumerate(args, start=1):
        key = (a.ptr, a.shape) if isinstance(a, wp.array) else a
        if state[i] != key:
            state[i] = key
            launch.set_param_at_index(i - 1, a)
    launch.launch()


# One pinned int32 staging element per device for the intersection-count
# readback, with an event fencing the copy. A pageable slice.numpy() readback
# allocates on every frame and costs about 17 us of host time on an idle
# stream, the pinned copy plus event sync about 6 us. The begin/end split lets
# the caller enqueue count-independent kernels between the copy and the wait,
# so they execute inside the sync bubble without delaying the readback.
_readback_bufs: dict = {}


def _read_count_begin(src: wp.array, index: int, device: wp.Device | None) -> tuple | int:
    """Enqueue the one-element readback copy and fence it with an event.

    On non-CUDA devices the read is synchronous and the value is returned
    directly. Pass the result to _read_count_end for the integer either way.
    """
    if device is None or not device.is_cuda:
        return int(src[index : index + 1].numpy()[0])
    key = str(device)
    buf = _readback_bufs.get(key)
    if buf is None:
        staging = wp.empty(1, dtype=wp.int32, device="cpu", pinned=True)
        buf = (staging, staging.numpy(), wp.Event(device))
        _readback_bufs[key] = buf
    wp.copy(buf[0], src, dest_offset=0, src_offset=index, count=1)
    wp.record_event(buf[2])
    return buf


def _read_count_end(pending: tuple | int) -> int:
    """Wait for the readback copy and return the count."""
    if isinstance(pending, int):
        return pending
    wp.synchronize_event(pending[2])
    return int(pending[1][0])


def _get_scratch(
    device: wp.Device | None,
    sig: tuple,
    isect_need: int,
    bins_need: int,
    isect_dtype: type = wp.int64,
) -> dict:
    key = str(device)  # wp.Device is unhashable, its string alias is stable
    entry = _scratch_cache.get(key)
    if entry is None or entry["sig"] != sig:
        # New workload signature. Drop everything first so the peak is the new
        # size, not old plus new. Purge the recorded launches first, they hold
        # the old array references.
        _launch_cache.clear()
        _scratch_cache.pop(key, None)
        entry = {
            "sig": sig,
            "isect_cap": 0,
            "isect_dtype": isect_dtype,
            "isect_ids": None,
            "gaussian_ids": None,
            "tile_bins": wp.empty(bins_need, dtype=wp.vec2i, device=device),
            "depth_mm": wp.empty(2 * max(sig[0], 1), dtype=wp.float32, device=device),
        }
        _scratch_cache[key] = entry
    if entry["isect_cap"] < isect_need or entry["isect_dtype"] != isect_dtype:
        cap = max(int(isect_need * _SCRATCH_HEADROOM) + 1, entry["isect_cap"])
        # Sort buffers move on realloc, so recorded launches must drop their
        # references to the old buffers before the free below.
        _launch_cache.clear()
        # Free before allocating larger, avoiding an old plus new transient peak.
        entry["isect_ids"] = None
        entry["gaussian_ids"] = None
        entry["isect_cap"] = 0
        entry["isect_dtype"] = isect_dtype
        entry["isect_ids"] = wp.empty(cap, dtype=isect_dtype, device=device)
        entry["gaussian_ids"] = wp.empty(cap, dtype=wp.int32, device=device)
        entry["isect_cap"] = cap
    return entry


def clear_scratch() -> None:
    """Release the persistent sort and bin scratch buffers on all devices.

    The backend caches grow-only scratch across renders. Call this to reclaim that
    memory, for example before switching to a very different workload size. Also
    purges the recorded launch cache, which references the freed scratch buffers.
    """
    _launch_cache.clear()
    _scratch_cache.clear()


# Each thread privately reduces this many gaussians before one atomic pair,
# cutting global atomics and their contention by the same factor.
_MINMAX_CHUNK = wp.constant(32)


@wp.kernel
def _seed_minmax(out_mm: wp.array[wp.float32]):
    b = wp.tid()  # one thread per image
    out_mm[2 * b] = 1.0e30
    out_mm[2 * b + 1] = -1.0e30


@wp.kernel
def _depth_minmax(
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    total: wp.int32,
    num_gaussians: wp.int32,
    # output, per-image [min, max] pairs of length 2*B, pre-seeded by _seed_minmax
    out_mm: wp.array[wp.float32],
):
    # The range is kept per image (image = idx // n) so a batched render quantizes
    # each view exactly as the corresponding unbatched render would. A chunk spans
    # at most one image boundary (n >> 32), handled by flushing the accumulator
    # when the image changes.
    tid = wp.tid()
    base = tid * _MINMAX_CHUNK
    img_cur = wp.int32(-1)
    lo = wp.float32(1.0e30)
    hi = wp.float32(-1.0e30)
    for k in range(_MINMAX_CHUNK):
        idx = base + k
        if idx < total:
            if radii[idx] > 0:  # culled gaussians emit no keys, exclude from range
                im = idx // num_gaussians
                if im != img_cur:
                    if img_cur >= 0:
                        wp.atomic_min(out_mm, 2 * img_cur, lo)
                        wp.atomic_max(out_mm, 2 * img_cur + 1, hi)
                    img_cur = im
                    lo = depths[idx]
                    hi = depths[idx]
                else:
                    lo = wp.min(lo, depths[idx])
                    hi = wp.max(hi, depths[idx])
    if img_cur >= 0:
        wp.atomic_min(out_mm, 2 * img_cur, lo)
        wp.atomic_max(out_mm, 2 * img_cur + 1, hi)


@wp.kernel
def _map_intersects_32bit(
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    map_opacities: wp.array[wp.float32],
    cum_tiles_hit: wp.array[wp.int32],
    depth_mm: wp.array[wp.float32],
    num_gaussians: wp.int32,
    opac_mod: wp.int32,
    tile_n_bits: wp.int32,
    depth_bits: wp.int32,
    tile_bounds_x: wp.int32,
    tile_bounds_y: wp.int32,
    # outputs
    isect_ids: wp.array[wp.int32],
    gaussian_ids: wp.array[wp.int32],
):
    # Packed 32 bit key. The key is (iid | tile_id | quant_depth) with
    # depth_bits = 31 - (image_n_bits + tile_n_bits), at least 16 when this kernel
    # is selected. The sign bit stays 0, so cub's signed radix sort orders the keys
    # as plain unsigned ascending over 4 passes instead of 8.
    # quant_depth linearly quantizes the camera depth into depth_bits buckets over
    # the per-image [dmin, dmax] range, which is monotone in depth. Near-coincident
    # gaussians in the same bucket keep gaussian-id order under the stable sort,
    # a perceptually negligible blend-order change (80+ dB vs the 64 bit key).
    # Linear-over-range beats truncating the float mantissa. It spends every bucket
    # inside the scene's actual depth span.
    idx = wp.tid()
    if radii[idx] <= 0:
        return
    n = num_gaussians
    bid = idx // n
    center = xys[idx]

    cur_idx = wp.int32(0)
    if idx > 0:
        cur_idx = cum_tiles_hit[idx - 1]

    dmin = depth_mm[2 * bid]
    drange = depth_mm[2 * bid + 1] - dmin
    maxq = wp.float32((wp.int32(1) << depth_bits) - wp.int32(1))
    depth_q = wp.int32(0)
    if drange > 0.0:
        f = (depths[idx] - dmin) / drange
        depth_q = wp.clamp(
            wp.int32(f * maxq), wp.int32(0), (wp.int32(1) << depth_bits) - wp.int32(1)
        )
    iid_enc = bid << (depth_bits + tile_n_bits)

    # The same ellipse walk as projection's tile count, emitting exactly
    # num_tiles_hit keys per gaussian so the cum offsets stay valid.
    # map_opacities is the raw opacity projection counted with. When the blend uses
    # a compensated opacity (antialiased mode) this stays raw so the emitted key
    # total still matches cum_tiles_hit exactly.
    opac = map_opacities[idx % opac_mod]
    t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
    conic = conics[idx]
    setup = ellipse_setup(
        conic[0], conic[1], conic[2], t, center[0], center[1], tile_bounds_x, tile_bounds_y
    )
    if not setup.valid:
        return
    I_min = ellipse_init_span(setup)
    for u in range(setup.rect_min[0], setup.rect_max[0]):
        rc = ellipse_column(u, setup, I_min)
        mn = wp.int32(rc[0])
        mx = wp.int32(rc[1])
        for v in range(mn, mx):
            if setup.isY:
                tile_id = u * tile_bounds_x + v
            else:
                tile_id = v * tile_bounds_x + u
            isect_ids[cur_idx] = iid_enc | (tile_id << depth_bits) | depth_q
            gaussian_ids[cur_idx] = idx
            cur_idx = cur_idx + 1
        I_min = wp.vec2(rc[2], rc[3])


@wp.kernel
def _map_intersects_64bit(
    xys: wp.array[wp.vec2],
    depths_int: wp.array[wp.int32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    map_opacities: wp.array[wp.float32],
    cum_tiles_hit: wp.array[wp.int32],
    num_gaussians: wp.int32,
    opac_mod: wp.int32,
    tile_n_bits: wp.int32,
    tile_bounds_x: wp.int32,
    tile_bounds_y: wp.int32,
    # outputs
    isect_ids: wp.array[wp.int64],
    gaussian_ids: wp.array[wp.int32],
):
    # 64 bit twin of _map_intersects_32bit for the too-many-tiles case. The key is
    # (iid | tile_id) << 32 | depth_bits, with the positive float depth's raw bits
    # sorting correctly as ints. Identical tile emission, only the key differs.
    idx = wp.tid()
    if radii[idx] <= 0:
        return
    n = num_gaussians
    bid = idx // n
    center = xys[idx]

    cur_idx = wp.int32(0)
    if idx > 0:
        cur_idx = cum_tiles_hit[idx - 1]

    depth_id = wp.int64(depths_int[idx])
    iid_enc = wp.int64(bid) << (wp.int64(32) + wp.int64(tile_n_bits))

    opac = map_opacities[idx % opac_mod]
    t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
    conic = conics[idx]
    setup = ellipse_setup(
        conic[0], conic[1], conic[2], t, center[0], center[1], tile_bounds_x, tile_bounds_y
    )
    if not setup.valid:
        return
    I_min = ellipse_init_span(setup)
    for u in range(setup.rect_min[0], setup.rect_max[0]):
        rc = ellipse_column(u, setup, I_min)
        mn = wp.int32(rc[0])
        mx = wp.int32(rc[1])
        for v in range(mn, mx):
            if setup.isY:
                tile_id = wp.int64(u * tile_bounds_x + v)
            else:
                tile_id = wp.int64(v * tile_bounds_x + u)
            isect_ids[cur_idx] = iid_enc | (tile_id << wp.int64(32)) | depth_id
            gaussian_ids[cur_idx] = idx
            cur_idx = cur_idx + 1
        I_min = wp.vec2(rc[2], rc[3])


@wp.kernel
def _tile_bin_edges_32bit(
    num_intersects: wp.int32,
    isect_ids_sorted: wp.array[wp.int32],
    num_tiles: wp.int32,
    tile_n_bits: wp.int32,
    depth_bits: wp.int32,
    # output
    tile_bins: wp.array[wp.vec2i],
):
    # Per (image, tile) bin edges into a [B*num_tiles] array. The flat bin index is
    # iid*num_tiles + tile_id, decoded from the key field above the depth bits.
    idx = wp.tid()
    if idx >= num_intersects:
        return
    mask = (wp.int32(1) << tile_n_bits) - wp.int32(1)
    key = isect_ids_sorted[idx] >> depth_bits
    cur_bin = (key >> tile_n_bits) * num_tiles + (key & mask)
    if idx == 0:
        tile_bins[cur_bin][0] = 0
        return
    if idx == num_intersects - 1:
        tile_bins[cur_bin][1] = num_intersects
    keyp = isect_ids_sorted[idx - 1] >> depth_bits
    prev_bin = (keyp >> tile_n_bits) * num_tiles + (keyp & mask)
    if prev_bin != cur_bin:
        tile_bins[prev_bin][1] = idx
        tile_bins[cur_bin][0] = idx


@wp.kernel
def _tile_bin_edges_64bit(
    num_intersects: wp.int32,
    isect_ids_sorted: wp.array[wp.int64],
    num_tiles: wp.int32,
    tile_n_bits: wp.int32,
    # output
    tile_bins: wp.array[wp.vec2i],
):
    # 64 bit twin of _tile_bin_edges_32bit. The (iid | tile) field sits above bit 32.
    idx = wp.tid()
    if idx >= num_intersects:
        return
    mask = (wp.int64(1) << wp.int64(tile_n_bits)) - wp.int64(1)
    key = isect_ids_sorted[idx] >> wp.int64(32)
    cur_bin = wp.int32(key >> wp.int64(tile_n_bits)) * num_tiles + wp.int32(key & mask)
    if idx == 0:
        tile_bins[cur_bin][0] = 0
        return
    if idx == num_intersects - 1:
        tile_bins[cur_bin][1] = num_intersects
    keyp = isect_ids_sorted[idx - 1] >> wp.int64(32)
    prev_bin = wp.int32(keyp >> wp.int64(tile_n_bits)) * num_tiles + wp.int32(keyp & mask)
    if prev_bin != cur_bin:
        tile_bins[prev_bin][1] = idx
        tile_bins[cur_bin][0] = idx


def _use_32bit_keys(depth_bits: int) -> bool:
    # The packed key needs at least 16 depth bits below the image and tile ids.
    # Tests patch this to force the 64 bit path on small scenes.
    return depth_bits >= 16


def _sort_and_bin(
    device: wp.Device | None,
    xys: wp.array,
    depths: wp.array,
    radii: wp.array,
    conics: wp.array,
    map_opacities: wp.array,
    cum_tiles_hit: wp.array,
    n: int,
    B: int,
    tile_bounds_x: int,
    tile_bounds_y: int,
) -> tuple[wp.array, wp.array, int]:
    """Emit sorted intersection keys and tile bins for one batched launch.

    Used by the forward blend and by the backward pass, which recomputes the
    identical sort from the saved cum_tiles_hit. The sort is deterministic, so it
    reproduces the forward gaussian_ids and tile_bins and the saved final_idx stays
    valid. Returns (gaussian_ids, tile_bins, num_intersects), where gaussian_ids is
    the full scratch buffer whose valid prefix is [0, num_intersects).
    """
    num_tiles = tile_bounds_x * tile_bounds_y
    # bits to index [0, num_tiles) and [0, B), the tile and image id fields of the sort key
    tile_n_bits = (num_tiles - 1).bit_length()
    image_n_bits = (B - 1).bit_length()
    upper_bits = image_n_bits + tile_n_bits
    depth_bits = 31 - upper_bits
    packed = _use_32bit_keys(depth_bits)
    if not packed and upper_bits > 32:
        raise ValueError(
            f"batched intersection key overflow: image_n_bits({image_n_bits}) + "
            f"tile_n_bits({tile_n_bits}) = {upper_bits} > 32 "
            f"(batch B={B}, n_tiles={num_tiles}). Reduce batch size or resolution."
        )
    total = B * n
    bins_len = B * num_tiles
    opac_mod = map_opacities.shape[0]

    # The one legitimate device to host sync, the total intersection count. The
    # copy is enqueued first and the wait happens below, so the tile_bins memset
    # and the depth range pre-pass execute inside the sync bubble while the host
    # waits for the scan result.
    pending = _read_count_begin(cum_tiles_hit, total - 1, device)

    isect_dtype = wp.int32 if packed else wp.int64
    scratch = _get_scratch(device, (B, n, num_tiles), 2, bins_len, isect_dtype)
    # tile_bins is allocated at exactly bins_len for this signature.
    tile_bins = scratch["tile_bins"]
    tile_bins.zero_()
    if packed:
        # Device-side per-image [dmin, dmax] for the depth quantization, no host
        # sync. Count-independent, so it launches before the readback wait.
        depth_mm = scratch["depth_mm"]
        _cached_launch(_seed_minmax, B, [depth_mm], device)
        _cached_launch(
            _depth_minmax,
            (total + int(_MINMAX_CHUNK) - 1) // int(_MINMAX_CHUNK),
            [depths, radii, total, n, depth_mm],
            device,
        )

    num_intersects = _read_count_end(pending)
    # Grow the sort buffers to the frame's count. Nothing above needs them.
    scratch = _get_scratch(
        device, (B, n, num_tiles), max(2 * num_intersects, 2), bins_len, isect_dtype
    )

    if num_intersects == 0:
        return scratch["gaussian_ids"], tile_bins, 0

    # The kernels and the sort take explicit counts, so the full-capacity scratch
    # arrays are passed without per-frame slicing. Every access stays inside
    # [0, 2*num_intersects) and the stable shapes keep the recorded launches from
    # repacking their array arguments each frame.
    isect_ids = scratch["isect_ids"]
    gaussian_ids = scratch["gaussian_ids"]
    if packed:
        _cached_launch(
            _map_intersects_32bit,
            total,
            [
                xys,
                depths,
                radii,
                conics,
                map_opacities,
                cum_tiles_hit,
                depth_mm,
                n,
                opac_mod,
                tile_n_bits,
                depth_bits,
                tile_bounds_x,
                tile_bounds_y,
                isect_ids,
                gaussian_ids,
            ],
            device,
        )
        wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, num_intersects)
        _cached_launch(
            _tile_bin_edges_32bit,
            num_intersects,
            [num_intersects, isect_ids, num_tiles, tile_n_bits, depth_bits, tile_bins],
            device,
        )
    else:
        _cached_launch(
            _map_intersects_64bit,
            total,
            [
                xys,
                depths.view(wp.int32),
                radii,
                conics,
                map_opacities,
                cum_tiles_hit,
                n,
                opac_mod,
                tile_n_bits,
                tile_bounds_x,
                tile_bounds_y,
                isect_ids,
                gaussian_ids,
            ],
            device,
        )
        wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, num_intersects)
        _cached_launch(
            _tile_bin_edges_64bit,
            num_intersects,
            [num_intersects, isect_ids, num_tiles, tile_n_bits, tile_bins],
            device,
        )
    return gaussian_ids, tile_bins, num_intersects
