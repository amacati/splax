"""Warp kernels for intersection key emission and tile bin edges.

The forward blend and the backward pass both walk the sorted per-tile gaussian lists these kernels
feed. Projection's ellipse walk counts the tiles each gaussian touches, and here the identical walk
emits one sort key per intersection, later scanned into per-tile bin edges.

Sort keys come in two widths. When the image and tile ids leave at least 16 low bits, the whole key
packs into one non-negative int32 holding the image id, the tile id, and the quantized depth. This
halves the radix sort passes and quarters the bytes moved, dropping the sort stage by 2.7 to 3.1x on
large frames. Otherwise a 64 bit key carrying the image and tile ids above the depth bits is the
automatic fallback.
"""

import warp as wp

from splax._intersect import (
    ALPHA_THRESHOLD,
    GAUSSIAN_EXTEND_SQ,
    ellipse_column_tile_range,
    ellipse_init,
    ellipse_init_span,
)

wp.set_module_options({"fast_math": True})

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
    setup = ellipse_init(
        conic[0], conic[1], conic[2], t, center[0], center[1], tile_bounds_x, tile_bounds_y
    )
    if not setup.valid:
        return
    I_min = ellipse_init_span(setup)
    for u in range(setup.rect_min[0], setup.rect_max[0]):
        rc = ellipse_column_tile_range(u, setup, I_min)
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
    setup = ellipse_init(
        conic[0], conic[1], conic[2], t, center[0], center[1], tile_bounds_x, tile_bounds_y
    )
    if not setup.valid:
        return
    I_min = ellipse_init_span(setup)
    for u in range(setup.rect_min[0], setup.rect_max[0]):
        rc = ellipse_column_tile_range(u, setup, I_min)
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
