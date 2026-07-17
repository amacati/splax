"""Warp rasterization kernels and their JAX FFI callables.

This module holds the GPU side of the rasterization stage: the tiled front-to-back blend kernels
(plain and depth-augmented), their backward twins, the shared sort and bin build, and the opt-in
post-sync CUDA graph path. The host-side ``_*_launch`` functions are wrapped into JAX FFI callables
that the API layer in ``splax._rasterize`` composes with ``jax.custom_vjp``.

Batching is native. Under jax.vmap the callable launches a single grid over the whole batch. The
image index is decoded from the block rank, packed into the sort key, and used to offset per-image
bin edges, outputs, and backgrounds. There is no host loop. Because of the host readback and
data-dependent scratch, the forward callable is not CUDA-graph capturable, so graph_mode=NONE.
"""

from __future__ import annotations

import math
import os
from typing import cast

import warp as wp
from warp.jax_experimental.ffi import JaxCallableGraphMode, jax_callable

from splax._batching import nested_vmap
from splax._intersect import (
    _MINMAX_CHUNK,
    BLOCK_SIZE,
    BLOCK_WIDTH,
    _bits_for_count,
    _cached_launch,
    _depth_minmax,
    _get_scratch,
    _graph_cache,
    _map_intersects_32bit,
    _read_count,
    _seed_minmax,
    _sort_and_bin,
    _tile_bin_edges_32bit_dev,
    _use_32bit_keys,
)

# Compile with approximate transcendentals and fp contraction, matching the CUDA reference's
# -use_fast_math build flag.
wp.set_module_options({"fast_math": True})

# Post-sync CUDA graph capture, opt-in. Captures the whole
# post-readback sequence (sentinel fill, depth minmax, map, sort, bin, blend) as
# a cached CUDA graph and replays it, collapsing ~11 device launches into one
# replay. Recovers the per-frame launch overhead on small and mid forward
# renders (+5.6 to +21.6 percent end to end, bit-identical output). Above the
# count threshold the bucket pad tax on the dominant sort eats the launch win,
# so large frames fall back to plain launches. Default off because graph capture
# inside the callback corrupts the CUDA context when the process concurrently
# drives foreign CUDA graph or stream work. It is safe under the jitted,
# splax-only production path. Packed-key forward path only.
#   SPLAX_POSTSYNC_GRAPHS=1   enable
#   SPLAX_GRAPH_THRESHOLD=N   max num_intersects to use a graph (default 2000000)
_POSTSYNC_GRAPHS = os.environ.get("SPLAX_POSTSYNC_GRAPHS", "0") == "1"
_GRAPH_THRESHOLD = int(os.environ.get("SPLAX_GRAPH_THRESHOLD", "2000000"))
_GRAPH_BUCKET_STEP = 1.05  # geometric bucket granularity, ~5 percent pad worst case


def _bucket_count(ni: int) -> int:
    """Round num_intersects up to the next ~5 percent geometric step."""
    if ni <= 1:
        return max(ni, 1)
    k = math.ceil(math.log(ni) / math.log(_GRAPH_BUCKET_STEP))
    return max(ni, int(math.ceil(_GRAPH_BUCKET_STEP**k)))


# One staged gaussian record: xy (2), opacity (1), conic (3). Packing every field
# the alpha test needs into one vector lets a thread stage its gaussian with a
# single shared write, and the whole record arrives with one shared read per
# tested gaussian in the blend loop.
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
# The depth forward stages depth inside the geometry record. A 7-float record also
# avoids the shared-memory access pattern that made the previous 6-float-plus-vec4
# tile split run 2x slower, see the kernel comment on _rasterize_fwd_depth.
_vec7 = wp.types.vector(length=7, dtype=wp.float32)


@wp.kernel
def _rasterize_fwd(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
):
    # Cooperative shared-memory blend. One 256-thread block per (image, tile)
    # stages 256 gaussian records per batch into shared memory, each thread
    # gathering exactly one gaussian with a single masked shared write per tile.
    # image_id = block // num_tiles decodes the batch element. Outputs are the
    # collapsed batched buffers (B*H, W), written at row image_id*H + i. The
    # gathered gaussian ids are flat (b*N + gid). xys and conics are batched, so
    # the flat id indexes them directly, while broadcast size-N colors and
    # opacities are shifted back via a per-thread modulo at gather time.
    tile_g, tr = wp.tid()  # launch_tiled: block index and thread rank
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles

    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li  # row (y)
    j = tile_x * BLOCK_WIDTH + lj  # col (x)

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    # Threads mapping outside the image stay live for the collective loads and
    # the block vote but are marked done and never write an output pixel.
    inside = (i < img_h) and (j < img_w)
    done = wp.bool(not inside)

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE

    T = wp.float32(1.0)
    cur_idx = wp.int32(0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)

    # Colors are staged in their own tile so a rejected gaussian only ever touches
    # its geometry record. The blend reads the color slot on acceptance alone.
    geo_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=_vec6, storage="shared")
    color_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.vec3, storage="shared")
    done_tile = wp.tile_zeros(shape=1, dtype=wp.int32, storage="shared")
    counted = wp.bool(False)

    for b in range(num_batches):
        # Whole-tile early-out vote, break once every thread in the block is done.
        # Each thread bumps the shared counter once, when it first turns done, so
        # the counter never needs resetting. The scatter's barrier doubles as the
        # guard between the previous batch's reads and this batch's staging writes.
        wp.tile_scatter_add(done_tile, 0, 1, done and not counted)
        counted = done
        if done_tile[0] >= BLOCK_SIZE:
            break

        # Per-thread gather of one gaussian record. Broadcast size-N attributes
        # must be read at the local gid, batched size B*N ones at the flat id.
        # Modulo by the array's own leading dim does both. The tail lanes of the
        # last batch clamp to the final intersection, staging a duplicate record
        # the blend loop never reads because it stops at batch_size.
        batch_start = range_start + b * BLOCK_SIZE
        src = wp.min(batch_start + tr, range_end - 1)
        g = gaussian_ids_sorted[src]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        wp.tile_scatter_masked(
            geo_tile, tr, _vec6(xy[0], xy[1], opac, conic[0], conic[1], conic[2]), True
        )
        wp.tile_scatter_masked(color_tile, tr, colors[g % color_mod], True)

        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                s = geo_tile[t]
                dx = s[0] - px
                dy = s[1] - py
                sigma = 0.5 * (s[3] * dx * dx + s[5] * dy * dy) + s[4] * dx * dy
                alpha = wp.min(0.999, s[2] * wp.exp(-sigma))
                if sigma < 0.0 or alpha < 1.0 / 255.0:
                    continue
                next_T = T * (1.0 - alpha)
                if next_T <= 1e-4:
                    done = wp.bool(True)
                    break
                vis = alpha * T
                pix_out = pix_out + color_tile[t] * vis
                T = next_T
                cur_idx = batch_start + t

    if inside:
        bg = background[wp.where(sel_bg, image_id, 0)]
        row = image_id * img_h + i
        final_Ts[row, j] = T
        final_idx[row, j] = cur_idx
        out_img[row, j] = pix_out + T * bg


# Depth-augmented forward. The expected-depth channel
# D(p) = sum_i w_i d_i with the alpha-blend weights w_i, for sparse-point depth
# regularization. A separate kernel so the default render never pays for the
# extra accumulator and load. Blend math, early-exit vote, staging, and batched
# indexing are identical to _rasterize_fwd, with depth packed into the geometry
# record. The packing matters: staging depth in a second vec4 tile alongside a
# vec6 geometry tile ran the whole kernel 2x slower than the plain color blend,
# while the 7-float record restores parity with it (bit-identical output).
# Background depth is 0, so the depth channel has no T*bg term.
@wp.kernel
def _rasterize_fwd_depth(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    depths: wp.array[wp.float32],
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
    out_depth: wp.array2d[wp.float32],
):
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles

    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    inside = (i < img_h) and (j < img_w)
    done = wp.bool(not inside)

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE

    T = wp.float32(1.0)
    cur_idx = wp.int32(0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)
    depth_out = wp.float32(0.0)

    geo_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=_vec7, storage="shared")
    color_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.vec3, storage="shared")
    done_tile = wp.tile_zeros(shape=1, dtype=wp.int32, storage="shared")
    counted = wp.bool(False)

    for b in range(num_batches):
        wp.tile_scatter_add(done_tile, 0, 1, done and not counted)
        counted = done
        if done_tile[0] >= BLOCK_SIZE:
            break

        batch_start = range_start + b * BLOCK_SIZE
        src = wp.min(batch_start + tr, range_end - 1)
        g = gaussian_ids_sorted[src]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        wp.tile_scatter_masked(
            geo_tile, tr, _vec7(xy[0], xy[1], opac, conic[0], conic[1], conic[2], depths[g]), True
        )
        wp.tile_scatter_masked(color_tile, tr, colors[g % color_mod], True)

        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                s = geo_tile[t]
                dx = s[0] - px
                dy = s[1] - py
                sigma = 0.5 * (s[3] * dx * dx + s[5] * dy * dy) + s[4] * dx * dy
                alpha = wp.min(0.999, s[2] * wp.exp(-sigma))
                if sigma < 0.0 or alpha < 1.0 / 255.0:
                    continue
                next_T = T * (1.0 - alpha)
                if next_T <= 1e-4:
                    done = wp.bool(True)
                    break
                vis = alpha * T
                pix_out = pix_out + color_tile[t] * vis
                depth_out = depth_out + s[6] * vis
                T = next_T
                cur_idx = batch_start + t

    if inside:
        bg = background[wp.where(sel_bg, image_id, 0)]
        row = image_id * img_h + i
        final_Ts[row, j] = T
        final_idx[row, j] = cur_idx
        out_img[row, j] = pix_out + T * bg
        out_depth[row, j] = depth_out


def _blend_setup(
    colors: wp.array,
    xys: wp.array,
    depths: wp.array,
    radii: wp.array,
    conics: wp.array,
    map_opacities: wp.array,
    cum_tiles_hit: wp.array,
    n: int,
    B_geom: int,
    img_h: int,
    img_w: int,
    num_intersects: int | None = None,
) -> tuple[wp.array, wp.array, int, int, int]:
    """Tile geometry plus the shared sort and bin build.

    B_geom is the geometry batch, how many distinct renders the sort covers.
    num_intersects may be passed in when the caller already read it back.
    Returns (gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles).
    """
    bw = int(BLOCK_WIDTH)
    tile_bounds_x = (img_w + bw - 1) // bw
    tile_bounds_y = (img_h + bw - 1) // bw
    gaussian_ids, tile_bins, num_intersects = _sort_and_bin(
        colors.device,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B_geom,
        tile_bounds_x,
        tile_bounds_y,
        num_intersects,
    )
    return (gaussian_ids, tile_bins, num_intersects, tile_bounds_x, tile_bounds_x * tile_bounds_y)


def _forward_graph(
    colors: wp.array,
    opacities: wp.array,
    map_opacities: wp.array,
    background: wp.array,
    xys: wp.array,
    depths: wp.array,
    radii: wp.array,
    conics: wp.array,
    cum_tiles_hit: wp.array,
    n: int,
    B: int,
    img_h: int,
    img_w: int,
    tile_bounds_x: int,
    tile_bounds_y: int,
    sel_bg: bool,
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
) -> tuple[bool, int | None]:
    """Captured-graph forward path.

    Returns (handled, num_intersects). handled is True if a graph replayed the
    frame. When False the caller runs the plain path, reusing the returned count
    so the host readback happens exactly once.

    Reads num_intersects, buckets it, and either replays a cached CUDA graph or
    captures one covering the whole post-sync sequence. Byte-identical to the
    plain packed path. The sort runs over the padded bucket with 0x7FFFFFFF
    sentinels sorted to the tail, and the bin kernel guards on the device-side
    real count re-read at replay, so no sentinel is ever binned.

    Falls back unless the packed key layout applies and 0 < num_intersects <
    _GRAPH_THRESHOLD.
    """
    num_tiles = tile_bounds_x * tile_bounds_y
    device = colors.device
    assert device is not None  # colors is always a live device array here
    # Never nest a capture inside an existing one. If the callback runs while the
    # stream is already being captured, our capture_begin would conflict and
    # corrupt the context. Fall back to plain launches and let the caller do its
    # own readback, a host sync is illegal during capture.
    if device.is_capturing:
        return False, None
    tile_n_bits = _bits_for_count(num_tiles)
    image_n_bits = _bits_for_count(B)
    depth_bits = 31 - (image_n_bits + tile_n_bits)
    packed = _use_32bit_keys(depth_bits)
    total = B * n
    # The single host readback, reused by the caller's plain path on a fallback.
    num_intersects = _read_count(cum_tiles_hit, total - 1, device)
    if not packed or num_intersects <= 0 or num_intersects >= _GRAPH_THRESHOLD:
        return False, num_intersects

    bucket = _bucket_count(num_intersects)
    bins_len = B * num_tiles
    scratch = _get_scratch(device, (B, n, num_tiles), 2 * bucket, bins_len, wp.int32)
    isect_ids = scratch["isect_ids"]
    gaussian_ids = scratch["gaussian_ids"]
    tile_bins = scratch["tile_bins"][:bins_len]
    depth_mm = scratch["depth_mm"]
    gen = scratch["gen"]

    def run() -> None:
        tile_bins.zero_()
        # Sentinel-fill the sort range. The map kernel overwrites the real
        # [0, count) prefix (its write count is data-dependent via cum_tiles_hit,
        # re-read at replay), leaving the tail as 0x7FFFFFFF, which sorts last.
        isect_ids[:bucket].fill_(0x7FFFFFFF)
        wp.launch(_seed_minmax, dim=B, inputs=[depth_mm], device=device)
        wp.launch(
            _depth_minmax,
            dim=(total + int(_MINMAX_CHUNK) - 1) // int(_MINMAX_CHUNK),
            inputs=[depths, radii, total, n, depth_mm],
            device=device,
        )
        wp.launch(
            _map_intersects_32bit,
            dim=total,
            inputs=[
                xys,
                depths,
                radii,
                conics,
                map_opacities,
                cum_tiles_hit,
                depth_mm,
                n,
                map_opacities.shape[0],
                tile_n_bits,
                depth_bits,
                tile_bounds_x,
                tile_bounds_y,
            ],
            outputs=[isect_ids[: 2 * bucket], gaussian_ids[: 2 * bucket]],
            device=device,
        )
        wp.utils.radix_sort_pairs(isect_ids[: 2 * bucket], gaussian_ids[: 2 * bucket], bucket)
        wp.launch(
            _tile_bin_edges_32bit_dev,
            dim=bucket,
            inputs=[
                cum_tiles_hit,
                total - 1,
                isect_ids[: 2 * bucket],
                num_tiles,
                tile_n_bits,
                depth_bits,
            ],
            outputs=[tile_bins],
            device=device,
        )
        wp.launch_tiled(
            _rasterize_fwd,
            dim=[B * num_tiles],
            inputs=[
                img_h,
                img_w,
                tile_bounds_x,
                num_tiles,
                colors.shape[0],
                opacities.shape[0],
                sel_bg,
                gaussian_ids[: 2 * bucket],
                tile_bins,
                xys,
                conics,
                colors,
                opacities,
                background,
            ],
            outputs=[final_Ts, final_idx, out_img],
            block_dim=int(BLOCK_SIZE),
            device=device,
        )

    # A graph records buffer addresses, so the key carries every operand pointer
    # plus the scratch generation. .ptr is already an int for live device arrays.
    key = (
        str(device),
        B,
        n,
        num_tiles,
        img_h,
        img_w,
        colors.shape[0],
        opacities.shape[0],
        map_opacities.shape[0],
        sel_bg,
        bucket,
        gen,
        xys.ptr,
        depths.ptr,
        radii.ptr,
        conics.ptr,
        cum_tiles_hit.ptr,
        colors.ptr,
        opacities.ptr,
        map_opacities.ptr,
        background.ptr,
        cast("wp.array", out_img).ptr,
        cast("wp.array", final_Ts).ptr,
        cast("wp.array", final_idx).ptr,
    )
    graph = _graph_cache.get(key)
    if graph is None:
        run()  # warm run loads modules and allocates cub temp before capture
        wp.capture_begin(device, force_module_load=False)
        try:
            run()
        finally:
            graph = wp.capture_end(device)
        _graph_cache[key] = graph
    wp.capture_launch(graph)
    return True, num_intersects


def _rasterize_launch(
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    map_opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    cum_tiles_hit: wp.array[wp.int32],
    num_gaussians: int,
    img_h: int,
    img_w: int,
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
) -> None:
    # B is recovered from an output shape, always full batch under expand_dims
    # (out_img collapses to (B*H, W)). N is static because vmap hides the batch
    # axis from this wrapper.
    n = num_gaussians
    B = out_img.shape[0] // img_h
    sel_bg = background.shape[0] > 1

    # Post-sync CUDA graph fast path, opt-in. Replays the whole
    # map/sort/bin/blend as one cached graph. On a fallback the readback is
    # reused so the host sync happens exactly once.
    ni_pre = None
    if _POSTSYNC_GRAPHS:
        bw = int(BLOCK_WIDTH)
        tbx = (img_w + bw - 1) // bw
        tby = (img_h + bw - 1) // bw
        handled, ni_pre = _forward_graph(
            colors,
            opacities,
            map_opacities,
            background,
            xys,
            depths,
            radii,
            conics,
            cum_tiles_hit,
            n,
            B,
            img_h,
            img_w,
            tbx,
            tby,
            sel_bg,
            final_Ts,
            final_idx,
            out_img,
        )
        if handled:
            return

    # Key emission uses map_opacities, the raw opacity projection counted with.
    # The blend uses opacities, compensated in antialiased mode. When not
    # antialiased the caller passes the same array for both.
    gaussian_ids, tile_bins, _num_isect, tile_bounds_x, num_tiles = _blend_setup(
        colors, xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B, img_h, img_w, ni_pre
    )

    _cached_launch(
        _rasterize_fwd,
        B * num_tiles,
        [
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            colors.shape[0],
            opacities.shape[0],
            sel_bg,
            gaussian_ids,
            tile_bins,
            xys,
            conics,
            colors,
            opacities,
            background,
            final_Ts,
            final_idx,
            out_img,
        ],
        colors.device,
        block_dim=int(BLOCK_SIZE),
    )


_rasterize_ffi = nested_vmap(
    jax_callable(
        _rasterize_launch,
        num_outputs=3,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=9,
    name="rasterize",
)


def _rasterize_depth_launch(
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    map_opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    cum_tiles_hit: wp.array[wp.int32],
    num_gaussians: int,
    img_h: int,
    img_w: int,
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
    out_depth: wp.array2d[wp.float32],
) -> None:
    # Depth-augmented twin of _rasterize_launch. Shares the exact sort and bin, so
    # the blend order matches the plain path bit for bit.
    n = num_gaussians
    B = out_img.shape[0] // img_h
    sel_bg = background.shape[0] > 1

    gaussian_ids, tile_bins, _num_isect, tile_bounds_x, num_tiles = _blend_setup(
        colors, xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B, img_h, img_w
    )

    _cached_launch(
        _rasterize_fwd_depth,
        B * num_tiles,
        [
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            colors.shape[0],
            opacities.shape[0],
            sel_bg,
            gaussian_ids,
            tile_bins,
            xys,
            conics,
            colors,
            opacities,
            background,
            depths,
            final_Ts,
            final_idx,
            out_img,
            out_depth,
        ],
        colors.device,
        block_dim=int(BLOCK_SIZE),
    )


_rasterize_depth_ffi = nested_vmap(
    jax_callable(
        _rasterize_depth_launch,
        num_outputs=4,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=9,
    name="rasterize_depth",
)


# Backward pass. A staged lockstep walk mirroring the forward blend. All 256
# threads of a tile block walk the sorted range back to front in shared-staged
# batches, starting at the block maximum of the pixels' final_idx. Each pixel
# reconstructs T by dividing out (1 - alpha) and accumulates parameter gradients
# with per-lane atomics, guarded per pixel so only indices at or below its own
# final_idx contribute, with the same sigma and alpha culling as the forward.
#
# The sort and bin structures are not saved from the forward. They are recomputed
# from the saved cum_tiles_hit via the shared _sort_and_bin. The sort is
# deterministic, so it reproduces the forward order and the saved final_Ts and
# final_idx line up.
#
# The alpha cotangent is zero because rasterize returns only the image.


@wp.kernel
def _rasterize_bwd_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.bool,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    vout_rows: wp.int32,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    # outputs, atomically accumulated per gaussian
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
):
    # One block per (output image, tile). image_id decodes the output image. The
    # geometry has its own batch B_geom, either equal to B_out (sel_geom True) or
    # 1 (sel_geom False, a single shared render differentiated against B target
    # images). Batched geometry writes grads at the flat id and broadcast
    # geometry gets one slot per output image. JAX reduces broadcast inputs over
    # the batch axis.
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = wp.where(sel_geom, image_id, 0)
    og_base = wp.where(sel_geom, 0, image_id * num_gaussians)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    # Threads mapping outside the image stay live for the collective staging but
    # never pass the validity guard, their bin_final sits below the range.
    inside = (i < img_h) and (j < img_w)
    bin_final = range_start - 1
    T = wp.float32(1.0)
    t_final = wp.float32(1.0)
    v_out = wp.vec3(0.0, 0.0, 0.0)
    bg = wp.vec3(0.0, 0.0, 0.0)
    if inside:
        frow = geom_image * img_h + i  # final_Ts and final_idx are geometry outputs
        bin_final = final_idx[frow, j]
        t_final = final_Ts[frow, j]
        T = t_final
        # The image cotangent arrives batched (B_out*H rows) for a view-dependent
        # loss but broadcast (H rows) for a view-independent one. Modulo by its
        # own row count reads the right row either way.
        v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
        bg = background[wp.where(sel_bg, image_id, 0)]
    buffer = wp.vec3(0.0, 0.0, 0.0)

    # Gaussians behind every pixel's last contributor never matter, so the walk
    # starts at the block maximum of final_idx instead of range_end.
    start_idx = wp.tile_max(wp.tile(bin_final))[0]
    num_batches = (start_idx - range_start + BLOCK_SIZE) // BLOCK_SIZE

    geo_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=_vec6, storage="shared")
    color_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.vec3, storage="shared")
    id_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.int32, storage="shared")
    sync_tile = wp.tile_empty(shape=1, dtype=wp.int32, storage="shared")

    for b in range(num_batches):
        # The scatters place their barrier after the write, so an empty scatter
        # guards the previous batch's shared reads against this batch's staging.
        wp.tile_scatter_add(sync_tile, 0, 0, False)

        # Per-thread gather of one gaussian record, back to front. Tail lanes
        # clamp to range_start, staging a duplicate record the guarded loop
        # never reads. Broadcast size-N attributes are read at the local gid,
        # batched size B*N ones at the flat id, via modulo as in the forward.
        batch_end = start_idx - b * BLOCK_SIZE
        src = wp.max(batch_end - tr, range_start)
        g = gaussian_ids_sorted[src]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        wp.tile_scatter_masked(
            geo_tile, tr, _vec6(xy[0], xy[1], opac, conic[0], conic[1], conic[2]), True
        )
        wp.tile_scatter_masked(color_tile, tr, colors[g % color_mod], True)
        wp.tile_scatter_masked(id_tile, tr, g, True)

        # Pixels whose last contributor lies below this batch skip it whole.
        batch_size = wp.min(BLOCK_SIZE, batch_end - range_start + 1)
        if batch_end - batch_size + 1 <= bin_final:
            for t in range(batch_size):
                idx = batch_end - t
                if idx > bin_final:
                    continue
                s = geo_tile[t]
                dx = s[0] - px
                dy = s[1] - py
                sigma = 0.5 * (s[3] * dx * dx + s[5] * dy * dy) + s[4] * dx * dy
                if sigma < 0.0:
                    continue
                vis = wp.exp(-sigma)
                alpha = wp.min(0.999, s[2] * vis)
                if alpha < 1.0 / 255.0:
                    continue

                ra = 1.0 / (1.0 - alpha)
                T = T * ra
                fac = alpha * T
                color = color_tile[t]
                og = og_base + id_tile[t]

                wp.atomic_add(v_colors, og, v_out * fac)

                v_alpha = float(0.0)
                v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
                v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
                v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
                v_alpha += -t_final * ra * bg[0] * v_out[0]
                v_alpha += -t_final * ra * bg[1] * v_out[1]
                v_alpha += -t_final * ra * bg[2] * v_out[2]

                buffer = buffer + color * fac

                # Where the alpha clamp is active alpha is constant, so the
                # sigma and opacity paths carry no gradient.
                if s[2] * vis <= 0.999:
                    v_sigma = -s[2] * vis * v_alpha
                    wp.atomic_add(
                        v_conic,
                        og,
                        wp.vec3(
                            0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy
                        ),
                    )
                    wp.atomic_add(
                        v_xy,
                        og,
                        wp.vec2(
                            v_sigma * (s[3] * dx + s[4] * dy), v_sigma * (s[4] * dx + s[5] * dy)
                        ),
                    )
                    wp.atomic_add(v_opacity, og, vis * v_alpha)


# Depth-augmented backward. The depth channel is handled exactly like
# a color channel. It contributes to v_alpha, hence to v_sigma and the conic, xy,
# and opacity grads, and produces a per-gaussian depth cotangent that flows
# through project's backward to the geometry and camera pose. Walk, staging, and
# color-grad math are identical to _rasterize_bwd_kernel, with depth packed next
# to color in the staged records.
@wp.kernel
def _rasterize_bwd_depth_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.bool,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    vout_rows: wp.int32,
    vdepth_rows: wp.int32,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    depths: wp.array[wp.float32],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    v_out_depth: wp.array2d[wp.float32],
    # outputs, atomically accumulated per gaussian
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_depths: wp.array[wp.float32],
):
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = wp.where(sel_geom, image_id, 0)
    og_base = wp.where(sel_geom, 0, image_id * num_gaussians)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    inside = (i < img_h) and (j < img_w)
    bin_final = range_start - 1
    T = wp.float32(1.0)
    t_final = wp.float32(1.0)
    v_out = wp.vec3(0.0, 0.0, 0.0)
    v_outd = wp.float32(0.0)
    bg = wp.vec3(0.0, 0.0, 0.0)
    if inside:
        frow = geom_image * img_h + i
        bin_final = final_idx[frow, j]
        t_final = final_Ts[frow, j]
        T = t_final
        v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
        v_outd = v_out_depth[(image_id * img_h + i) % vdepth_rows, j]
        bg = background[wp.where(sel_bg, image_id, 0)]
    buffer = wp.vec3(0.0, 0.0, 0.0)
    dbuffer = wp.float32(0.0)

    start_idx = wp.tile_max(wp.tile(bin_final))[0]
    num_batches = (start_idx - range_start + BLOCK_SIZE) // BLOCK_SIZE

    geo_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=_vec6, storage="shared")
    cd_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.vec4, storage="shared")
    id_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.int32, storage="shared")
    sync_tile = wp.tile_empty(shape=1, dtype=wp.int32, storage="shared")

    for b in range(num_batches):
        wp.tile_scatter_add(sync_tile, 0, 0, False)

        batch_end = start_idx - b * BLOCK_SIZE
        src = wp.max(batch_end - tr, range_start)
        g = gaussian_ids_sorted[src]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        color = colors[g % color_mod]
        wp.tile_scatter_masked(
            geo_tile, tr, _vec6(xy[0], xy[1], opac, conic[0], conic[1], conic[2]), True
        )
        wp.tile_scatter_masked(cd_tile, tr, wp.vec4(color[0], color[1], color[2], depths[g]), True)
        wp.tile_scatter_masked(id_tile, tr, g, True)

        batch_size = wp.min(BLOCK_SIZE, batch_end - range_start + 1)
        if batch_end - batch_size + 1 <= bin_final:
            for t in range(batch_size):
                idx = batch_end - t
                if idx > bin_final:
                    continue
                s = geo_tile[t]
                dx = s[0] - px
                dy = s[1] - py
                sigma = 0.5 * (s[3] * dx * dx + s[5] * dy * dy) + s[4] * dx * dy
                if sigma < 0.0:
                    continue
                vis = wp.exp(-sigma)
                alpha = wp.min(0.999, s[2] * vis)
                if alpha < 1.0 / 255.0:
                    continue

                ra = 1.0 / (1.0 - alpha)
                T = T * ra
                fac = alpha * T
                cd = cd_tile[t]
                og = og_base + id_tile[t]

                wp.atomic_add(v_colors, og, v_out * fac)
                wp.atomic_add(v_depths, og, v_outd * fac)

                v_alpha = float(0.0)
                v_alpha += (cd[0] * T - buffer[0] * ra) * v_out[0]
                v_alpha += (cd[1] * T - buffer[1] * ra) * v_out[1]
                v_alpha += (cd[2] * T - buffer[2] * ra) * v_out[2]
                # depth channel, background depth is 0 so there is no t_final*bg term
                v_alpha += (cd[3] * T - dbuffer * ra) * v_outd
                v_alpha += -t_final * ra * bg[0] * v_out[0]
                v_alpha += -t_final * ra * bg[1] * v_out[1]
                v_alpha += -t_final * ra * bg[2] * v_out[2]

                buffer = buffer + wp.vec3(cd[0], cd[1], cd[2]) * fac
                dbuffer = dbuffer + cd[3] * fac

                # Where the alpha clamp is active alpha is constant, so the
                # sigma and opacity paths carry no gradient.
                if s[2] * vis <= 0.999:
                    v_sigma = -s[2] * vis * v_alpha
                    wp.atomic_add(
                        v_conic,
                        og,
                        wp.vec3(
                            0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy
                        ),
                    )
                    wp.atomic_add(
                        v_xy,
                        og,
                        wp.vec2(
                            v_sigma * (s[3] * dx + s[4] * dy), v_sigma * (s[4] * dx + s[5] * dy)
                        ),
                    )
                    wp.atomic_add(v_opacity, og, vis * v_alpha)


# Warp-aggregated backward twin for short tile ranges. One 32-thread block per
# _SUBTILES-th of a tile (a pair of pixel rows) walks the range in lockstep with
# uniform broadcast reads and no shared staging. On a single warp wp.tile_sum
# lowers to shuffle reductions, so the gradient contributions are reduced across
# the lanes and lane 0 fires one atomic stream per gaussian, cutting the
# same-address atomic pressure 32x. This wins where tile ranges are short, since
# there is nothing for the staged walk to amortize and most lanes contribute to
# the same gaussian at once. On deep ranges the 8x higher gather traffic loses
# against the staged walk, see the launch gate.
_SUBTILES = BLOCK_SIZE // 32


@wp.kernel
def _rasterize_bwd_warp_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.bool,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    vout_rows: wp.int32,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    # outputs, atomically accumulated per gaussian
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
):
    blk, tr = wp.tid()
    tile_g = blk // _SUBTILES
    sub = blk % _SUBTILES
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = wp.where(sel_geom, image_id, 0)
    og_base = wp.where(sel_geom, 0, image_id * num_gaussians)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = sub * 2 + tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    inside = (i < img_h) and (j < img_w)
    bin_final = range_start - 1
    T = wp.float32(1.0)
    t_final = wp.float32(1.0)
    v_out = wp.vec3(0.0, 0.0, 0.0)
    bg = wp.vec3(0.0, 0.0, 0.0)
    if inside:
        frow = geom_image * img_h + i
        bin_final = final_idx[frow, j]
        t_final = final_Ts[frow, j]
        T = t_final
        v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
        bg = background[wp.where(sel_bg, image_id, 0)]
    buffer = wp.vec3(0.0, 0.0, 0.0)

    start_idx = wp.tile_max(wp.tile(bin_final))[0]

    for idx in range(start_idx, range_start - 1, -1):
        g = gaussian_ids_sorted[idx]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        dx = xy[0] - px
        dy = xy[1] - py
        sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
        vis = wp.exp(-sigma)
        alpha = wp.min(0.999, opac * vis)
        valid = (idx <= bin_final) and sigma >= 0.0 and alpha >= 1.0 / 255.0
        if wp.tile_max(wp.tile(wp.where(valid, 1, 0)))[0] == 0:
            continue

        v_rgb = wp.vec3(0.0, 0.0, 0.0)
        v_con = wp.vec3(0.0, 0.0, 0.0)
        v_xyl = wp.vec2(0.0, 0.0)
        v_op = wp.float32(0.0)
        if valid:
            ra = 1.0 / (1.0 - alpha)
            T = T * ra
            fac = alpha * T
            color = colors[g % color_mod]
            v_rgb = v_out * fac

            v_alpha = float(0.0)
            v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
            v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
            v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
            v_alpha += -t_final * ra * bg[0] * v_out[0]
            v_alpha += -t_final * ra * bg[1] * v_out[1]
            v_alpha += -t_final * ra * bg[2] * v_out[2]

            buffer = buffer + color * fac

            if opac * vis <= 0.999:
                v_sigma = -opac * vis * v_alpha
                v_con = wp.vec3(0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy)
                v_xyl = wp.vec2(
                    v_sigma * (conic[0] * dx + conic[1] * dy),
                    v_sigma * (conic[1] * dx + conic[2] * dy),
                )
                v_op = vis * v_alpha

        s_rgb = wp.tile_sum(wp.tile(v_rgb, preserve_type=True))[0]
        s_con = wp.tile_sum(wp.tile(v_con, preserve_type=True))[0]
        s_xy = wp.tile_sum(wp.tile(v_xyl, preserve_type=True))[0]
        s_op = wp.tile_sum(wp.tile(v_op))[0]
        if tr == 0:
            og = og_base + g
            wp.atomic_add(v_colors, og, s_rgb)
            wp.atomic_add(v_conic, og, s_con)
            wp.atomic_add(v_xy, og, s_xy)
            wp.atomic_add(v_opacity, og, s_op)


# Depth-augmented twin of _rasterize_bwd_warp_kernel. Walk, aggregation, and
# color-grad math are identical, with the depth channel handled as in
# _rasterize_bwd_depth_kernel.
@wp.kernel
def _rasterize_bwd_depth_warp_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.bool,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.bool,
    vout_rows: wp.int32,
    vdepth_rows: wp.int32,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    depths: wp.array[wp.float32],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    v_out_depth: wp.array2d[wp.float32],
    # outputs, atomically accumulated per gaussian
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_depths: wp.array[wp.float32],
):
    blk, tr = wp.tid()
    tile_g = blk // _SUBTILES
    sub = blk % _SUBTILES
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = wp.where(sel_geom, image_id, 0)
    og_base = wp.where(sel_geom, 0, image_id * num_gaussians)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = sub * 2 + tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    inside = (i < img_h) and (j < img_w)
    bin_final = range_start - 1
    T = wp.float32(1.0)
    t_final = wp.float32(1.0)
    v_out = wp.vec3(0.0, 0.0, 0.0)
    v_outd = wp.float32(0.0)
    bg = wp.vec3(0.0, 0.0, 0.0)
    if inside:
        frow = geom_image * img_h + i
        bin_final = final_idx[frow, j]
        t_final = final_Ts[frow, j]
        T = t_final
        v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
        v_outd = v_out_depth[(image_id * img_h + i) % vdepth_rows, j]
        bg = background[wp.where(sel_bg, image_id, 0)]
    buffer = wp.vec3(0.0, 0.0, 0.0)
    dbuffer = wp.float32(0.0)

    start_idx = wp.tile_max(wp.tile(bin_final))[0]

    for idx in range(start_idx, range_start - 1, -1):
        g = gaussian_ids_sorted[idx]
        xy = xys[g]
        conic = conics[g]
        opac = opacities[g % opac_mod]
        dx = xy[0] - px
        dy = xy[1] - py
        sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
        vis = wp.exp(-sigma)
        alpha = wp.min(0.999, opac * vis)
        valid = (idx <= bin_final) and sigma >= 0.0 and alpha >= 1.0 / 255.0
        if wp.tile_max(wp.tile(wp.where(valid, 1, 0)))[0] == 0:
            continue

        v_rgb = wp.vec3(0.0, 0.0, 0.0)
        v_con = wp.vec3(0.0, 0.0, 0.0)
        v_xyl = wp.vec2(0.0, 0.0)
        v_op = wp.float32(0.0)
        v_dep = wp.float32(0.0)
        if valid:
            ra = 1.0 / (1.0 - alpha)
            T = T * ra
            fac = alpha * T
            color = colors[g % color_mod]
            d = depths[g]
            v_rgb = v_out * fac
            v_dep = v_outd * fac

            v_alpha = float(0.0)
            v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
            v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
            v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
            # depth channel, background depth is 0 so there is no t_final*bg term
            v_alpha += (d * T - dbuffer * ra) * v_outd
            v_alpha += -t_final * ra * bg[0] * v_out[0]
            v_alpha += -t_final * ra * bg[1] * v_out[1]
            v_alpha += -t_final * ra * bg[2] * v_out[2]

            buffer = buffer + color * fac
            dbuffer = dbuffer + d * fac

            if opac * vis <= 0.999:
                v_sigma = -opac * vis * v_alpha
                v_con = wp.vec3(0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy)
                v_xyl = wp.vec2(
                    v_sigma * (conic[0] * dx + conic[1] * dy),
                    v_sigma * (conic[1] * dx + conic[2] * dy),
                )
                v_op = vis * v_alpha

        s_rgb = wp.tile_sum(wp.tile(v_rgb, preserve_type=True))[0]
        s_con = wp.tile_sum(wp.tile(v_con, preserve_type=True))[0]
        s_xy = wp.tile_sum(wp.tile(v_xyl, preserve_type=True))[0]
        s_op = wp.tile_sum(wp.tile(v_op))[0]
        s_dep = wp.tile_sum(wp.tile(v_dep))[0]
        if tr == 0:
            og = og_base + g
            wp.atomic_add(v_colors, og, s_rgb)
            wp.atomic_add(v_conic, og, s_con)
            wp.atomic_add(v_xy, og, s_xy)
            wp.atomic_add(v_opacity, og, s_op)
            wp.atomic_add(v_depths, og, s_dep)


def _rasterize_bwd_launch(
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    map_opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    cum_tiles_hit: wp.array[wp.int32],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    num_gaussians: int,
    img_h: int,
    img_w: int,
    # outputs
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
) -> None:
    # Two batch sizes. B_out comes from the grad output v_xy, always full batch
    # under expand_dims, and is how many images the blend walks. B_geom comes from
    # the geometry residual cum_tiles_hit and is how many distinct renders the
    # sort covers. They agree in the multi-view regime, but a shared render
    # differentiated against B target images gives B_geom=1 < B_out. A residual or
    # cotangent is not a reliable B_out signal because a cotangent can arrive
    # broadcast, which is why B_out comes from an output.
    n = num_gaussians
    B_out = v_xy.shape[0] // n
    B_geom = cum_tiles_hit.shape[0] // n
    sel_geom = B_geom > 1
    sel_bg = background.shape[0] > 1
    vout_rows = v_out_img.shape[0]

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = _blend_setup(
        colors, xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B_geom, img_h, img_w
    )

    # atomics accumulate, so outputs must start at zero
    v_colors.zero_()
    v_opacity.zero_()
    v_xy.zero_()
    v_conic.zero_()
    if num_intersects == 0:
        return
    # Kernel choice by mean tile range. Below one staged batch per tile the
    # warp-aggregated walk wins, above it the staged lockstep walk does.
    if num_intersects < B_geom * num_tiles * int(BLOCK_SIZE):
        _cached_launch(
            _rasterize_bwd_warp_kernel,
            B_out * num_tiles * int(_SUBTILES),
            [
                img_h,
                img_w,
                tile_bounds_x,
                num_tiles,
                n,
                sel_geom,
                colors.shape[0],
                opacities.shape[0],
                sel_bg,
                vout_rows,
                gaussian_ids,
                tile_bins,
                xys,
                conics,
                colors,
                opacities,
                background,
                final_Ts,
                final_idx,
                v_out_img,
                v_xy,
                v_conic,
                v_colors,
                v_opacity,
            ],
            colors.device,
            block_dim=32,
        )
        return
    _cached_launch(
        _rasterize_bwd_kernel,
        B_out * num_tiles,
        [
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            n,
            sel_geom,
            colors.shape[0],
            opacities.shape[0],
            sel_bg,
            vout_rows,
            gaussian_ids,
            tile_bins,
            xys,
            conics,
            colors,
            opacities,
            background,
            final_Ts,
            final_idx,
            v_out_img,
            v_xy,
            v_conic,
            v_colors,
            v_opacity,
        ],
        colors.device,
        block_dim=int(BLOCK_SIZE),
    )


_rasterize_bwd_ffi = nested_vmap(
    jax_callable(
        _rasterize_bwd_launch,
        num_outputs=4,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=12,
    name="rasterize_bwd",
)


def _rasterize_bwd_depth_launch(
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    map_opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    cum_tiles_hit: wp.array[wp.int32],
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    v_out_img: wp.array2d[wp.vec3],
    v_out_depth: wp.array2d[wp.float32],
    num_gaussians: int,
    img_h: int,
    img_w: int,
    # outputs
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_depths: wp.array[wp.float32],
) -> None:
    n = num_gaussians
    B_out = v_xy.shape[0] // n
    B_geom = cum_tiles_hit.shape[0] // n
    sel_geom = B_geom > 1
    sel_bg = background.shape[0] > 1
    vout_rows = v_out_img.shape[0]
    vdepth_rows = v_out_depth.shape[0]

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = _blend_setup(
        colors, xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B_geom, img_h, img_w
    )

    v_colors.zero_()
    v_opacity.zero_()
    v_xy.zero_()
    v_conic.zero_()
    v_depths.zero_()
    if num_intersects == 0:
        return
    # Kernel choice by mean tile range, as in _rasterize_bwd_launch.
    if num_intersects < B_geom * num_tiles * int(BLOCK_SIZE):
        _cached_launch(
            _rasterize_bwd_depth_warp_kernel,
            B_out * num_tiles * int(_SUBTILES),
            [
                img_h,
                img_w,
                tile_bounds_x,
                num_tiles,
                n,
                sel_geom,
                colors.shape[0],
                opacities.shape[0],
                sel_bg,
                vout_rows,
                vdepth_rows,
                gaussian_ids,
                tile_bins,
                xys,
                conics,
                colors,
                opacities,
                background,
                depths,
                final_Ts,
                final_idx,
                v_out_img,
                v_out_depth,
                v_xy,
                v_conic,
                v_colors,
                v_opacity,
                v_depths,
            ],
            colors.device,
            block_dim=32,
        )
        return
    _cached_launch(
        _rasterize_bwd_depth_kernel,
        B_out * num_tiles,
        [
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            n,
            sel_geom,
            colors.shape[0],
            opacities.shape[0],
            sel_bg,
            vout_rows,
            vdepth_rows,
            gaussian_ids,
            tile_bins,
            xys,
            conics,
            colors,
            opacities,
            background,
            depths,
            final_Ts,
            final_idx,
            v_out_img,
            v_out_depth,
            v_xy,
            v_conic,
            v_colors,
            v_opacity,
            v_depths,
        ],
        colors.device,
        block_dim=int(BLOCK_SIZE),
    )


_rasterize_bwd_depth_ffi = nested_vmap(
    jax_callable(
        _rasterize_bwd_depth_launch,
        num_outputs=5,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=13,
    name="rasterize_bwd_depth",
)
