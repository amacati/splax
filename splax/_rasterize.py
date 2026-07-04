"""Warp port of the reference CUDA rasterization stage (forward only), batch-native.

Ports the reference CUDA rasterizer (map_gaussian_to_intersects,
get_tile_bin_edges, rasterize_fwd, and its XLA op glue) faithfully
into Warp kernels, wrapped as a JAX FFI call via jax_callable. The callable:

  1. reads back num_intersects = cum_tiles_hit[-1] (one legitimate D->H sync),
  2. allocates scratch (isect keys, gaussian ids, tile bins) of that size,
  3. emits 64-bit intersection keys (image_id | tile_id | depth_bits) per
     gaussian/tile,
  4. sorts them once, globally (stable 64-bit radix sort),
  5. computes per-(image,tile) bin edges,
  6. blends front-to-back with per-pixel early termination.

Batching is native (gsplat O1). Under jax.vmap(expand_dims) the callable
launches a single grid over the whole batch (B*N for projection-derived stages,
B*num_tiles tile-blocks for the blend); the batch/image index is decoded from the
thread/block rank, packed into the sort key (image id above the tile bits), and
used to offset per-image bin edges, outputs, and backgrounds -- exactly gsplat's
launch-once design, no host loop. Per-input "batched or broadcast" indexing lets
projection intermediates arrive at full batch B while colors/opacities/background
may be broadcast (size 1). For B=1 every path reduces to the unbatched kernel.

Because of the host readback + data-dependent scratch alloc, this callable is NOT
CUDA-graph capturable, so graph_mode=NONE.

The public entry point returns the rendered image.
"""

import os
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental.ffi import jax_callable, JaxCallableGraphMode

from splax._project import (
    ALPHA_THRESHOLD,
    GAUSSIAN_EXTEND_SQ,
    _accutile_col,
    _accutile_first_imin,
    _accutile_setup,
)

wp.init()

# Tile geometry: a 16x16 pixel tile is processed by one 256-thread block, which
# cooperatively stages 256 gaussian records per batch into shared memory (faithful
# port of forward.cu::rasterize_fwd). block_width is always 16 (splax.render), so
# these are compile-time constants (tile shapes must be static).
BLOCK_WIDTH = wp.constant(16)
BLOCK_SIZE = wp.constant(256)  # BLOCK_WIDTH * BLOCK_WIDTH

# Packed grad contribution for the block-reduction backward variant: the 9
# per-gaussian grad scalars (colors 3, conic 3, xy 2, opacity 1) packed into one
# vector so a single wp.tile_sum reduces them all in one block barrier.
vec9 = wp.types.vector(length=9, dtype=wp.float32)

# --- gsplat-style 64-thread / 4-pixel CTA variant (survey O4) -----------------
# gsplat's RasterizeToPixels3DGSSerialBatchFwd uses (TILE, CTA) = (16, 64): a
# 64-thread block owns a 16x16 tile, each thread holding PIXELS_PER_THREAD=4
# pixels in registers, staging shared batches of only CTA_SIZE (64) gaussians.
# The 64 threads form a 16-wide x 4-tall lane grid; a thread's 4 pixels stride
# down the tile by ROW_STRIDE=4 rows (same column), covering all 16 rows. This
# reads each staged gaussian record 64x instead of 256x (4x less shared traffic
# per gaussian) and hides expf latency across the 4-pixel ILP; smaller batches
# give finer-grained early-exit. _USE_CTA64 selects the default at launch.
CTA_SIZE = wp.constant(64)  # threads per tile block
PIXELS_PER_THREAD = wp.constant(4)  # BLOCK_SIZE // CTA_SIZE = 256 / 64
ROW_STRIDE = wp.constant(4)  # CTA_SIZE // BLOCK_WIDTH = 64 / 16
ALL_DONE = wp.constant(15)  # (1 << PIXELS_PER_THREAD) - 1
# survey O4 verdict: the 64-thread/4-pixel CTA (_rasterize_fwd_cta64) regresses the
# dense, blend-bound configs ~3x (train.ply 512^2: 17.7 vs 5.9 ms) and is only
# break-even on sparse clouds, so the 256-thread/1-pixel kernel stays the default.
# gsplat's win from register-blocked pixels does not carry over: at 64 threads the
# 16x16 tile is worked by 2 warps instead of 8, and Warp's 4-pixel inner loop over
# vec4 register lanes does not recover the lost occupancy/ILP on deep tiles.
_USE_CTA64 = False
# When _USE_CTA64 is on, select the register-lane vec4 kernel (_rasterize_fwd_cta64)
# or the hand-unrolled independent-scalar-chain variant (_rasterize_fwd_cta64_unrolled,
# survey O4b). The vec4 kernel lets Warp's codegen thread the four pixel lanes through
# a single wp.vec_t<4> aggregate AND reuses the same Python temps (dy/sigma/alpha/...)
# across the compile-time-unrolled `for p`, so Warp emits phi-style wp::where merges
# that chain each pixel's expf onto the previous one -> serialized. The unrolled variant
# writes four fully independent scalar chains (T0..T3, distinct temps per pixel) so nvcc
# sees no cross-pixel dependency and can pipeline. See reports/phase5_o4.md addendum.
_CTA64_UNROLLED = False


def _bits_for_count(count: int) -> int:
    """gsplat MathUtils.h::bits_for_count: bits to index [0, count)."""
    return 0 if count <= 1 else (count - 1).bit_length()


# --- Persistent grow-only scratch cache (survey O3) ---------------------------
# The rasterize pipeline needs three per-frame buffers whose sizes depend on the
# (data-dependent) intersection count:
#   * isect_ids / gaussian_ids -- the radix-sort key/value ping-pong buffers. Warp's
#     wp.utils.radix_sort_pairs mandates a 2*count backing array because it drives a
#     cub::DoubleBuffer(keys, keys+count) internally (the 2N *is* the double buffer;
#     it is already the "halved aux" state gsplat gets from cub::DoubleBuffer, and
#     CUB's own temp storage is separately grow-only cached per stream inside warp).
#   * tile_bins -- the per-(image, tile) bin-edge array of length B*num_tiles.
# Re-allocating these from warp's mempool every frame costs a cold ~14 ms re-malloc
# of ~3 GB at B=8/1M/1080p (warm ~6 ms). We keep one set of buffers per device,
# keyed on the static shape signature (B, N, num_tiles): callers invoke this from
# jitted JAX functions, so the signature is fixed per compiled executable and
# repeated calls reuse stable-address buffers with zero mempool traffic. When the
# signature changes (new workload), everything is dropped and re-allocated, so a
# big config's scratch never lingers into a smaller one. Within one signature the
# sort buffers still track the running max of the (data-dependent, per-viewpoint)
# intersection count with 1.25x headroom, settling to zero reallocations after
# the first frames.
# Correctness notes:
#   * Keyed on device (per-device correctness); single stream / single process, so
#     no locking is needed.
#   * The sort key/id buffers are fully overwritten in their valid [0, num_intersects)
#     prefix every frame (every intersection slot is written exactly once by the map
#     kernel; culled gaussians contribute 0 tiles), so no stale-data hazard there.
#   * tile_bins is the exception: the bin-edge kernel only touches bins that own
#     intersections, leaving empty bins at their default (0, 0) = empty range. So the
#     used prefix MUST be explicitly zeroed each frame before the kernel runs.
#   * rasterize runs graph_mode=NONE (host readback + data-dependent shapes are not
#     CUDA-graph capturable), so reused buffer addresses raise no stale-capture
#     hazard; they are simply reused eagerly.
_SCRATCH_HEADROOM = 1.25
_scratch_cache: dict = {}  # device -> {sig, isect_cap, isect_ids, gaussian_ids, tile_bins}

# Packed 32-bit sort key (phase 8r), default on. Set SPLAX_PACK_KEYS=0 to force the
# legacy 64-bit key everywhere (A/B benchmarking + packed-vs-64bit quality checks).
_PACK_KEYS = os.environ.get("SPLAX_PACK_KEYS", "1") == "1"

# --- Post-sync CUDA-graph capture (phase 8s), opt-in -------------------------
# Capture the whole post-readback sequence (sentinel fill / depth minmax / map /
# sort / bin / blend) as a cached CUDA graph and replay it, collapsing ~11 device
# launches to one graph replay. Recovers the per-frame launch overhead the premise
# measured (7-29% of the post-sync region at small/mid workloads; the bucket pad
# tax regresses very large frames, hence the count threshold). Default OFF: the
# forward path stays byte-identical until enabled. Packed-path + forward-only.
#   SPLAX_POSTSYNC_GRAPHS=1   enable
#   SPLAX_GRAPH_THRESHOLD=N   max num_intersects to use a graph (default 2_000_000;
#                             above it the pad tax on the dominant sort/blend eats
#                             the launch win, so fall back to plain launches).
_POSTSYNC_GRAPHS = os.environ.get("SPLAX_POSTSYNC_GRAPHS", "0") == "1"
_GRAPH_THRESHOLD = int(os.environ.get("SPLAX_GRAPH_THRESHOLD", "2000000"))
_GRAPH_BUCKET_STEP = 1.05  # geometric bucket granularity (~5% pad worst case)
_graph_cache: dict = {}  # (device, key-tuple) -> wp.Graph

# --- Split-heavy-tile load balancing (phase 8t), inference-only, opt-in --------
# The one-block-per-tile blend makes frame time follow the HEAVIEST tile's
# depth-sorted bin (reports/phase1.md, phase4b.md): a dense scene at low resolution
# lands in few tiles with one enormous bin (~80k gaussians at 512^2) while most
# tiles idle, so a handful of blocks grind serially on idle SMs. Alpha compositing
# is associative over depth-ordered segments -- a segment blends to (C_seg, T_seg)
# and two adjacent segments combine as C = C_a + T_a*C_b, T = T_a*T_b -- so a heavy
# tile's bin can be split into K contiguous depth-ordered sub-ranges, each blended
# by its own block in parallel, then merged per pixel by a tiny composite pass.
# This trades the heaviest tile's serial depth (~max_bin) for max_bin/K, filling the
# idle SMs (FlashGS / Balanced-3DGS / gsplat load-balancing family).
#
# Applied ONLY to the pure-inference blend (splax.inference.render): the training
# forward keeps final_Ts/final_idx as backward residuals, and the segment split
# reorders the per-pixel final_idx bookkeeping, so the differentiable path stays on
# the unsplit kernel (byte-identical, backward untouched). Inference discards those
# residuals, so only out_img must match -- and it does to a few ULP (segment
# compositing only reorders float adds across segment boundaries).
#   SPLAX_TILE_SPLIT=1        enable (default off: inference stays byte-identical)
#   SPLAX_SPLIT_THRESHOLD=N   split a tile whose bin exceeds N gaussians (default 8000)
#   SPLAX_SPLIT_KCAP=K        max sub-ranges a tile is split into (default 16)
_TILE_SPLIT = os.environ.get("SPLAX_TILE_SPLIT", "0") == "1"
_SPLIT_THRESHOLD = int(os.environ.get("SPLAX_SPLIT_THRESHOLD", "8000"))
_SPLIT_KCAP = int(os.environ.get("SPLAX_SPLIT_KCAP", "16"))
_split_scratch_cache: dict = {}  # device -> {sig, caps, buffers}


def _bucket_count(ni: int) -> int:
    """Round num_intersects up to the next ~5% geometric step (graph reuse bucket)."""
    if ni <= 1:
        return max(ni, 1)
    import math

    k = math.ceil(math.log(ni) / math.log(_GRAPH_BUCKET_STEP))
    return max(ni, int(math.ceil(_GRAPH_BUCKET_STEP**k)))


def clear_graph_cache() -> None:
    """Release all cached post-sync CUDA graphs (all devices)."""
    _graph_cache.clear()


def _get_scratch(
    device: wp.Device | None,
    sig: tuple,
    isect_need: int,
    bins_need: int,
    isect_dtype: type = wp.int64,
) -> dict:
    key = str(device)  # wp.Device is unhashable; its string alias is stable per device
    entry = _scratch_cache.get(key)
    if entry is None or entry["sig"] != sig:
        # new workload signature: drop everything (refs -> warp mempool) first so
        # the peak is the new size, not old+new. Any captured CUDA graph records the
        # addresses of these scratch buffers (isect/gaussian ids, tile_bins,
        # depth_mm); freeing them makes every cached graph dangling, so purge the
        # graph cache too (phase 8s). A stale-address ABA -- a freed buffer's address
        # later reused for a different tensor while an old graph keyed on it lingers --
        # would otherwise replay into freed memory (illegal access). Destroy the
        # graphs BEFORE dropping the scratch refs so each graph is torn down while
        # the buffers it recorded are still alive.
        _graph_cache.clear()
        _scratch_cache.pop(key, None)
        entry = {
            "sig": sig,
            "isect_cap": 0,
            "isect_dtype": isect_dtype,
            "isect_ids": None,
            "gaussian_ids": None,
            "tile_bins": wp.empty(bins_need, dtype=wp.vec2i, device=device),
            "depth_mm": wp.empty(2 * max(sig[0], 1), dtype=wp.float32, device=device),
            # generation counter: bumped on every (re)allocation of the sort buffers.
            # A captured CUDA graph records the buffer *addresses*, so any realloc
            # invalidates every graph keyed on the old generation (phase 8s).
            "gen": 0,
        }
        _scratch_cache[key] = entry
    if entry["isect_cap"] < isect_need or entry["isect_dtype"] != isect_dtype:
        cap = max(int(isect_need * _SCRATCH_HEADROOM) + 1, entry["isect_cap"])
        entry["gen"] += 1
        _graph_cache.clear()  # sort buffers move on realloc -> invalidate all graphs
        # free before allocating larger, avoiding an old+new transient peak. The
        # sort-key buffer is int32 in the packed path (phase 8r), int64 otherwise;
        # dtype is fixed by the signature (packing depends only on B/num_tiles), so
        # a dtype change only happens if the SPLAX_PACK_KEYS switch is toggled.
        entry["isect_ids"] = None
        entry["gaussian_ids"] = None
        entry["isect_cap"] = 0
        entry["isect_dtype"] = isect_dtype
        entry["isect_ids"] = wp.empty(cap, dtype=isect_dtype, device=device)
        entry["gaussian_ids"] = wp.empty(cap, dtype=wp.int32, device=device)
        entry["isect_cap"] = cap
    return entry


def clear_scratch() -> None:
    """Release the persistent rasterize scratch buffers (all devices).

    The Warp backend caches grow-only sort and bin-edge scratch across renders
    (survey O3). Call this to free that memory, e.g. before switching to a very
    different workload size or to reclaim the peak sort footprint. Also purges the
    post-sync CUDA-graph cache (its graphs reference the freed scratch addresses).
    """
    _graph_cache.clear()
    _scratch_cache.clear()


@wp.kernel
def _map_gaussian_to_intersects(
    xys: wp.array[wp.vec2],
    depths_int: wp.array[wp.int32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    map_opacities: wp.array[wp.float32],
    cum_tiles_hit: wp.array[wp.int32],
    num_gaussians: wp.int32,
    opac_mod: wp.int32,
    tight: wp.int32,
    tile_n_bits: wp.int32,
    tile_bounds_x: wp.int32,
    tile_bounds_y: wp.int32,
    block_width: wp.int32,
    # outputs
    isect_ids: wp.array[wp.int64],
    gaussian_ids: wp.array[wp.int32],
):
    # Launched over B*N flat threads. bid decodes the batch/image element; the
    # per-gaussian arrays (xys, depths, radii, conics) and cum_tiles_hit are batched
    # (full B*N), so index by the flat idx directly. The gaussian id stored is the
    # flat idx (b*N+gid) -- the blend uses it to gather batched arrays, and shifts
    # it back to a local gid for broadcast (size-N) attributes.
    idx = wp.tid()
    if radii[idx] <= 0:
        return
    n = num_gaussians
    bid = idx // n
    center = xys[idx]
    bw = wp.float32(block_width)

    cur_idx = wp.int32(0)
    if idx > 0:
        cur_idx = cum_tiles_hit[idx - 1]

    # depth bits reinterpreted as int (depths are positive -> raw bits sort as ints)
    depth_id = wp.int64(depths_int[idx])
    # image id occupies the bits above the tile field (gsplat IntersectTile.cu:189).
    iid_enc = wp.int64(bid) << (wp.int64(32) + wp.int64(tile_n_bits))

    if tight != 0:
        # AccuTile ellipse walk (survey O6): recompute the SAME opacity-aware ellipse
        # setup + column ranges as projection's tile COUNT (shared _accutile_* funcs),
        # emitting exactly num_tiles_hit keys per gaussian so cum offsets stay valid.
        # ``map_opacities`` is the RAW opacity projection counted with -- when the
        # blend uses ρ-compensated opacity (antialiased mode) this stays raw so the
        # emitted key total still matches cum_tiles_hit exactly.
        opac = map_opacities[idx % opac_mod]
        t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
        conic = conics[idx]
        setup = _accutile_setup(
            conic[0],
            conic[1],
            conic[2],
            t,
            center[0],
            center[1],
            block_width,
            tile_bounds_x,
            tile_bounds_y,
        )
        if setup.valid == 0:
            return
        I_min = _accutile_first_imin(setup, bw)
        for u in range(setup.rect_min[0], setup.rect_max[0]):
            rc = _accutile_col(u, setup, bw, I_min)
            mn = wp.int32(rc[0])
            mx = wp.int32(rc[1])
            for v in range(mn, mx):
                if setup.isY != 0:
                    tile_id = wp.int64(u * tile_bounds_x + v)
                else:
                    tile_id = wp.int64(v * tile_bounds_x + u)
                isect_ids[cur_idx] = iid_enc | (tile_id << wp.int64(32)) | depth_id
                gaussian_ids[cur_idx] = idx
                cur_idx = cur_idx + 1
            I_min = wp.vec2(rc[2], rc[3])
        return

    # legacy isotropic bbox emit: pix_radius is (float)radii[idx]. Because radius =
    # ceil(...) in projection, (int)radius == radius exactly, so the int-truncated
    # radii here reproduces the float bbox used to size num_tiles_hit.
    r = wp.float32(radii[idx])
    tc_x = center[0] / bw
    tc_y = center[1] / bw
    tr = r / bw
    tmin_x = wp.min(wp.max(0, wp.int32(tc_x - tr)), tile_bounds_x)
    tmax_x = wp.min(wp.max(0, wp.int32(tc_x + tr + 1.0)), tile_bounds_x)
    tmin_y = wp.min(wp.max(0, wp.int32(tc_y - tr)), tile_bounds_y)
    tmax_y = wp.min(wp.max(0, wp.int32(tc_y + tr + 1.0)), tile_bounds_y)

    for i in range(tmin_y, tmax_y):
        for j in range(tmin_x, tmax_x):
            tile_id = wp.int64(i * tile_bounds_x + j)
            isect_ids[cur_idx] = iid_enc | (tile_id << wp.int64(32)) | depth_id
            gaussian_ids[cur_idx] = idx
            cur_idx = cur_idx + 1


@wp.kernel
def _get_tile_bin_edges(
    num_intersects: wp.int32,
    isect_ids_sorted: wp.array[wp.int64],
    num_tiles: wp.int32,
    tile_n_bits: wp.int32,
    # output
    tile_bins: wp.array[wp.vec2i],
):
    # Per-(image,tile) bin edges into a [B*num_tiles] array (gsplat
    # RasterizeToPixels...Fwd.cu:79: isect_offsets += image_id*grid_h*grid_w). The
    # flat bin index is iid*num_tiles + tile_id, decoded from the key's upper field.
    idx = wp.tid()
    if idx >= num_intersects:
        return
    mask = (wp.int64(1) << wp.int64(tile_n_bits)) - wp.int64(1)
    key = isect_ids_sorted[idx] >> wp.int64(32)  # iid<<tile_n_bits | tile_id
    cur_bin = wp.int32(key >> wp.int64(tile_n_bits)) * num_tiles + wp.int32(key & mask)
    if idx == 0:
        tile_bins[cur_bin][0] = 0
        return
    if idx == num_intersects - 1:
        tile_bins[cur_bin][1] = num_intersects
    keyp = isect_ids_sorted[idx - 1] >> wp.int64(32)
    prev_bin = wp.int32(keyp >> wp.int64(tile_n_bits)) * num_tiles + wp.int32(
        keyp & mask
    )
    if prev_bin != cur_bin:
        tile_bins[prev_bin][1] = idx
        tile_bins[cur_bin][0] = idx


# --- Packed 32-bit key variant (phase 8r) -------------------------------------
# When image+tile ids leave >=16 bits below bit 31, we pack the whole sort key into
# a single non-negative int32: (iid | tile_id | quant_depth) laid out as
#   [ iid | tile_id | quant_depth ]  with fields (image_n_bits, tile_n_bits, depth_bits)
#   depth_bits = 31 - (image_n_bits + tile_n_bits)  (>=16 when packing)
# The top bit (31, the sign) is always 0, so cub's signed-int radix sort orders the
# keys as plain unsigned ascending -- same (tile, depth) ordering as the 64-bit path,
# but over 32 bits (4 radix passes) instead of 64 (8 passes), halving both the pass
# count and the bytes moved per pass.
#
# quant_depth LINEARLY quantizes the camera depth into ``depth_bits`` buckets over the
# per-frame [dmin, dmax] range (reduced device-side by _depth_minmax, no host sync):
#   q = floor((d - dmin) / (dmax - dmin) * (2^depth_bits - 1)).
# This is MONOTONE in d, so front-to-back order is preserved; it is coarser than the
# 64-bit path's full float depth word, so gaussians landing in the same bucket
# (~(dmax-dmin)/2^depth_bits apart, ~1e-5 of the range at depth_bits=18) keep
# gaussian-id order under the stable sort instead of exact depth order -- a
# blend-order change confined to near-coincident gaussians (perceptually negligible;
# see report). Linear-over-range beats truncating the float mantissa: it spends every
# bucket inside the scene's actual depth span instead of wasting the exponent field on
# the unused [1e-38, 1e38] float range.
_MINMAX_CHUNK = wp.constant(32)  # gaussians reduced per thread before one atomic pair


@wp.kernel
def _depth_minmax(
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    total: wp.int32,
    num_gaussians: wp.int32,
    # output: PER-IMAGE [min, max] pairs (length 2*B), pre-seeded to [+big, -big].
    out_mm: wp.array[wp.float32],
):
    # Grid-stride: each thread privately reduces _MINMAX_CHUNK consecutive gaussians
    # into registers, then issues one atomic pair -- cutting global atomics (and their
    # contention) by _MINMAX_CHUNK. The range is kept PER IMAGE (indexed by the flat
    # id's image = idx // n) so a batched (vmap) render quantizes each view exactly as
    # the corresponding unbatched B=1 render would (batch-native == stack-of-unbatched).
    # A 32-wide chunk spans at most one image boundary (n >> 32), handled by flushing
    # the accumulator when the image changes. For B=1 the image is always 0.
    tid = wp.tid()
    base = tid * _MINMAX_CHUNK
    img_cur = wp.int32(-1)
    lo = wp.float32(1.0e30)
    hi = wp.float32(-1.0e30)
    for k in range(_MINMAX_CHUNK):
        idx = base + k
        if idx < total:
            if radii[idx] > 0:  # culled gaussians emit no keys -> exclude from range
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
def _seed_minmax(out_mm: wp.array[wp.float32]):
    b = wp.tid()  # one thread per image
    out_mm[2 * b] = 1.0e30
    out_mm[2 * b + 1] = -1.0e30


@wp.kernel
def _map_gaussian_to_intersects_p32(
    xys: wp.array[wp.vec2],
    depths: wp.array[wp.float32],
    radii: wp.array[wp.int32],
    conics: wp.array[wp.vec3],
    map_opacities: wp.array[wp.float32],
    cum_tiles_hit: wp.array[wp.int32],
    depth_mm: wp.array[wp.float32],
    num_gaussians: wp.int32,
    opac_mod: wp.int32,
    tight: wp.int32,
    tile_n_bits: wp.int32,
    depth_bits: wp.int32,
    tile_bounds_x: wp.int32,
    tile_bounds_y: wp.int32,
    block_width: wp.int32,
    # outputs
    isect_ids: wp.array[wp.int32],
    gaussian_ids: wp.array[wp.int32],
):
    # int32-packed twin of _map_gaussian_to_intersects. Identical tile emission
    # (tight ellipse walk / legacy bbox); only the key composition differs.
    idx = wp.tid()
    if radii[idx] <= 0:
        return
    n = num_gaussians
    bid = idx // n
    center = xys[idx]
    bw = wp.float32(block_width)

    cur_idx = wp.int32(0)
    if idx > 0:
        cur_idx = cum_tiles_hit[idx - 1]

    # linear depth quantization over this image's [dmin, dmax] range (monotone in d).
    dmin = depth_mm[2 * bid]
    drange = depth_mm[2 * bid + 1] - dmin
    maxq = wp.float32((wp.int32(1) << depth_bits) - wp.int32(1))
    depth_q = wp.int32(0)
    if drange > 0.0:
        f = (depths[idx] - dmin) / drange
        depth_q = wp.clamp(
            wp.int32(f * maxq), wp.int32(0), (wp.int32(1) << depth_bits) - wp.int32(1)
        )
    # image id occupies the field above (tile_n_bits + depth_bits).
    iid_enc = bid << (depth_bits + tile_n_bits)

    if tight != 0:
        opac = map_opacities[idx % opac_mod]
        t = wp.min(GAUSSIAN_EXTEND_SQ, 2.0 * wp.log(opac / ALPHA_THRESHOLD))
        conic = conics[idx]
        setup = _accutile_setup(
            conic[0],
            conic[1],
            conic[2],
            t,
            center[0],
            center[1],
            block_width,
            tile_bounds_x,
            tile_bounds_y,
        )
        if setup.valid == 0:
            return
        I_min = _accutile_first_imin(setup, bw)
        for u in range(setup.rect_min[0], setup.rect_max[0]):
            rc = _accutile_col(u, setup, bw, I_min)
            mn = wp.int32(rc[0])
            mx = wp.int32(rc[1])
            for v in range(mn, mx):
                if setup.isY != 0:
                    tile_id = u * tile_bounds_x + v
                else:
                    tile_id = v * tile_bounds_x + u
                isect_ids[cur_idx] = iid_enc | (tile_id << depth_bits) | depth_q
                gaussian_ids[cur_idx] = idx
                cur_idx = cur_idx + 1
            I_min = wp.vec2(rc[2], rc[3])
        return

    r = wp.float32(radii[idx])
    tc_x = center[0] / bw
    tc_y = center[1] / bw
    tr = r / bw
    tmin_x = wp.min(wp.max(0, wp.int32(tc_x - tr)), tile_bounds_x)
    tmax_x = wp.min(wp.max(0, wp.int32(tc_x + tr + 1.0)), tile_bounds_x)
    tmin_y = wp.min(wp.max(0, wp.int32(tc_y - tr)), tile_bounds_y)
    tmax_y = wp.min(wp.max(0, wp.int32(tc_y + tr + 1.0)), tile_bounds_y)

    for i in range(tmin_y, tmax_y):
        for j in range(tmin_x, tmax_x):
            tile_id = i * tile_bounds_x + j
            isect_ids[cur_idx] = iid_enc | (tile_id << depth_bits) | depth_q
            gaussian_ids[cur_idx] = idx
            cur_idx = cur_idx + 1


@wp.kernel
def _get_tile_bin_edges_p32(
    num_intersects: wp.int32,
    isect_ids_sorted: wp.array[wp.int32],
    num_tiles: wp.int32,
    tile_n_bits: wp.int32,
    depth_bits: wp.int32,
    # output
    tile_bins: wp.array[wp.vec2i],
):
    # int32-packed twin of _get_tile_bin_edges: the (iid|tile) field sits above the
    # depth_bits-wide depth field (not bit 32).
    idx = wp.tid()
    if idx >= num_intersects:
        return
    mask = (wp.int32(1) << tile_n_bits) - wp.int32(1)
    key = isect_ids_sorted[idx] >> depth_bits  # iid<<tile_n_bits | tile_id
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
def _get_tile_bin_edges_p32_dev(
    count_arr: wp.array[wp.int32],
    count_idx: wp.int32,
    isect_ids_sorted: wp.array[wp.int32],
    num_tiles: wp.int32,
    tile_n_bits: wp.int32,
    depth_bits: wp.int32,
    # output
    tile_bins: wp.array[wp.vec2i],
):
    # Device-count-guarded twin of _get_tile_bin_edges_p32 for the captured-graph
    # path (phase 8s). The real intersection count is read from device memory
    # (count_arr[count_idx] == cum_tiles_hit[total-1]) at *replay* time, not baked
    # into the graph. The sort runs over the padded bucket (sentinel keys 0x7FFFFFFF
    # at the tail); this guard makes every thread with idx >= real count return, so
    # the writes are byte-identical to the non-graph _get_tile_bin_edges_p32 launched
    # at dim=count -- the sentinel tail is never touched (no OOB from the huge
    # sentinel tile field). The blend reads only real bins.
    idx = wp.tid()
    num_intersects = count_arr[count_idx]
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
def _rasterize_fwd(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # Cooperative shared-memory blend (faithful port of forward.cu::rasterize_fwd).
    # One block per (image, tile): dim = B*num_tiles, so image_id = block //
    # num_tiles decodes the batch element (gsplat blockIdx.x). Outputs are the
    # collapsed batched buffers (B*H, W), written at row image_id*H + i. The
    # gathered gaussian ids are flat (b*N+gid); xys/conics are batched so the flat
    # id indexes them directly, while broadcast (size-N) colors/opacities are
    # shifted back by image_id*N.
    tile_g, tr = wp.tid()  # launch_tiled: tile_g = block index, tr = thread rank
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

    # threads mapping outside the image stay live (needed for collective loads /
    # the block vote) but are marked done and never write an output pixel.
    inside = (i < img_h) and (j < img_w)
    done = wp.bool(not inside)

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE

    T = wp.float32(1.0)
    cur_idx = wp.int32(0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)

    # Broadcast (size-N) attributes must be read at the local gid, batched (size
    # B*N) ones at the flat id. Modulo by the array's own leading dim does both:
    # for real lanes id % N = gid (broadcast) and id % (B*N) = id (batched); it also
    # keeps every OOB/zero-filled/neighbouring-tile lane the cooperative load pulls
    # in-bounds (a negative subtraction would fault). Unused lanes are skipped by
    # the batch_size bound, so the exact wrapped value there is irrelevant.

    for b in range(num_batches):
        # whole-tile early-out vote: break once every thread in the block is done.
        done_count = wp.tile_sum(wp.tile(wp.where(done, 1, 0)))
        if wp.tile_extract(done_count, 0) >= BLOCK_SIZE:
            break

        batch_start = range_start + b * BLOCK_SIZE
        # cooperatively stage this batch of gaussian records into shared memory.
        id_tile = wp.tile_load(
            gaussian_ids_sorted, BLOCK_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        # broadcast/batched attribute index via modulo (see note above). The
        # divisor tile is built with wp.tile_full (a static shape) rather than
        # wp.tile(scalar) (a block_dim-sized shape): both blend variants live in
        # this module, which Warp codegens under a single block_dim, so a
        # block_dim-independent tile is required for the two static tile widths
        # (256 here, 64 in the CTA kernel) to compile side by side.
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )

        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                dx = xy[0] - px
                dy = xy[1] - py
                sigma = (
                    0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                )
                alpha = wp.min(0.999, opac * wp.exp(-sigma))
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
        bg = background[image_id * sel_bg]
        row = image_id * img_h + i
        final_Ts[row, j] = T
        final_idx[row, j] = cur_idx
        out_img[row, j] = pix_out + T * bg


# --- Split-heavy-tile blend (phase 8t) ----------------------------------------
# Three cooperating kernels build and run the load-balanced blend:
#   1. _plan_tile_split -- one thread per (image,tile). Tiles whose bin exceeds
#      split_threshold are cut into K = min(ceil(count/threshold), kcap) contiguous
#      depth-ordered segments; the plan (compacted via two device atomics) records
#      per-split metadata and a per-segment work list. Light tiles produce nothing
#      here (they are blended by _rasterize_fwd_unsplit).
#   2. _rasterize_fwd_seg -- one 256-thread block per segment (worst-case launch,
#      device-count guarded). Byte-identical blend math to _rasterize_fwd over the
#      segment's sub-range, writing per-pixel partials (C_seg, T_seg) to a slot buffer
#      keyed on the compact split index. No background term (the composite adds it).
#   3. _composite_tile_split -- one thread per pixel of each split tile (device-count
#      guarded). Merges the K partials front-to-back: C = Σ (Π_{s'<s} T_s') C_s,
#      T = Π T_s, writing out_img = C + T*bg. Non-split tiles are already final.


@wp.kernel
def _plan_tile_split(
    tile_bins: wp.array[wp.vec2i],
    n_bins: wp.int32,
    split_threshold: wp.int32,
    kcap: wp.int32,
    seg_cap: wp.int32,
    split_cap: wp.int32,
    # atomic counters (length 1, pre-zeroed): [0] = #splits, [1] = #segments
    split_counter: wp.array[wp.int32],
    seg_counter: wp.array[wp.int32],
    # per-split metadata (length split_cap)
    split_tile_g: wp.array[wp.int32],
    split_k: wp.array[wp.int32],
    # per-segment work list (length seg_cap)
    seg_tile_g: wp.array[wp.int32],
    seg_start: wp.array[wp.int32],
    seg_end: wp.array[wp.int32],
    seg_split_idx: wp.array[wp.int32],
    seg_ord: wp.array[wp.int32],
):
    g = wp.tid()
    if g >= n_bins:
        return
    rng = tile_bins[g]
    count = rng[1] - rng[0]
    if count <= split_threshold:
        return  # light tile: handled by _rasterize_fwd_unsplit
    k = (count + split_threshold - 1) // split_threshold
    if k > kcap:
        k = kcap
    seg_len = (count + k - 1) // k
    si = wp.atomic_add(split_counter, 0, 1)
    if si >= split_cap:
        return  # overflow guard (host bound should preclude this)
    # publish metadata before the segment loop so a counted split index always has
    # valid (tile, K); on the (host-bound-precluded) seg overflow, K=0 -> composite
    # no-ops for this pixel instead of reading uninitialized slots.
    split_tile_g[si] = g
    split_k[si] = 0
    base = wp.atomic_add(seg_counter, 0, k)
    if base + k > seg_cap:
        return
    split_k[si] = k
    for s in range(k):
        ss = rng[0] + s * seg_len
        se = ss + seg_len
        if se > rng[1]:
            se = rng[1]
        j = base + s
        seg_tile_g[j] = g
        seg_start[j] = ss
        seg_end[j] = se
        seg_split_idx[j] = si
        seg_ord[j] = s


@wp.kernel
def _rasterize_fwd_unsplit(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
    split_threshold: wp.int32,
    gaussian_ids_sorted: wp.array[wp.int32],
    tile_bins: wp.array[wp.vec2i],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    # outputs
    out_img: wp.array2d[wp.vec3],
):
    # Byte-identical to _rasterize_fwd's blend, but only for LIGHT tiles (count <=
    # split_threshold); heavy tiles return immediately (their pixels are produced by
    # the segment + composite kernels). No final_Ts/final_idx (inference discards them).
    tile_g, tr = wp.tid()
    tile_range = tile_bins[tile_g]
    if (tile_range[1] - tile_range[0]) > split_threshold:
        return
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

    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE
    T = wp.float32(1.0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)

    for b in range(num_batches):
        done_count = wp.tile_sum(wp.tile(wp.where(done, 1, 0)))
        if wp.tile_extract(done_count, 0) >= BLOCK_SIZE:
            break
        batch_start = range_start + b * BLOCK_SIZE
        id_tile = wp.tile_load(
            gaussian_ids_sorted, BLOCK_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                dx = xy[0] - px
                dy = xy[1] - py
                sigma = (
                    0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                )
                alpha = wp.min(0.999, opac * wp.exp(-sigma))
                if sigma < 0.0 or alpha < 1.0 / 255.0:
                    continue
                next_T = T * (1.0 - alpha)
                if next_T <= 1e-4:
                    done = wp.bool(True)
                    break
                vis = alpha * T
                pix_out = pix_out + color_tile[t] * vis
                T = next_T

    if inside:
        bg = background[image_id * sel_bg]
        row = image_id * img_h + i
        out_img[row, j] = pix_out + T * bg


@wp.kernel
def _rasterize_fwd_seg(
    color_mod: wp.int32,
    opac_mod: wp.int32,
    kcap: wp.int32,
    seg_counter: wp.array[wp.int32],
    seg_tile_g: wp.array[wp.int32],
    seg_start: wp.array[wp.int32],
    seg_end: wp.array[wp.int32],
    seg_split_idx: wp.array[wp.int32],
    seg_ord: wp.array[wp.int32],
    gaussian_ids_sorted: wp.array[wp.int32],
    xys: wp.array[wp.vec2],
    conics: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
    opacities: wp.array[wp.float32],
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    # outputs (slot buffers indexed (split_idx*kcap + seg_ord)*BLOCK_SIZE + tr)
    slot_c: wp.array[wp.vec3],
    slot_t: wp.array[wp.float32],
):
    # One 256-thread block per work-list segment. Blends the segment's sub-range
    # front-to-back for the tile's 256 pixels exactly like _rasterize_fwd (local T
    # starts at 1), then stores the per-pixel partial (C_seg, T_seg) to the slot.
    seg_g, tr = wp.tid()
    if seg_g >= seg_counter[0]:
        return
    tile_g = seg_tile_g[seg_g]
    si = seg_split_idx[seg_g]
    so = seg_ord[seg_g]
    tile_local = tile_g % num_tiles
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj
    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    range_start = seg_start[seg_g]
    range_end = seg_end[seg_g]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Segment self-saturation early-out: a pixel whose LOCAL T within this segment
    # collapses stops accumulating (its later, deeper gaussians are occluded within
    # the segment). Cross-segment occlusion is resolved by the composite; a segment
    # cannot see preceding segments' T, so it never early-outs on global opacity.
    done = wp.bool(False)
    T = wp.float32(1.0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)

    for b in range(num_batches):
        done_count = wp.tile_sum(wp.tile(wp.where(done, 1, 0)))
        if wp.tile_extract(done_count, 0) >= BLOCK_SIZE:
            break
        batch_start = range_start + b * BLOCK_SIZE
        id_tile = wp.tile_load(
            gaussian_ids_sorted, BLOCK_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                dx = xy[0] - px
                dy = xy[1] - py
                sigma = (
                    0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                )
                alpha = wp.min(0.999, opac * wp.exp(-sigma))
                if sigma < 0.0 or alpha < 1.0 / 255.0:
                    continue
                next_T = T * (1.0 - alpha)
                if next_T <= 1e-4:
                    # freeze at the last visible contribution, then stop (matches the
                    # unsplit kernel: the saturating gaussian is NOT accumulated).
                    done = wp.bool(True)
                    break
                vis = alpha * T
                pix_out = pix_out + color_tile[t] * vis
                T = next_T

    slot = (si * kcap + so) * BLOCK_SIZE + tr
    slot_c[slot] = pix_out
    slot_t[slot] = T


@wp.kernel
def _composite_tile_split(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    sel_bg: wp.int32,
    kcap: wp.int32,
    split_counter: wp.array[wp.int32],
    split_tile_g: wp.array[wp.int32],
    split_k: wp.array[wp.int32],
    slot_c: wp.array[wp.vec3],
    slot_t: wp.array[wp.float32],
    background: wp.array[wp.vec3],
    # output
    out_img: wp.array2d[wp.vec3],
):
    # One thread per pixel of each split tile: dim = split_cap*BLOCK_SIZE, guarded on
    # the device split count. Merges the tile's K depth-ordered partials front-to-back.
    gid = wp.tid()
    si = gid // BLOCK_SIZE
    if si >= split_counter[0]:
        return
    tr = gid % BLOCK_SIZE
    tile_g = split_tile_g[si]
    k = split_k[si]
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj
    if i >= img_h or j >= img_w:
        return
    T = wp.float32(1.0)
    C = wp.vec3(0.0, 0.0, 0.0)
    for s in range(k):
        slot = (si * kcap + s) * BLOCK_SIZE + tr
        C = C + T * slot_c[slot]
        T = T * slot_t[slot]
    bg = background[image_id * sel_bg]
    out_img[image_id * img_h + i, j] = C + T * bg


# --- Depth-augmented forward (survey T2) --------------------------------------
# Opt-in expected-depth channel D(p) = Σ wᵢ dᵢ (alpha-blend weights wᵢ = αᵢ Tᵢ over
# the per-gaussian camera-space depth dᵢ), for COLMAP sparse-point depth
# regularization. This is a SEPARATE kernel from _rasterize_fwd: the default render
# never pays for the extra accumulator/load, staying byte-identical. Blend math,
# early-exit vote and batched image_id indexing are identical to _rasterize_fwd; the
# only additions are the depth staging load and the depth_out accumulator. Background
# depth is 0, so the depth channel has no T·bg term (unlike colour).
@wp.kernel
def _rasterize_fwd_depth(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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

    for b in range(num_batches):
        done_count = wp.tile_sum(wp.tile(wp.where(done, 1, 0)))
        if wp.tile_extract(done_count, 0) >= BLOCK_SIZE:
            break

        batch_start = range_start + b * BLOCK_SIZE
        id_tile = wp.tile_load(
            gaussian_ids_sorted, BLOCK_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        depth_tile = wp.tile_load_indexed(
            depths, indices=id_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(BLOCK_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(BLOCK_SIZE,), axis=0, storage="shared"
        )

        batch_size = wp.min(BLOCK_SIZE, range_end - batch_start)
        if not done:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                dx = xy[0] - px
                dy = xy[1] - py
                sigma = (
                    0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
                )
                alpha = wp.min(0.999, opac * wp.exp(-sigma))
                if sigma < 0.0 or alpha < 1.0 / 255.0:
                    continue
                next_T = T * (1.0 - alpha)
                if next_T <= 1e-4:
                    done = wp.bool(True)
                    break
                vis = alpha * T
                pix_out = pix_out + color_tile[t] * vis
                depth_out = depth_out + depth_tile[t] * vis
                T = next_T
                cur_idx = batch_start + t

    if inside:
        bg = background[image_id * sel_bg]
        row = image_id * img_h + i
        final_Ts[row, j] = T
        final_idx[row, j] = cur_idx
        out_img[row, j] = pix_out + T * bg
        out_depth[row, j] = depth_out


@wp.kernel
def _rasterize_fwd_cta64(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # gsplat-style 64-thread CTA / 4-pixels-per-thread blend (survey O4). Identical
    # blend math and batched image_id indexing to _rasterize_fwd; only the CTA
    # geometry differs. Launched one 64-thread block per (image, tile); dim =
    # B*num_tiles, so image_id = block // num_tiles. Each thread owns column
    # thread_x and PIXELS_PER_THREAD pixels strided down the tile by ROW_STRIDE
    # (gsplat's out_y[p] = tile_y*16 + thread_y + p*ROW_STRIDE), held in registers
    # (wp.vec4 lanes). done_mask carries one bit per owned pixel; the whole-tile
    # vote counts threads with all 4 pixels done (== gsplat's
    # __syncthreads_count(done_mask == ALL_DONE) >= CTA_SIZE).
    tile_g, tr = wp.tid()  # launch_tiled: tile_g = block index, tr = thread rank
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles

    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    thread_x = tr % BLOCK_WIDTH  # 0..15
    thread_y = tr // BLOCK_WIDTH  # 0..3

    out_x = tile_x * BLOCK_WIDTH + thread_x  # col (x), shared by all 4 pixels
    base_y = tile_y * BLOCK_WIDTH + thread_y
    px = wp.float32(out_x) + 0.5

    # per-pixel register state across the 4 strided rows this thread owns.
    py = wp.vec4(0.0, 0.0, 0.0, 0.0)
    T = wp.vec4(1.0, 1.0, 1.0, 1.0)
    pr = wp.vec4(0.0, 0.0, 0.0, 0.0)
    pg = wp.vec4(0.0, 0.0, 0.0, 0.0)
    pb = wp.vec4(0.0, 0.0, 0.0, 0.0)
    cur_idx = wp.vec4i(0, 0, 0, 0)

    # threads mapping outside the image stay live (needed for collective loads /
    # the block vote) but are marked done and never write an output pixel. A
    # thread can straddle the image edge, so each of its 4 pixels is checked.
    done_mask = wp.int32(0)
    if out_x >= img_w:
        done_mask = ALL_DONE
    for p in range(PIXELS_PER_THREAD):
        oy = base_y + p * ROW_STRIDE
        py[p] = wp.float32(oy) + 0.5
        if oy >= img_h:
            done_mask = done_mask | (1 << p)

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + CTA_SIZE - 1) // CTA_SIZE

    for b in range(num_batches):
        # whole-tile early-out vote: break once every thread has all 4 pixels done.
        done_count = wp.tile_sum(wp.tile(wp.where(done_mask == ALL_DONE, 1, 0)))
        if wp.tile_extract(done_count, 0) >= CTA_SIZE:
            break

        batch_start = range_start + b * CTA_SIZE
        # cooperatively stage a 64-wide batch of gaussian records into shared mem.
        id_tile = wp.tile_load(
            gaussian_ids_sorted, CTA_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        # broadcast/batched attribute index via modulo (see _rasterize_fwd note).
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(CTA_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(CTA_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )

        batch_size = wp.min(CTA_SIZE, range_end - batch_start)
        if done_mask != ALL_DONE:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                col = color_tile[t]
                dx = xy[0] - px
                for p in range(PIXELS_PER_THREAD):
                    if (done_mask & (1 << p)) == 0:
                        dy = xy[1] - py[p]
                        sigma = (
                            0.5 * (conic[0] * dx * dx + conic[2] * dy * dy)
                            + conic[1] * dx * dy
                        )
                        alpha = wp.min(0.999, opac * wp.exp(-sigma))
                        if sigma >= 0.0 and alpha >= 1.0 / 255.0:
                            next_T = T[p] * (1.0 - alpha)
                            if next_T <= 1e-4:
                                done_mask = done_mask | (1 << p)
                            else:
                                vis = alpha * T[p]
                                pr[p] = pr[p] + col[0] * vis
                                pg[p] = pg[p] + col[1] * vis
                                pb[p] = pb[p] + col[2] * vis
                                cur_idx[p] = batch_start + t
                                T[p] = next_T
                if done_mask == ALL_DONE:
                    break  # this thread's tile pixels are all saturated

    if out_x < img_w:
        bg = background[image_id * sel_bg]
        for p in range(PIXELS_PER_THREAD):
            oy = base_y + p * ROW_STRIDE
            if oy < img_h:
                row = image_id * img_h + oy
                final_Ts[row, out_x] = T[p]
                final_idx[row, out_x] = cur_idx[p]
                out_img[row, out_x] = wp.vec3(pr[p], pg[p], pb[p]) + T[p] * bg


@wp.kernel
def _rasterize_fwd_cta64_unrolled(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # Hand-unrolled twin of _rasterize_fwd_cta64 (survey O4b). Same 64-thread CTA
    # geometry, same batched image_id indexing, byte-identical blend math -- but the
    # four owned pixels are carried as FOUR INDEPENDENT SCALAR CHAINS (T0..T3, pr/pg/pb
    # 0..3, cur0..3, done0..3) instead of wp.vec4 lanes indexed by a `for p` loop.
    # Diagnosis (reports/phase5_o4.md addendum): Warp DOES unroll the `for p in range(4)`,
    # but because every iteration reuses the same Python temps (dy, sigma, alpha, next_T,
    # vis) and mutates a single wp.vec_t<4> aggregate via assign_inplace, its SSA pass
    # emits wp::where phi-merges that thread pixel p's values into pixel p+1 -- a false
    # serial dependency that stops nvcc pipelining the four expf chains. Distinct names
    # per pixel break that chain, giving nvcc four independent chains to overlap (this is
    # exactly what gsplat's block-scoped `const float dy = ...` per iteration achieves).
    tile_g, tr = wp.tid()  # launch_tiled: tile_g = block index, tr = thread rank
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles

    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    thread_x = tr % BLOCK_WIDTH  # 0..15
    thread_y = tr // BLOCK_WIDTH  # 0..3

    out_x = tile_x * BLOCK_WIDTH + thread_x  # col (x), shared by all 4 pixels
    base_y = tile_y * BLOCK_WIDTH + thread_y
    px = wp.float32(out_x) + 0.5

    # four strided rows this thread owns (out_y[p] = base_y + p*ROW_STRIDE)
    oy0 = base_y
    oy1 = base_y + ROW_STRIDE
    oy2 = base_y + 2 * ROW_STRIDE
    oy3 = base_y + 3 * ROW_STRIDE
    py0 = wp.float32(oy0) + 0.5
    py1 = wp.float32(oy1) + 0.5
    py2 = wp.float32(oy2) + 0.5
    py3 = wp.float32(oy3) + 0.5

    T0 = wp.float32(1.0)
    T1 = wp.float32(1.0)
    T2 = wp.float32(1.0)
    T3 = wp.float32(1.0)
    pr0 = wp.float32(0.0)
    pg0 = wp.float32(0.0)
    pb0 = wp.float32(0.0)
    pr1 = wp.float32(0.0)
    pg1 = wp.float32(0.0)
    pb1 = wp.float32(0.0)
    pr2 = wp.float32(0.0)
    pg2 = wp.float32(0.0)
    pb2 = wp.float32(0.0)
    pr3 = wp.float32(0.0)
    pg3 = wp.float32(0.0)
    pb3 = wp.float32(0.0)
    cur0 = wp.int32(0)
    cur1 = wp.int32(0)
    cur2 = wp.int32(0)
    cur3 = wp.int32(0)

    # per-pixel done flags: OOB column marks all four; each row checked separately
    # (a thread can straddle the image edge, some rows in / some out).
    col_oob = out_x >= img_w
    done0 = wp.bool(col_oob or (oy0 >= img_h))
    done1 = wp.bool(col_oob or (oy1 >= img_h))
    done2 = wp.bool(col_oob or (oy2 >= img_h))
    done3 = wp.bool(col_oob or (oy3 >= img_h))

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + CTA_SIZE - 1) // CTA_SIZE

    for b in range(num_batches):
        # whole-tile early-out vote: break once every thread has all 4 pixels done.
        all_done = done0 and done1 and done2 and done3
        done_count = wp.tile_sum(wp.tile(wp.where(all_done, 1, 0)))
        if wp.tile_extract(done_count, 0) >= CTA_SIZE:
            break

        batch_start = range_start + b * CTA_SIZE
        # cooperatively stage a 64-wide batch of gaussian records into shared mem.
        id_tile = wp.tile_load(
            gaussian_ids_sorted, CTA_SIZE, offset=batch_start, storage="shared"
        )
        xy_tile = wp.tile_load_indexed(
            xys, indices=id_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        conic_tile = wp.tile_load_indexed(
            conics, indices=id_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        cid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(CTA_SIZE, color_mod, dtype=wp.int32)
        )
        color_tile = wp.tile_load_indexed(
            colors, indices=cid_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )
        oid_tile = wp.tile_map(
            wp.mod, id_tile, wp.tile_full(CTA_SIZE, opac_mod, dtype=wp.int32)
        )
        opac_tile = wp.tile_load_indexed(
            opacities, indices=oid_tile, shape=(CTA_SIZE,), axis=0, storage="shared"
        )

        batch_size = wp.min(CTA_SIZE, range_end - batch_start)
        if not all_done:
            for t in range(batch_size):
                conic = conic_tile[t]
                xy = xy_tile[t]
                opac = opac_tile[t]
                col = color_tile[t]
                c0 = conic[0]
                c1 = conic[1]
                c2 = conic[2]
                col0 = col[0]
                col1 = col[1]
                col2 = col[2]
                xyy = xy[1]
                dx = xy[0] - px

                # --- pixel 0 (independent chain) ---
                if not done0:
                    dy0 = xyy - py0
                    sigma0 = 0.5 * (c0 * dx * dx + c2 * dy0 * dy0) + c1 * dx * dy0
                    alpha0 = wp.min(0.999, opac * wp.exp(-sigma0))
                    if sigma0 >= 0.0 and alpha0 >= 1.0 / 255.0:
                        next_T0 = T0 * (1.0 - alpha0)
                        if next_T0 <= 1e-4:
                            done0 = wp.bool(True)
                        else:
                            vis0 = alpha0 * T0
                            pr0 = pr0 + col0 * vis0
                            pg0 = pg0 + col1 * vis0
                            pb0 = pb0 + col2 * vis0
                            cur0 = batch_start + t
                            T0 = next_T0

                # --- pixel 1 (independent chain) ---
                if not done1:
                    dy1 = xyy - py1
                    sigma1 = 0.5 * (c0 * dx * dx + c2 * dy1 * dy1) + c1 * dx * dy1
                    alpha1 = wp.min(0.999, opac * wp.exp(-sigma1))
                    if sigma1 >= 0.0 and alpha1 >= 1.0 / 255.0:
                        next_T1 = T1 * (1.0 - alpha1)
                        if next_T1 <= 1e-4:
                            done1 = wp.bool(True)
                        else:
                            vis1 = alpha1 * T1
                            pr1 = pr1 + col0 * vis1
                            pg1 = pg1 + col1 * vis1
                            pb1 = pb1 + col2 * vis1
                            cur1 = batch_start + t
                            T1 = next_T1

                # --- pixel 2 (independent chain) ---
                if not done2:
                    dy2 = xyy - py2
                    sigma2 = 0.5 * (c0 * dx * dx + c2 * dy2 * dy2) + c1 * dx * dy2
                    alpha2 = wp.min(0.999, opac * wp.exp(-sigma2))
                    if sigma2 >= 0.0 and alpha2 >= 1.0 / 255.0:
                        next_T2 = T2 * (1.0 - alpha2)
                        if next_T2 <= 1e-4:
                            done2 = wp.bool(True)
                        else:
                            vis2 = alpha2 * T2
                            pr2 = pr2 + col0 * vis2
                            pg2 = pg2 + col1 * vis2
                            pb2 = pb2 + col2 * vis2
                            cur2 = batch_start + t
                            T2 = next_T2

                # --- pixel 3 (independent chain) ---
                if not done3:
                    dy3 = xyy - py3
                    sigma3 = 0.5 * (c0 * dx * dx + c2 * dy3 * dy3) + c1 * dx * dy3
                    alpha3 = wp.min(0.999, opac * wp.exp(-sigma3))
                    if sigma3 >= 0.0 and alpha3 >= 1.0 / 255.0:
                        next_T3 = T3 * (1.0 - alpha3)
                        if next_T3 <= 1e-4:
                            done3 = wp.bool(True)
                        else:
                            vis3 = alpha3 * T3
                            pr3 = pr3 + col0 * vis3
                            pg3 = pg3 + col1 * vis3
                            pb3 = pb3 + col2 * vis3
                            cur3 = batch_start + t
                            T3 = next_T3

                if done0 and done1 and done2 and done3:
                    break  # this thread's tile pixels are all saturated

    if out_x < img_w:
        bg = background[image_id * sel_bg]
        if oy0 < img_h:
            row0 = image_id * img_h + oy0
            final_Ts[row0, out_x] = T0
            final_idx[row0, out_x] = cur0
            out_img[row0, out_x] = wp.vec3(pr0, pg0, pb0) + T0 * bg
        if oy1 < img_h:
            row1 = image_id * img_h + oy1
            final_Ts[row1, out_x] = T1
            final_idx[row1, out_x] = cur1
            out_img[row1, out_x] = wp.vec3(pr1, pg1, pb1) + T1 * bg
        if oy2 < img_h:
            row2 = image_id * img_h + oy2
            final_Ts[row2, out_x] = T2
            final_idx[row2, out_x] = cur2
            out_img[row2, out_x] = wp.vec3(pr2, pg2, pb2) + T2 * bg
        if oy3 < img_h:
            row3 = image_id * img_h + oy3
            final_Ts[row3, out_x] = T3
            final_idx[row3, out_x] = cur3
            out_img[row3, out_x] = wp.vec3(pr3, pg3, pb3) + T3 * bg


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
    tight: int,
    opac_mod: int,
    tile_bounds_x: int,
    tile_bounds_y: int,
    num_tiles: int,
    bw: int,
    num_intersects: int | None = None,
) -> tuple[wp.array, wp.array, int]:
    """Shared intersection sort + tile-bin build (survey O1/O3).

    Emits per-(image, tile, gaussian) 64-bit keys, one global stable radix sort,
    and per-(image, tile) bin edges, into the signature-keyed grow-only scratch.
    Used by both the forward blend and the backward pass (which recomputes the
    identical sort from the saved cum_tiles_hit -- the
    deterministic sort reproduces the forward gaussian_ids_sorted / tile_bins so
    the saved final_idx stays valid). Returns (gaussian_ids, tile_bins,
    num_intersects).
    """
    tile_n_bits = _bits_for_count(num_tiles)
    image_n_bits = _bits_for_count(B)
    # Packed 32-bit key (phase 8r): pack iid|tile|quant_depth into one non-negative
    # int32 when the depth field would keep >=16 bits (i.e. image+tile <= 15 bits),
    # halving the radix sort's pass count (8->4) and bytes/pass. Otherwise fall back
    # to the 64-bit (tile_id<<32 | depth_bits) key. Packing is a deterministic
    # function of (B, num_tiles), so the backward's recomputed sort matches the
    # forward's bit-for-bit.
    upper_bits = image_n_bits + tile_n_bits
    depth_bits = 31 - upper_bits
    packed = _PACK_KEYS and depth_bits >= 16
    if not packed and upper_bits > 32:
        raise ValueError(
            f"batched intersection key overflow: image_n_bits({image_n_bits}) + "
            f"tile_n_bits({tile_n_bits}) = {upper_bits} > 32 "
            f"(batch B={B}, n_tiles={num_tiles}). Reduce batch size or resolution."
        )
    total = B * n
    bins_len = B * num_tiles

    # The one legitimate device->host sync: total intersection count over the batch.
    # (May be passed in by the forward launch when it already read it for the graph
    # eligibility check -- avoids a redundant readback on the gated fallback.)
    if num_intersects is None:
        num_intersects = int(cum_tiles_hit[total - 1 : total].numpy()[0])

    isect_dtype = wp.int32 if packed else wp.int64
    scratch = _get_scratch(
        device, (B, n, num_tiles), max(2 * num_intersects, 2), bins_len, isect_dtype
    )
    tile_bins = scratch["tile_bins"][:bins_len]
    tile_bins.zero_()

    if num_intersects > 0:
        isect_ids = scratch["isect_ids"][: 2 * num_intersects]
        gaussian_ids = scratch["gaussian_ids"][: 2 * num_intersects]
        if packed:
            # Device-side [dmin, dmax] over the emitted (radii>0) depths for the linear
            # depth quantization -- no host sync (result stays in a device array).
            depth_mm = scratch["depth_mm"]
            wp.launch(_seed_minmax, dim=B, inputs=[depth_mm], device=device)
            wp.launch(
                _depth_minmax,
                dim=(total + int(_MINMAX_CHUNK) - 1) // int(_MINMAX_CHUNK),
                inputs=[depths, radii, total, n, depth_mm],
                device=device,
            )
            wp.launch(
                _map_gaussian_to_intersects_p32,
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
                    opac_mod,
                    tight,
                    tile_n_bits,
                    depth_bits,
                    tile_bounds_x,
                    tile_bounds_y,
                    bw,
                ],
                outputs=[isect_ids, gaussian_ids],
                device=device,
            )
            wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, num_intersects)
            wp.launch(
                _get_tile_bin_edges_p32,
                dim=num_intersects,
                inputs=[num_intersects, isect_ids, num_tiles, tile_n_bits, depth_bits],
                outputs=[tile_bins],
                device=device,
            )
        else:
            wp.launch(
                _map_gaussian_to_intersects,
                dim=total,
                inputs=[
                    xys,
                    depths.view(wp.int32),
                    radii,
                    conics,
                    map_opacities,
                    cum_tiles_hit,
                    n,
                    opac_mod,
                    tight,
                    tile_n_bits,
                    tile_bounds_x,
                    tile_bounds_y,
                    bw,
                ],
                outputs=[isect_ids, gaussian_ids],
                device=device,
            )
            wp.utils.radix_sort_pairs(isect_ids, gaussian_ids, num_intersects)
            wp.launch(
                _get_tile_bin_edges,
                dim=num_intersects,
                inputs=[num_intersects, isect_ids, num_tiles, tile_n_bits],
                outputs=[tile_bins],
                device=device,
            )
        gaussian_ids = gaussian_ids[:num_intersects]
    else:
        gaussian_ids = scratch["gaussian_ids"][:1]
    return gaussian_ids, tile_bins, num_intersects


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
    tight: int,
    img_h: int,
    img_w: int,
    block_width: int,
    num_intersects: int | None = None,
) -> tuple[wp.array, wp.array, int, int, int]:
    """Shared blend-launcher preamble: tile geometry + the sort/bin build.

    ``B_geom`` is the geometry batch (how many distinct renders the sort/bin
    covers). ``num_intersects`` may be passed in when the caller already read it
    (graph-eligibility check), avoiding a redundant host readback. Returns
    (gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles).
    """
    assert block_width == 16, (
        "cooperative blend kernel is specialized for block_width=16"
    )
    tile_bounds_x = (img_w + block_width - 1) // block_width
    tile_bounds_y = (img_h + block_width - 1) // block_width
    num_tiles = tile_bounds_x * tile_bounds_y
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
        tight,
        map_opacities.shape[0],
        tile_bounds_x,
        tile_bounds_y,
        num_tiles,
        block_width,
        num_intersects,
    )
    return gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles


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
    block_width: int,
    tight: int,
    tile_bounds_x: int,
    tile_bounds_y: int,
    num_tiles: int,
    sel_bg: int,
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
) -> tuple[bool, int | None]:
    """Captured-graph forward path (phase 8s).

    Returns (handled, num_intersects). ``handled`` is True if a graph replayed the
    frame; when False the caller runs the plain path, reusing the returned
    num_intersects so the host readback is done exactly once.

    Reads num_intersects (the one legitimate host sync), buckets it, and either
    replays a cached CUDA graph or captures one covering the whole post-sync
    sequence: sentinel fill / depth minmax / map / sort / bin (device-count guarded)
    / blend. Byte-identical to the plain packed path -- the sort runs over the padded
    bucket with 0x7FFFFFFF sentinels sorted to the tail, and the bin kernel guards on
    the device-side real count (re-read at replay), so no sentinel is ever binned and
    the writes match _get_tile_bin_edges_p32 launched at dim=count.

    Falls back (returns False) unless: packed key layout applies, the default blend
    kernel is selected, and 0 < num_intersects < _GRAPH_THRESHOLD.
    """
    device = colors.device
    assert device is not None  # colors is always a live device array here
    # Never nest a capture inside an existing one: if the callback runs while the
    # stream is already being captured (e.g. an XLA command buffer, or a foreign
    # CUDA op mid-capture), our wp.capture_begin would conflict and corrupt the
    # context. Fall back to plain launches -- and let the caller do its own readback
    # (a host sync is illegal during capture, so we must not read the count here).
    if device.is_capturing:
        return False, None
    tile_n_bits = _bits_for_count(num_tiles)
    image_n_bits = _bits_for_count(B)
    depth_bits = 31 - (image_n_bits + tile_n_bits)
    packed = _PACK_KEYS and depth_bits >= 16
    total = B * n
    # single host readback (reused by the caller's plain path on a fallback).
    num_intersects = int(cum_tiles_hit[total - 1 : total].numpy()[0])
    if (
        _USE_CTA64
        or not packed
        or num_intersects <= 0
        or num_intersects >= _GRAPH_THRESHOLD
    ):
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
        # sentinel-fill the sort range; map overwrites [0, num_intersects) with real
        # keys (its write count is data-dependent via cum_tiles_hit, re-read at replay),
        # leaving the [count, bucket) tail as 0x7FFFFFFF -- sorts to the very tail.
        isect_ids[:bucket].fill_(0x7FFFFFFF)
        wp.launch(_seed_minmax, dim=B, inputs=[depth_mm], device=device)
        wp.launch(
            _depth_minmax,
            dim=(total + int(_MINMAX_CHUNK) - 1) // int(_MINMAX_CHUNK),
            inputs=[depths, radii, total, n, depth_mm],
            device=device,
        )
        wp.launch(
            _map_gaussian_to_intersects_p32,
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
                tight,
                tile_n_bits,
                depth_bits,
                tile_bounds_x,
                tile_bounds_y,
                block_width,
            ],
            outputs=[isect_ids[: 2 * bucket], gaussian_ids[: 2 * bucket]],
            device=device,
        )
        wp.utils.radix_sort_pairs(
            isect_ids[: 2 * bucket], gaussian_ids[: 2 * bucket], bucket
        )
        wp.launch(
            _get_tile_bin_edges_p32_dev,
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

    # .ptr is already an int for live device arrays (no int() -- the warp stubs
    # type it as optional, and the raw value keys the cache identically).
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
        tight,
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
        cast(wp.array, out_img).ptr,
        cast(wp.array, final_Ts).ptr,
        cast(wp.array, final_idx).ptr,
    )
    graph = _graph_cache.get(key)
    if graph is None:
        run()  # warm: load modules + allocate cub temp for `bucket` before capture
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
    block_width: int,
    tight: int,
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
) -> None:
    n = num_gaussians
    # B is recovered from an output shape (always full batch under expand_dims:
    # out_img collapses to (B*H, W)). N is static (vmap hides the batch axis from
    # the wrapper). The modulo divisor of a per-gaussian attribute is its own
    # leading dim (N if broadcast, B*N if batched); id % divisor yields the
    # correct, always in-bounds read index.
    B = out_img.shape[0] // img_h
    sel_bg = 1 if background.shape[0] > 1 else 0

    # Post-sync CUDA-graph fast path (phase 8s, opt-in). Handles the whole
    # map/sort/bin/blend as one cached graph replay; returns (False, num_intersects)
    # (falls back to the plain launches below, reusing the count) unless eligible
    # (packed key, default blend, count in range).
    ni_pre = None
    if _POSTSYNC_GRAPHS:
        assert block_width == 16
        tbx = (img_w + block_width - 1) // block_width
        tby = (img_h + block_width - 1) // block_width
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
            block_width,
            tight,
            tbx,
            tby,
            tbx * tby,
            sel_bg,
            final_Ts,
            final_idx,
            out_img,
        )
        if handled:
            return

    # Tile-count key emission uses map_opacities (the RAW opacity projection counted
    # with); the blend below uses opacities (ρ-compensated in antialiased mode). When
    # not antialiased the caller passes map_opacities == opacities, so this is a no-op.
    gaussian_ids, tile_bins, _num_isect, tile_bounds_x, num_tiles = _blend_setup(
        colors,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B,
        tight,
        img_h,
        img_w,
        block_width,
        ni_pre,
    )

    # One block per (image, tile); dim = B*num_tiles. The CTA variant is chosen by
    # the module constant _USE_CTA64 (survey O4): the 64-thread/4-pixel kernel
    # (gsplat geometry) or the 256-thread/1-pixel kernel. Both take identical
    # inputs/outputs and produce identical blend results.
    if _USE_CTA64:
        kernel = (
            _rasterize_fwd_cta64_unrolled if _CTA64_UNROLLED else _rasterize_fwd_cta64
        )
        block_dim = int(CTA_SIZE)
    else:
        kernel = _rasterize_fwd
        block_dim = int(BLOCK_SIZE)
    wp.launch_tiled(
        kernel,
        dim=[B * num_tiles],
        inputs=[
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
        ],
        outputs=[final_Ts, final_idx, out_img],
        block_dim=block_dim,
        device=colors.device,
    )


# graph_mode=NONE: the callable does a host readback (num_intersects) and
# data-dependent scratch allocation, which is not CUDA-graph capturable.
# vmap_method="expand_dims": native batching -- the callable launches a single
# grid over the whole batch and decodes the image id from the block rank / sort
# key, with per-input batched/broadcast indexing (projection intermediates
# arrive at full batch B; colors/opacities/background may be broadcast at size 1).
_rasterize_ffi = jax_callable(
    _rasterize_launch,
    num_outputs=3,
    graph_mode=JaxCallableGraphMode.NONE,
    vmap_method="expand_dims",
)


# --- Split-heavy-tile inference blend (phase 8t) ------------------------------
def _get_split_scratch(
    device: wp.Device | None, sig: tuple, seg_cap: int, split_cap: int, kcap: int
) -> dict:
    """Grow-only, signature-keyed scratch for the tile-split blend (mirrors
    _get_scratch's HWM policy). Buffers grow with the running-max segment/split
    counts (data-dependent per viewpoint) and are dropped when the workload
    signature (B, n, num_tiles) changes, so a big config never lingers into a
    small one."""
    key = "split:" + str(device)
    entry = _split_scratch_cache.get(key)
    if entry is None or entry["sig"] != sig or entry["kcap"] != kcap:
        _split_scratch_cache.pop(key, None)
        entry = {
            "sig": sig,
            "kcap": kcap,
            "seg_cap": 0,
            "split_cap": 0,
            "counters": wp.zeros(2, dtype=wp.int32, device=device),
        }
        _split_scratch_cache[key] = entry
    if entry["seg_cap"] < seg_cap:
        cap = max(int(seg_cap * _SCRATCH_HEADROOM) + 1, entry["seg_cap"])
        for name in ("seg_tile_g", "seg_start", "seg_end", "seg_split_idx", "seg_ord"):
            entry[name] = wp.empty(cap, dtype=wp.int32, device=device)
        entry["seg_cap"] = cap
    if entry["split_cap"] < split_cap:
        cap = max(int(split_cap * _SCRATCH_HEADROOM) + 1, entry["split_cap"])
        entry["split_tile_g"] = wp.empty(cap, dtype=wp.int32, device=device)
        entry["split_k"] = wp.empty(cap, dtype=wp.int32, device=device)
        entry["slot_c"] = wp.empty(
            cap * kcap * int(BLOCK_SIZE), dtype=wp.vec3, device=device
        )
        entry["slot_t"] = wp.empty(
            cap * kcap * int(BLOCK_SIZE), dtype=wp.float32, device=device
        )
        entry["split_cap"] = cap
    return entry


def _rasterize_split_launch(
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
    block_width: int,
    tight: int,
    # outputs: mirror _rasterize_launch's 3-output signature so the FFI batching /
    # shape convention is identical (inference discards final_Ts/final_idx). The split
    # kernels never write final_Ts/final_idx (they are allocated but unused, then DCE'd).
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
) -> None:
    """Load-balanced inference blend (phase 8t). Splits heavy tile bins across
    blocks via associative segment compositing, then merges per pixel. Same
    sort/bin as the plain path; only the blend scheduling differs. Image matches
    the unsplit path to a few ULP (segment adds reorder across boundaries only)."""
    n = num_gaussians
    device = colors.device
    B = out_img.shape[0] // img_h
    sel_bg = 1 if background.shape[0] > 1 else 0
    color_mod = colors.shape[0]
    opac_mod = opacities.shape[0]

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = _blend_setup(
        colors,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B,
        tight,
        img_h,
        img_w,
        block_width,
        None,
    )
    n_bins = B * num_tiles
    thr = _SPLIT_THRESHOLD
    kcap = _SPLIT_KCAP

    # Light/empty tiles (and heavy tiles' skip) -> direct write; covers every pixel
    # not owned by a split tile (heavy tiles return without writing; the composite
    # fills them). Together they tile the image exactly once.
    wp.launch_tiled(
        _rasterize_fwd_unsplit,
        dim=[n_bins],
        inputs=[
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            color_mod,
            opac_mod,
            sel_bg,
            thr,
            gaussian_ids,
            tile_bins,
            xys,
            conics,
            colors,
            opacities,
            background,
        ],
        outputs=[out_img],
        block_dim=int(BLOCK_SIZE),
        device=device,
    )

    if num_intersects <= thr:
        return  # no tile can exceed the split threshold -> nothing to split

    # Host worst-case bounds (num_intersects is already read): a split tile holds
    # > thr gaussians so #splits <= ni/thr; #segments = Σ ceil(count/thr) (kcap-capped)
    # <= ni/thr + #splits <= 2*ni/thr. Launch at these dims, guard on device counts.
    split_cap = min(num_intersects // thr + 1, n_bins) + 1
    seg_cap = 2 * (num_intersects // thr) + B + 4
    sc = _get_split_scratch(device, (B, n, num_tiles), seg_cap, split_cap, kcap)
    counters = sc["counters"]
    counters.zero_()

    wp.launch(
        _plan_tile_split,
        dim=n_bins,
        inputs=[
            tile_bins,
            n_bins,
            thr,
            kcap,
            sc["seg_cap"],
            sc["split_cap"],
            counters[0:1],
            counters[1:2],
            sc["split_tile_g"],
            sc["split_k"],
            sc["seg_tile_g"],
            sc["seg_start"],
            sc["seg_end"],
            sc["seg_split_idx"],
            sc["seg_ord"],
        ],
        device=device,
    )
    wp.launch_tiled(
        _rasterize_fwd_seg,
        dim=[sc["seg_cap"]],
        inputs=[
            color_mod,
            opac_mod,
            kcap,
            counters[1:2],
            sc["seg_tile_g"],
            sc["seg_start"],
            sc["seg_end"],
            sc["seg_split_idx"],
            sc["seg_ord"],
            gaussian_ids,
            xys,
            conics,
            colors,
            opacities,
            tile_bounds_x,
            num_tiles,
        ],
        outputs=[sc["slot_c"], sc["slot_t"]],
        block_dim=int(BLOCK_SIZE),
        device=device,
    )
    wp.launch(
        _composite_tile_split,
        dim=sc["split_cap"] * int(BLOCK_SIZE),
        inputs=[
            img_h,
            img_w,
            tile_bounds_x,
            num_tiles,
            sel_bg,
            kcap,
            counters[0:1],
            sc["split_tile_g"],
            sc["split_k"],
            sc["slot_c"],
            sc["slot_t"],
            background,
        ],
        outputs=[out_img],
        device=device,
    )


_rasterize_split_ffi = jax_callable(
    _rasterize_split_launch,
    num_outputs=3,
    graph_mode=JaxCallableGraphMode.NONE,
    vmap_method="expand_dims",
)


def _rasterize_split_call(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
    map_opacities: jax.Array | None = None,
) -> jax.Array:
    """Inference-only split blend entry (phase 8t): returns just the image (the
    final_Ts/final_idx outputs mirror _rasterize_call's shape convention and are
    discarded)."""
    if map_opacities is None:
        map_opacities = opacities
    _final_Ts, _final_idx, out_img = _rasterize_split_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        int(block_width),
        int(tight),
        output_dims=(H, W),
    )
    return out_img


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
    block_width: int,
    tight: int,
    # outputs
    final_Ts: wp.array2d[wp.float32],
    final_idx: wp.array2d[wp.int32],
    out_img: wp.array2d[wp.vec3],
    out_depth: wp.array2d[wp.float32],
) -> None:
    # Depth-augmented twin of _rasterize_launch (survey T2). Shares the exact
    # sort/bin (so the blend order matches the plain path bit-for-bit) and adds the
    # expected-depth output. Always the 256-thread kernel (the CTA64 experimental
    # variant is not depth-augmented; it is off by default).
    n = num_gaussians
    B = out_img.shape[0] // img_h
    sel_bg = 1 if background.shape[0] > 1 else 0

    gaussian_ids, tile_bins, _num_isect, tile_bounds_x, num_tiles = _blend_setup(
        colors,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B,
        tight,
        img_h,
        img_w,
        block_width,
    )

    wp.launch_tiled(
        _rasterize_fwd_depth,
        dim=[B * num_tiles],
        inputs=[
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
        ],
        outputs=[final_Ts, final_idx, out_img, out_depth],
        block_dim=int(BLOCK_SIZE),
        device=colors.device,
    )


_rasterize_depth_ffi = jax_callable(
    _rasterize_depth_launch,
    num_outputs=4,
    graph_mode=JaxCallableGraphMode.NONE,
    vmap_method="expand_dims",
)


# --- Backward pass --------------------------------------------------------------
# Plain per-pixel port of backward.cu::rasterize_bwd. One thread per pixel walks
# the tile's gaussians back-to-front from the saved final_idx down to the tile's
# first intersection, reconstructing T by dividing out (1-alpha) and accumulating
# parameter gradients with plain wp.atomic_add into the per-gaussian grad buffers.
# (A block-reduction variant exists behind SPLAX_BWD_REDUCE, below.)
#
# Residual flow (recompute-not-store design): the sort/bin structures (gaussian_ids_sorted,
# tile_bins) are NOT saved from the forward; they are recomputed here from the
# saved cum_tiles_hit via the shared _sort_and_bin (deterministic -> reproduces the
# forward order, so the saved final_Ts/final_idx line up). ``tight`` must match the
# forward so the key emission is identical.
#
# v_out_alpha is 0: the public rasterize returns only the image (no alpha channel),
# so the alpha cotangent is zero, as in the reference img.sum()-style grads.


@wp.kernel
def _rasterize_bwd_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # outputs (atomic-accumulated per gaussian)
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
):
    # launch_tiled over B_out*num_tiles blocks of BLOCK_SIZE threads: one thread == one
    # pixel. image_id (0..B_out) decodes the OUTPUT/cotangent image. The GEOMETRY
    # (sort, tile_bins, final_Ts, xys/conics) has its own batch B_geom which is either
    # B_out (batched: sel_geom=1 -> geom_image=image_id) or 1 (broadcast: sel_geom=0 ->
    # geom_image=0 shared -- e.g. a single shared render differentiated against B target
    # images). Grads are written at og = image_id*N*(1-sel_geom) + g: for batched geom
    # g already encodes the image (og==g, bit-identical to B=1); for broadcast geom g is
    # in [0,N) so the per-output-image slot is image_id*N + g. JAX reduces broadcast
    # inputs over the batch axis afterwards.
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = image_id * sel_geom
    og_base = image_id * num_gaussians * (1 - sel_geom)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj
    if (i >= img_h) or (j >= img_w):
        return  # pixel outside the image writes nothing

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return  # empty tile

    frow = geom_image * img_h + i  # final_Ts/final_idx are geometry outputs (B_geom*H)
    bin_final = final_idx[frow, j]
    t_final = final_Ts[frow, j]  # transmittance after the last contributing gaussian
    T = t_final
    v_out = v_out_img[
        (image_id * img_h + i) % vout_rows, j
    ]  # cotangent row (B_out*H) or bcast
    bg = background[image_id * sel_bg]  # sel_bg==0 -> row 0 (broadcast background)
    buffer = wp.vec3(0.0, 0.0, 0.0)

    # Walk back-to-front from the last contributor to the tile's first gaussian.
    # Culled contributors (sigma<0 / alpha<1/255) are skipped exactly as in the
    # forward, so T reconstruction and the contributing set match the fwd blend.
    for idx in range(bin_final, range_start - 1, -1):
        g = gaussian_ids_sorted[idx]
        conic = conics[g]
        xy = xys[g]
        opac = opacities[g % opac_mod]
        dx = xy[0] - px
        dy = xy[1] - py
        sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
        if sigma < 0.0:
            continue
        vis = wp.exp(-sigma)
        alpha = wp.min(0.99, opac * vis)
        if alpha < 1.0 / 255.0:
            continue

        ra = 1.0 / (1.0 - alpha)
        T = T * ra
        fac = alpha * T
        color = colors[g % color_mod]
        og = og_base + g  # per-output-image grad slot (og==g for batched geometry)

        wp.atomic_add(v_colors, og, v_out * fac)

        v_alpha = float(0.0)
        v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
        v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
        v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
        # background contribution (v_out_alpha == 0 -> no alpha term)
        v_alpha += -t_final * ra * bg[0] * v_out[0]
        v_alpha += -t_final * ra * bg[1] * v_out[1]
        v_alpha += -t_final * ra * bg[2] * v_out[2]

        buffer = buffer + color * fac

        v_sigma = -opac * vis * v_alpha
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
                v_sigma * (conic[0] * dx + conic[1] * dy),
                v_sigma * (conic[1] * dx + conic[2] * dy),
            ),
        )
        wp.atomic_add(v_opacity, og, vis * v_alpha)


# --- Depth-augmented backward (survey T2) -------------------------------------
# Twin of _rasterize_bwd_kernel (plain per-pixel atomics) that additionally routes
# the expected-depth cotangent. The depth channel is handled EXACTLY like a colour
# channel (gsplat renders depth as an extra alpha-composited channel): it contributes
# to v_alpha -- hence to v_sigma and thus v_conic/v_xy/v_opacity -- and produces a
# per-gaussian depth cotangent v_depths (the weight wᵢ times the pixel depth
# cotangent). v_depths then flows through project's backward (which already consumes
# v_depths) to means/scales/quats/viewmat. Background depth is 0, so the
# depth channel has no t_final·bg term. Colour-grad math is byte-identical to
# _rasterize_bwd_kernel; only the depth additions are new.
@wp.kernel
def _rasterize_bwd_depth_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # outputs (atomic-accumulated per gaussian)
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_depths: wp.array[wp.float32],
):
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = image_id * sel_geom
    og_base = image_id * num_gaussians * (1 - sel_geom)
    tile_x = tile_local % tile_bounds_x
    tile_y = tile_local // tile_bounds_x
    li = tr // BLOCK_WIDTH
    lj = tr % BLOCK_WIDTH
    i = tile_y * BLOCK_WIDTH + li
    j = tile_x * BLOCK_WIDTH + lj
    if (i >= img_h) or (j >= img_w):
        return

    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    tile_range = tile_bins[geom_image * num_tiles + tile_local]
    range_start = tile_range[0]
    range_end = tile_range[1]
    if range_end <= range_start:
        return

    frow = geom_image * img_h + i
    bin_final = final_idx[frow, j]
    t_final = final_Ts[frow, j]
    T = t_final
    v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
    v_outd = v_out_depth[(image_id * img_h + i) % vdepth_rows, j]
    bg = background[image_id * sel_bg]
    buffer = wp.vec3(0.0, 0.0, 0.0)
    dbuffer = wp.float32(0.0)

    for idx in range(bin_final, range_start - 1, -1):
        g = gaussian_ids_sorted[idx]
        conic = conics[g]
        xy = xys[g]
        opac = opacities[g % opac_mod]
        dx = xy[0] - px
        dy = xy[1] - py
        sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
        if sigma < 0.0:
            continue
        vis = wp.exp(-sigma)
        alpha = wp.min(0.99, opac * vis)
        if alpha < 1.0 / 255.0:
            continue

        ra = 1.0 / (1.0 - alpha)
        T = T * ra
        fac = alpha * T
        color = colors[g % color_mod]
        d = depths[g]
        og = og_base + g

        wp.atomic_add(v_colors, og, v_out * fac)
        wp.atomic_add(v_depths, og, v_outd * fac)

        v_alpha = float(0.0)
        v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
        v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
        v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
        # depth channel (background depth = 0, so no t_final·bg term)
        v_alpha += (d * T - dbuffer * ra) * v_outd
        # colour background contribution (v_out_alpha == 0 -> no alpha term)
        v_alpha += -t_final * ra * bg[0] * v_out[0]
        v_alpha += -t_final * ra * bg[1] * v_out[1]
        v_alpha += -t_final * ra * bg[2] * v_out[2]

        buffer = buffer + color * fac
        dbuffer = dbuffer + d * fac

        v_sigma = -opac * vis * v_alpha
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
                v_sigma * (conic[0] * dx + conic[1] * dy),
                v_sigma * (conic[1] * dx + conic[2] * dy),
            ),
        )
        wp.atomic_add(v_opacity, og, vis * v_alpha)


# --- Block-reduction backward variant (verified negative, kept as A/B switch) --
# gsplat's training-specific optimization: reduce the per-pixel gradient
# contributions across the block *before* the atomic, so one atomic is issued per
# gaussian per tile instead of one per pixel (256x fewer atomics), removing the
# 256-way atomic contention hotspot. warp-lang exposes no warp-scoped shuffle, so
# the reduction is a whole-block wp.tile_sum (one block barrier per gaussian). All
# 256 threads walk the tile's gaussians in lockstep down to the *block-max*
# final_idx (wp.tile_max); a thread whose own pixel's walk has passed its final_idx
# (or is out of image / culled) contributes a zero vec9. Mechanics mirror the
# viewmat tile_sum reduction in _project.py, but here per-pixel early termination
# makes the block barrier a net loss (reports/phase6g_lego_reverification.md).
# Gradient math is byte-identical to _rasterize_bwd_kernel.
@wp.kernel
def _rasterize_bwd_reduce_kernel(
    img_h: wp.int32,
    img_w: wp.int32,
    tile_bounds_x: wp.int32,
    num_tiles: wp.int32,
    num_gaussians: wp.int32,
    sel_geom: wp.int32,
    color_mod: wp.int32,
    opac_mod: wp.int32,
    sel_bg: wp.int32,
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
    # outputs (one atomic per gaussian per tile from thread 0)
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
):
    tile_g, tr = wp.tid()
    image_id = tile_g // num_tiles
    tile_local = tile_g % num_tiles
    geom_image = (
        image_id * sel_geom
    )  # geometry image (== image_id when batched, else 0)
    og_base = image_id * num_gaussians * (1 - sel_geom)  # per-output grad slot base
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
        return  # empty tile -- uniform across the whole block, safe collective skip

    valid = (i < img_h) and (j < img_w)
    px = wp.float32(j) + 0.5
    py = wp.float32(i) + 0.5

    # Out-of-image lanes stay live (contribute zeros) so every lane reaches every
    # collective below; their bin_final = range_start-1 keeps them out of the max.
    bin_final = range_start - 1
    t_final = float(0.0)
    v_out = wp.vec3(0.0, 0.0, 0.0)
    if valid:
        frow = geom_image * img_h + i
        bin_final = final_idx[frow, j]
        t_final = final_Ts[frow, j]
        v_out = v_out_img[(image_id * img_h + i) % vout_rows, j]
    T = t_final
    bg = background[image_id * sel_bg]
    buffer = wp.vec3(0.0, 0.0, 0.0)

    # Block-max final_idx bounds the lockstep walk (uniform for the whole block).
    bmax = wp.tile_extract(wp.tile_max(wp.tile(bin_final)), 0)

    for idx in range(bmax, range_start - 1, -1):
        g = gaussian_ids_sorted[idx]
        c = vec9(0.0)
        # A lane contributes only while idx is within its own per-pixel walk.
        if valid and (idx <= bin_final):
            conic = conics[g]
            xy = xys[g]
            opac = opacities[g % opac_mod]
            dx = xy[0] - px
            dy = xy[1] - py
            sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
            if sigma >= 0.0:
                vis = wp.exp(-sigma)
                alpha = wp.min(0.99, opac * vis)
                if alpha >= 1.0 / 255.0:
                    ra = 1.0 / (1.0 - alpha)
                    T = T * ra
                    fac = alpha * T
                    color = colors[g % color_mod]
                    v_alpha = float(0.0)
                    v_alpha += (color[0] * T - buffer[0] * ra) * v_out[0]
                    v_alpha += (color[1] * T - buffer[1] * ra) * v_out[1]
                    v_alpha += (color[2] * T - buffer[2] * ra) * v_out[2]
                    v_alpha += -t_final * ra * bg[0] * v_out[0]
                    v_alpha += -t_final * ra * bg[1] * v_out[1]
                    v_alpha += -t_final * ra * bg[2] * v_out[2]
                    buffer = buffer + color * fac
                    v_sigma = -opac * vis * v_alpha
                    c[0] = v_out[0] * fac
                    c[1] = v_out[1] * fac
                    c[2] = v_out[2] * fac
                    c[3] = 0.5 * v_sigma * dx * dx
                    c[4] = v_sigma * dx * dy
                    c[5] = 0.5 * v_sigma * dy * dy
                    c[6] = v_sigma * (conic[0] * dx + conic[1] * dy)
                    c[7] = v_sigma * (conic[1] * dx + conic[2] * dy)
                    c[8] = vis * v_alpha
        s = wp.tile_extract(wp.tile_sum(wp.tile(c, preserve_type=True)), 0)
        if tr == 0:
            og = og_base + g  # og==g for batched geometry
            wp.atomic_add(v_colors, og, wp.vec3(s[0], s[1], s[2]))
            wp.atomic_add(v_conic, og, wp.vec3(s[3], s[4], s[5]))
            wp.atomic_add(v_xy, og, wp.vec2(s[6], s[7]))
            wp.atomic_add(v_opacity, og, s[8])


# Module switch: plain per-pixel atomics (shipped default, benchmarked winner) vs
# block tile_sum reduction. Set SPLAX_BWD_REDUCE=1 to select the reduction for A/B
# benchmarking.
_BWD_REDUCE = os.environ.get("SPLAX_BWD_REDUCE", "0") == "1"


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
    block_width: int,
    tight: int,
    # outputs
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
) -> None:
    n = num_gaussians
    # Two batch sizes. B_out (from the grad OUTPUT v_xy, always full batch under
    # expand_dims: v_xy collapses to (B_out*N,)) is how many images the blend walks
    # and how the per-view grads are laid out. B_geom (from the GEOMETRY residual
    # cum_tiles_hit) is how many distinct renders the sort/bin covers. They agree in
    # the multi-view regime (batched viewmat -> batched geometry), but a shared
    # render differentiated against B target images gives B_geom=1 < B_out: geometry
    # is read from image 0 and the per-output grads land in image_id's slot. A
    # residual/cotangent is not a reliable B_out signal (a cotangent can arrive
    # broadcast if the loss treats views identically), which is why B_out comes from
    # an output. For B=1 every path reduces to the single-image kernel.
    B_out = v_xy.shape[0] // n
    B_geom = cum_tiles_hit.shape[0] // n
    sel_geom = 1 if B_geom > 1 else 0
    sel_bg = 1 if background.shape[0] > 1 else 0
    # v_out_img is a COTANGENT: it arrives batched (B_out*H) for a view-dependent loss
    # but BROADCAST (H) for a view-independent one (e.g. a linear loss like img.sum(),
    # whose image cotangent is the same for every view). Indexing its row by modulo of
    # its own row count reads image_id*H+i when batched and i when broadcast -- both in
    # bounds.
    vout_rows = v_out_img.shape[0]

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = _blend_setup(
        colors,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B_geom,
        tight,
        img_h,
        img_w,
        block_width,
    )

    # atomics accumulate -> outputs must start at zero (memset equivalent).
    v_colors.zero_()
    v_opacity.zero_()
    v_xy.zero_()
    v_conic.zero_()
    if num_intersects == 0:
        return
    kernel = _rasterize_bwd_reduce_kernel if _BWD_REDUCE else _rasterize_bwd_kernel
    wp.launch_tiled(
        kernel,
        dim=[B_out * num_tiles],
        inputs=[
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
        ],
        outputs=[v_xy, v_conic, v_colors, v_opacity],
        block_dim=int(BLOCK_SIZE),
        device=colors.device,
    )


# graph_mode=NONE: host readback + data-dependent sort scratch (like the forward).
# vmap_method="expand_dims": batch-native backward. Under jax.vmap the launch
# recovers B from the collapsed (B*H, W) cotangent, runs one global sort and
# one block per (image, tile), and writes per-view grads at the flat gaussian id;
# JAX reduces broadcast (shared-gaussian) inputs over the batch axis. B=1 unchanged.
_rasterize_bwd_ffi = jax_callable(
    _rasterize_bwd_launch,
    num_outputs=4,
    graph_mode=JaxCallableGraphMode.NONE,
    vmap_method="expand_dims",
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
    block_width: int,
    tight: int,
    # outputs
    v_colors: wp.array[wp.vec3],
    v_opacity: wp.array[wp.float32],
    v_xy: wp.array[wp.vec2],
    v_conic: wp.array[wp.vec3],
    v_depths: wp.array[wp.float32],
) -> None:
    # Depth-augmented twin of _rasterize_bwd_launch (survey T2). Uses only the plain
    # per-pixel atomic kernel (the block-reduction variant is not depth-augmented).
    n = num_gaussians
    B_out = v_xy.shape[0] // n
    B_geom = cum_tiles_hit.shape[0] // n
    sel_geom = 1 if B_geom > 1 else 0
    sel_bg = 1 if background.shape[0] > 1 else 0
    vout_rows = v_out_img.shape[0]
    vdepth_rows = v_out_depth.shape[0]

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = _blend_setup(
        colors,
        xys,
        depths,
        radii,
        conics,
        map_opacities,
        cum_tiles_hit,
        n,
        B_geom,
        tight,
        img_h,
        img_w,
        block_width,
    )

    v_colors.zero_()
    v_opacity.zero_()
    v_xy.zero_()
    v_conic.zero_()
    v_depths.zero_()
    if num_intersects == 0:
        return
    wp.launch_tiled(
        _rasterize_bwd_depth_kernel,
        dim=[B_out * num_tiles],
        inputs=[
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
        ],
        outputs=[v_xy, v_conic, v_colors, v_opacity, v_depths],
        block_dim=int(BLOCK_SIZE),
        device=colors.device,
    )


_rasterize_bwd_depth_ffi = jax_callable(
    _rasterize_bwd_depth_launch,
    num_outputs=5,
    graph_mode=JaxCallableGraphMode.NONE,
    vmap_method="expand_dims",
)


def _rasterize_call(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # map_opacities: RAW opacity for the tight tile-count key emission (must match the
    # projection that produced cum_tiles_hit). Defaults to opacities -> byte-identical
    # to the plain path. In antialiased mode opacities is ρ-compensated (blend) while
    # map_opacities stays raw (count), so the key total still matches.
    if map_opacities is None:
        map_opacities = opacities
    out = _rasterize_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        int(block_width),
        int(tight),
        output_dims=(H, W),
    )
    # the FFI returns exactly (final_Ts, final_idx, out_img); typed as a Sequence.
    return cast(tuple[jax.Array, jax.Array, jax.Array], out)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13))
def _rasterize_diff(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
) -> jax.Array:
    _final_Ts, _final_idx, out_img = _rasterize_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        block_width,
        tight,
        map_opacities,
    )
    return out_img


def _rasterize_diff_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    final_Ts, final_idx, out_img = _rasterize_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        block_width,
        tight,
        map_opacities,
    )
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return out_img, residuals


def _rasterize_diff_bwd(
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
    residuals: tuple[jax.Array, ...],
    v_img: jax.Array,
) -> tuple[jax.Array | None, ...]:
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    v_colors, v_opacity, v_xy, v_conic = _rasterize_bwd_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        int(n),
        int(H),
        int(W),
        int(block_width),
        int(tight),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    # cotangents for (colors, opacities, map_opacities, background, xys, depths, radii,
    # conics, cum_tiles_hit). map_opacities feeds only the integer tile-count emission
    # (non-diff, like radii/cum), so it gets None -- exactly how the opacity's old
    # tile-count role was already non-differentiated. background/depths/radii/cum are
    # non-diff (match the reference CUDA backward). The ρ factor on the opacity grad and ρ's own grad
    # chain are handled outside rasterize: opacities here is the ρ-compensated blend
    # opacity, and JAX splits its cotangent back to raw opacity (×ρ) and to conics
    # (via the compensation) at the training.render level.
    return (v_colors, v_opacity, None, None, v_xy, None, None, v_conic, None)


_rasterize_diff.defvjp(_rasterize_diff_fwd, _rasterize_diff_bwd)


def rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    block_width: int,
    tight: bool = False,
    map_opacities: jax.Array | None = None,
) -> jax.Array:
    """Warp rasterization: blends projected gaussians into the image.

    Returns the rendered (H, W, 3) image.

    Differentiable w.r.t. colors, opacities, xys, conics via
    jax.custom_vjp (background/depths/radii/cum_tiles_hit are non-diff). When no
    gradient is requested the primal is identical to the
    forward-only path -- custom_vjp only intercepts differentiation, and the
    final_Ts/final_idx the FFI already computed are simply discarded, so pure
    inference does not regress.

    ``tight=True`` selects the AccuTile ellipse-walk key emission (survey O6); it
    MUST match the projection that produced ``cum_tiles_hit`` (i.e. one done with
    ``opacities`` supplied), otherwise the per-gaussian sort offsets are corrupt.
    Default ``False`` is the legacy isotropic-bbox emission.

    ``map_opacities`` (anti-aliased mode) is the RAW opacity used for the tight
    tile-count key emission; it must equal the opacity projection counted with.
    ``opacities`` is then the ρ-compensated opacity used only in the blend. Default
    ``None`` sets map_opacities = opacities, i.e. byte-identical to the plain path.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    return _rasterize_diff(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
        int(block_width),
        bool(tight),
    )


# --- Depth-augmented differentiable rasterize (survey T2) ----------------------
# Same as _rasterize_call but returns (image, expected_depth) from the depth FFI.
def _rasterize_depth_call(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if map_opacities is None:
        map_opacities = opacities
    out = _rasterize_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        int(n),
        int(H),
        int(W),
        int(block_width),
        int(tight),
        output_dims=(H, W),
    )
    # the FFI returns exactly (final_Ts, final_idx, out_img, out_depth); typed as a Sequence.
    return cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array], out)


@partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13))
def _rasterize_depth_diff(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
) -> tuple[jax.Array, jax.Array]:
    _final_Ts, _final_idx, out_img, out_depth = _rasterize_depth_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        block_width,
        tight,
        map_opacities,
    )
    return out_img, out_depth


def _rasterize_depth_diff_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    map_opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    final_Ts, final_idx, out_img, out_depth = _rasterize_depth_call(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        n,
        H,
        W,
        block_width,
        tight,
        map_opacities,
    )
    residuals = (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    )
    return (out_img, out_depth), residuals


def _rasterize_depth_diff_bwd(
    n: int,
    H: int,
    W: int,
    block_width: int,
    tight: bool,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array | None, ...]:
    (
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        final_Ts,
        final_idx,
    ) = residuals
    v_img, v_depth_img = cotangents
    v_colors, v_opacity, v_xy, v_conic, v_depths = _rasterize_bwd_depth_ffi(
        colors,
        opacities.reshape(n),
        map_opacities.reshape(n),
        background.reshape(1, 3),
        xys,
        depths.reshape(n),
        radii.reshape(n).astype(jnp.int32),
        conics,
        cum_tiles_hit.reshape(n).astype(jnp.int32),
        final_Ts,
        final_idx,
        v_img,
        v_depth_img,
        int(n),
        int(H),
        int(W),
        int(block_width),
        int(tight),
        output_dims=n,
    )
    v_opacity = v_opacity.reshape(opacities.shape)
    v_depths = v_depths.reshape(depths.shape)
    # cotangents for (colors, opacities, map_opacities, background, xys, depths,
    # radii, conics, cum_tiles_hit). Unlike the plain rasterize, depths now carries a
    # NONZERO cotangent (v_depths) -- the whole point of T2: it flows through project's
    # backward to means/scales/quats/viewmat.
    return (v_colors, v_opacity, None, None, v_xy, v_depths, None, v_conic, None)


_rasterize_depth_diff.defvjp(_rasterize_depth_diff_fwd, _rasterize_depth_diff_bwd)


def rasterize_depth(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    block_width: int,
    tight: bool = False,
    map_opacities: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Depth-augmented rasterize (survey T2): returns (image, expected_depth).

    Identical to :func:`rasterize` but additionally renders the alpha-blended
    expected-depth map D(p) = Σ wᵢ dᵢ (wᵢ = αᵢ Tᵢ, the same visibility weight as the
    colour blend; dᵢ the per-gaussian camera-space depth). Differentiable w.r.t. the
    same inputs as :func:`rasterize`, and additionally routes a nonzero cotangent for
    ``depths`` (which flows through :func:`splax.project`'s backward to the gaussian
    geometry / camera pose). Used for COLMAP sparse-point depth regularization
    (gsplat ``depth_loss``). This is a SEPARATE path from :func:`rasterize`; the
    plain render never pays for the extra depth channel.
    """
    n = colors.shape[0]
    H, W = img_shape
    if map_opacities is None:
        map_opacities = opacities
    return _rasterize_depth_diff(
        colors,
        opacities,
        map_opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        int(n),
        int(H),
        int(W),
        int(block_width),
        bool(tight),
    )
