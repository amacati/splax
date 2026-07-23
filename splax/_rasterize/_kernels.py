"""Warp rasterization kernels and their JAX FFI callables.

Blend kernels are available with and without depth support, their backward implementation, and
shared kernels for sorting and binning.

Kernel launch functions are wrapped into JAX FFI callables that the API layer in
``splax._rasterize`` composes with ``jax.custom_vjp``.

Batching is native. Under jax.vmap the callable launches a single grid over the whole batch. The
image index is decoded from the block rank, packed into the sort key, and used to offset per-image
bin edges, outputs, and backgrounds. Because of the host readback and data-dependent scratch, the
forward callable is not CUDA-graph capturable.
"""

from __future__ import annotations

import warp as wp
from warp import JaxCallableGraphMode, jax_callable

from splax._batching import nested_vmap
from splax._cache import cached_launch
from splax._intersect import BLOCK_SIZE, BLOCK_WIDTH
from splax._rasterize._sort import sort_and_bin

wp.set_module_options({"fast_math": True})  # Fastmath significantly accelerates the kernels.

# Gaussian records consist of xy (2), opacity (1) and conic (3). Packing everything into one vector
# allows single shared writes/reads per gaussian in the blend loop.
_vec6 = wp.types.vector(length=6, dtype=wp.float32)
# When including depth, the record grows by one to xy (2), opacity (1), conic (3), depth (1).
_vec7 = wp.types.vector(length=7, dtype=wp.float32)


# region forward kernels


def _rasterize_warp(
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
    n = num_gaussians
    B = out_img.shape[0] // img_h  # out_img collapses to (B*H, W) with batching, so we recover B
    sel_bg = background.shape[0] > 1

    # sorting must use the same map_opacities as the forward blend to reproduce the order and index
    gaussian_ids, tile_bins, _, tile_bounds_x, num_tiles = sort_and_bin(
        xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B, img_h, img_w
    )
    cached_launch(
        _rasterize_kernel,
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
        _rasterize_warp,
        num_outputs=3,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=9,
    name="rasterize",
)


@wp.kernel
def _rasterize_kernel(
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
    """Rasterization kernel with cooperative shared-memory blend."""
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

    # Threads mapping outside the image stay live for the collective loads and the block vote but
    # are marked done and never write an output pixel.
    inside = (i < img_h) and (j < img_w)
    done = wp.bool(not inside)

    tile_range = tile_bins[tile_g]
    range_start = tile_range[0]
    range_end = tile_range[1]
    num_batches = (range_end - range_start + BLOCK_SIZE - 1) // BLOCK_SIZE

    T = wp.float32(1.0)
    cur_idx = wp.int32(0)
    pix_out = wp.vec3(0.0, 0.0, 0.0)

    # Colors are staged in their own tile so a rejected gaussian only loads its geometry record
    geo_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=_vec6, storage="shared")
    color_tile = wp.tile_empty(shape=BLOCK_SIZE, dtype=wp.vec3, storage="shared")
    done_tile = wp.tile_zeros(shape=1, dtype=wp.int32, storage="shared")
    counted = wp.bool(False)

    # We chunk up all Gaussians in this tile into BLOCK_SIZE batches
    for b in range(num_batches):
        # If every thread in the block is done, we break early and skip the rest of the Gaussians.
        # Double-acts as a sync barrier to ensure previous reads are complete before the next batch
        wp.tile_scatter_add(done_tile, 0, 1, done and not counted)
        counted = done
        if done_tile[0] >= BLOCK_SIZE:
            break

        # Per-thread gather of one gaussian record with batch-aware indexing
        #
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

        # The last batch gets clamped to the final intersection
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


def _rasterize_depth_warp(
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
    # Depth-augmented version of _rasterize_warp
    n = num_gaussians
    B = out_img.shape[0] // img_h
    sel_bg = background.shape[0] > 1

    gaussian_ids, tile_bins, _, tile_bounds_x, num_tiles = sort_and_bin(
        xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B, img_h, img_w
    )

    cached_launch(
        _rasterize_depth_kernel,
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
        _rasterize_depth_warp,
        num_outputs=4,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=9,
    name="rasterize_depth",
)


@wp.kernel
def _rasterize_depth_kernel(
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
    """Separate depth-augmented rasterization kernel to only compute expected depth if required.

    Additional memory loads are expensive, so we avoid paying them in the default rasterizer.
    """
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
                # Also compute the expected depth with the alpha blend weights
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


# region backward kernels


def _rasterize_bwd_warp(
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

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = sort_and_bin(
        xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B_geom, img_h, img_w
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
        cached_launch(
            _rasterize_bwd_agg_kernel,
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
    cached_launch(
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
        _rasterize_bwd_warp,
        num_outputs=4,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=12,
    name="rasterize_bwd",
)


# Backward pass. A staged lockstep walk mirroring the forward blend. All 256
# threads of a tile block walk the sorted range back to front in shared-staged
# batches, starting at the block maximum of the pixels' final_idx. Each pixel
# reconstructs T by dividing out (1 - alpha) and accumulates parameter gradients
# with per-lane atomics, guarded per pixel so only indices at or below its own
# final_idx contribute, with the same sigma and alpha culling as the forward.
#
# The sort and bin structures are not saved from the forward. They are recomputed
# from the saved cum_tiles_hit via the shared sort_and_bin. The sort is
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

                color = color_tile[t]
                og = og_base + id_tile[t]
                v_rgb, v_con, v_xyl, v_op, T, buffer = _blend_color_vjp(
                    dx, dy, s[3], s[4], s[5], s[2], vis, alpha, color, T, buffer, t_final, bg, v_out
                )
                wp.atomic_add(v_colors, og, v_rgb)
                # Where the alpha clamp is active alpha is constant, so the sigma and opacity paths
                # carry no gradient and their atomics are skipped.
                if s[2] * vis <= 0.999:
                    wp.atomic_add(v_conic, og, v_con)
                    wp.atomic_add(v_xy, og, v_xyl)
                    wp.atomic_add(v_opacity, og, v_op)


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
def _rasterize_bwd_agg_kernel(
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
            color = colors[g % color_mod]
            v_rgb, v_con, v_xyl, v_op, T, buffer = _blend_color_vjp(
                dx,
                dy,
                conic[0],
                conic[1],
                conic[2],
                opac,
                vis,
                alpha,
                color,
                T,
                buffer,
                t_final,
                bg,
                v_out,
            )

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


def _rasterize_bwd_depth_warp(
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

    gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles = sort_and_bin(
        xys, depths, radii, conics, map_opacities, cum_tiles_hit, n, B_geom, img_h, img_w
    )

    v_colors.zero_()
    v_opacity.zero_()
    v_xy.zero_()
    v_conic.zero_()
    v_depths.zero_()
    if num_intersects == 0:
        return
    # Kernel choice by mean tile range, as in _rasterize_bwd_warp.
    if num_intersects < B_geom * num_tiles * int(BLOCK_SIZE):
        cached_launch(
            _rasterize_bwd_depth_agg_kernel,
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
    cached_launch(
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
        _rasterize_bwd_depth_warp,
        num_outputs=5,
        graph_mode=JaxCallableGraphMode.NONE,
        vmap_method="expand_dims",
    ),
    n_arrays=13,
    name="rasterize_bwd_depth",
)


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

                cd = cd_tile[t]
                og = og_base + id_tile[t]
                color = wp.vec3(cd[0], cd[1], cd[2])
                v_rgb, v_dep, v_con, v_xyl, v_op, T, buffer, dbuffer = _blend_depth_vjp(
                    dx,
                    dy,
                    s[3],
                    s[4],
                    s[5],
                    s[2],
                    vis,
                    alpha,
                    color,
                    cd[3],
                    T,
                    buffer,
                    dbuffer,
                    t_final,
                    bg,
                    v_out,
                    v_outd,
                )
                wp.atomic_add(v_colors, og, v_rgb)
                wp.atomic_add(v_depths, og, v_dep)
                # Where the alpha clamp is active alpha is constant, so the sigma and opacity paths
                # carry no gradient and their atomics are skipped.
                if s[2] * vis <= 0.999:
                    wp.atomic_add(v_conic, og, v_con)
                    wp.atomic_add(v_xy, og, v_xyl)
                    wp.atomic_add(v_opacity, og, v_op)


# Depth-augmented twin of _rasterize_bwd_agg_kernel. Walk, aggregation, and
# color-grad math are identical, with the depth channel handled as in
# _rasterize_bwd_depth_kernel.
@wp.kernel
def _rasterize_bwd_depth_agg_kernel(
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
            color = colors[g % color_mod]
            v_rgb, v_dep, v_con, v_xyl, v_op, T, buffer, dbuffer = _blend_depth_vjp(
                dx,
                dy,
                conic[0],
                conic[1],
                conic[2],
                opac,
                vis,
                alpha,
                color,
                depths[g],
                T,
                buffer,
                dbuffer,
                t_final,
                bg,
                v_out,
                v_outd,
            )

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


@wp.func
def _blend_color_vjp(
    dx: wp.float32,
    dy: wp.float32,
    a: wp.float32,
    b: wp.float32,
    c: wp.float32,
    opac: wp.float32,
    vis: wp.float32,
    alpha: wp.float32,
    color: wp.vec3,
    T_in: wp.float32,
    buffer_in: wp.vec3,
    t_final: wp.float32,
    bg: wp.vec3,
    v_out: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec2, wp.float32, wp.float32, wp.vec3]:
    # Per-gaussian blend vjp for one pixel, shared by the staged and aggregated backward walks.
    # From the blend state at this gaussian (running transmittance T_in, colour accumulator
    # buffer_in), it returns the colour, conic, xy and opacity cotangent contributions, plus the T
    # and buffer for the next (front-ward) gaussian. The conic/xy/opacity grads are zero where the
    # alpha clamp is active, since alpha is then constant. The caller accumulates the grads its own
    # way (per-thread atomics or a warp tile_sum).
    ra = 1.0 / (1.0 - alpha)
    T = T_in * ra
    fac = alpha * T
    v_rgb = v_out * fac
    v_alpha = (color[0] * T - buffer_in[0] * ra) * v_out[0]
    v_alpha += (color[1] * T - buffer_in[1] * ra) * v_out[1]
    v_alpha += (color[2] * T - buffer_in[2] * ra) * v_out[2]
    v_alpha += -t_final * ra * bg[0] * v_out[0]
    v_alpha += -t_final * ra * bg[1] * v_out[1]
    v_alpha += -t_final * ra * bg[2] * v_out[2]
    buffer = buffer_in + color * fac
    v_con = wp.vec3(0.0, 0.0, 0.0)
    v_xy = wp.vec2(0.0, 0.0)
    v_op = wp.float32(0.0)
    if opac * vis <= 0.999:
        v_sigma = -opac * vis * v_alpha
        v_con = wp.vec3(0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy)
        v_xy = wp.vec2(v_sigma * (a * dx + b * dy), v_sigma * (b * dx + c * dy))
        v_op = vis * v_alpha
    return v_rgb, v_con, v_xy, v_op, T, buffer


@wp.func
def _blend_depth_vjp(
    dx: wp.float32,
    dy: wp.float32,
    a: wp.float32,
    b: wp.float32,
    c: wp.float32,
    opac: wp.float32,
    vis: wp.float32,
    alpha: wp.float32,
    color: wp.vec3,
    depth: wp.float32,
    T_in: wp.float32,
    buffer_in: wp.vec3,
    dbuffer_in: wp.float32,
    t_final: wp.float32,
    bg: wp.vec3,
    v_out: wp.vec3,
    v_outd: wp.float32,
) -> tuple[wp.vec3, wp.float32, wp.vec3, wp.vec2, wp.float32, wp.float32, wp.vec3, wp.float32]:
    # Depth-augmented _blend_color_vjp. The depth channel blends like a colour channel, so it
    # adds a term to v_alpha and produces its own cotangent v_depth and accumulator dbuffer.
    # Background depth is zero, so there is no t_final*bg term for it.
    ra = 1.0 / (1.0 - alpha)
    T = T_in * ra
    fac = alpha * T
    v_rgb = v_out * fac
    v_depth = v_outd * fac
    v_alpha = (color[0] * T - buffer_in[0] * ra) * v_out[0]
    v_alpha += (color[1] * T - buffer_in[1] * ra) * v_out[1]
    v_alpha += (color[2] * T - buffer_in[2] * ra) * v_out[2]
    v_alpha += (depth * T - dbuffer_in * ra) * v_outd
    v_alpha += -t_final * ra * bg[0] * v_out[0]
    v_alpha += -t_final * ra * bg[1] * v_out[1]
    v_alpha += -t_final * ra * bg[2] * v_out[2]
    buffer = buffer_in + color * fac
    dbuffer = dbuffer_in + depth * fac
    v_con = wp.vec3(0.0, 0.0, 0.0)
    v_xy = wp.vec2(0.0, 0.0)
    v_op = wp.float32(0.0)
    if opac * vis <= 0.999:
        v_sigma = -opac * vis * v_alpha
        v_con = wp.vec3(0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy)
        v_xy = wp.vec2(v_sigma * (a * dx + b * dy), v_sigma * (b * dx + c * dy))
        v_op = vis * v_alpha
    return v_rgb, v_depth, v_con, v_xy, v_op, T, buffer, dbuffer
