"""Host orchestration of the intersection sort and binning.

Emits sorted intersection keys and per-tile bin edges for one batched launch, driving the key
kernels, the radix sort, and the bin-edge scan around the one device-to-host intersection-count
readback.
"""

import warp as wp

from splax._cache import begin_count_read, cached_launch, cached_scratch, fetch_count_read
from splax._intersect import BLOCK_WIDTH
from splax._rasterize._sort._kernels import (
    _MINMAX_CHUNK,
    _depth_minmax,
    _map_intersects_32bit,
    _map_intersects_64bit,
    _seed_minmax,
    _tile_bin_edges_32bit,
    _tile_bin_edges_64bit,
)


def _use_32bit_keys(depth_bits: int) -> bool:
    # The packed key needs at least 16 depth bits below the image and tile ids.
    # Tests patch this to force the 64 bit path on small scenes.
    return depth_bits >= 16


def sort_and_bin(
    xys: wp.array,
    depths: wp.array,
    radii: wp.array,
    conics: wp.array,
    map_opacities: wp.array,
    cum_tiles_hit: wp.array,
    n: int,
    B: int,
    img_h: int,
    img_w: int,
) -> tuple[wp.array, wp.array, int, int, int]:
    """Emit sorted intersection keys and tile bins for one batched launch.

    B is the geometry batch, how many distinct renders the sort covers. Used by the forward blend
    and by the backward pass, which recomputes the identical sort from the saved cum_tiles_hit. The
    sort is deterministic, so it reproduces the forward gaussian_ids and tile_bins and the saved
    final_idx stays valid. Returns (gaussian_ids, tile_bins, num_intersects, tile_bounds_x,
    num_tiles), where gaussian_ids is the full scratch buffer whose valid prefix is
    [0, num_intersects).
    """
    device = xys.device
    bw = int(BLOCK_WIDTH)
    tile_bounds_x = (img_w + bw - 1) // bw
    tile_bounds_y = (img_h + bw - 1) // bw
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
    pending = begin_count_read(cum_tiles_hit, total - 1, device)

    isect_dtype = wp.int32 if packed else wp.int64
    scratch = cached_scratch(device, (B, n, num_tiles), 2, bins_len, isect_dtype)
    # tile_bins persists across frames and the bin-edge kernel writes only bins that own
    # intersections, so zero the previous frame's stale edges.
    tile_bins = scratch["tile_bins"]
    tile_bins.zero_()
    if packed:
        # Per-image [dmin, dmax] for the depth quantization, computed device-side with no host sync.
        # Count-independent, so it launches before the readback wait. depth_mm persists across
        # frames and its reduction is an atomic min-max, so seed the sentinels before accumulating.
        depth_mm = scratch["depth_mm"]
        cached_launch(_seed_minmax, B, [depth_mm], device)
        cached_launch(
            _depth_minmax,
            (total + int(_MINMAX_CHUNK) - 1) // int(_MINMAX_CHUNK),
            [depths, radii, total, n, depth_mm],
            device,
        )

    num_intersects = fetch_count_read(pending)
    # Grow the sort buffers to the frame's count. Nothing above needs them.
    scratch = cached_scratch(
        device, (B, n, num_tiles), max(2 * num_intersects, 2), bins_len, isect_dtype
    )
    # The grow above always allocates, so the sort buffers are live from here on.
    isect_ids = scratch["isect_ids"]
    gaussian_ids = scratch["gaussian_ids"]
    assert isect_ids is not None and gaussian_ids is not None

    if num_intersects == 0:
        return gaussian_ids, tile_bins, 0

    # The kernels and the sort take explicit counts, so the full-capacity scratch
    # arrays are passed without per-frame slicing. Every access stays inside
    # [0, 2*num_intersects) and the stable shapes keep the recorded launches from
    # repacking their array arguments each frame.
    if packed:
        cached_launch(
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
        cached_launch(
            _tile_bin_edges_32bit,
            num_intersects,
            [num_intersects, isect_ids, num_tiles, tile_n_bits, depth_bits, tile_bins],
            device,
        )
    else:
        cached_launch(
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
        cached_launch(
            _tile_bin_edges_64bit,
            num_intersects,
            [num_intersects, isect_ids, num_tiles, tile_n_bits, tile_bins],
            device,
        )
    return gaussian_ids, tile_bins, num_intersects, tile_bounds_x, num_tiles
