"""Warp backend caches shared by the rasterization pipeline.

Three per-device caches keep the render loop off the allocator and the launch fast path:
recorded kernel launches that replay with only the changed arguments repacked, grow-only sort and
bin scratch buffers keyed on the workload shape, and a pinned staging element for the one
intersection-count readback. All three are purged together, so a recorded launch never keeps a
freed scratch buffer alive through its retained argument list.
"""

from typing import TypedDict

import warp as wp


class _ScratchEntry(TypedDict):
    """One device's scratch buffers, keyed on the workload shape signature."""

    sig: tuple
    isect_cap: int
    isect_dtype: type
    isect_ids: wp.array | None
    gaussian_ids: wp.array | None
    tile_bins: wp.array
    depth_mm: wp.array


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
_scratch_cache: dict[str, _ScratchEntry] = {}

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
) -> _ScratchEntry:
    key = str(device)  # wp.Device is unhashable, its string alias is stable
    entry = _scratch_cache.get(key)
    if entry is None or entry["sig"] != sig:
        # New workload signature. Drop everything first so the peak is the new
        # size, not old plus new. Purge the recorded launches first, they hold
        # the old array references.
        _launch_cache.clear()
        _scratch_cache.pop(key, None)
        entry: _ScratchEntry = {
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
