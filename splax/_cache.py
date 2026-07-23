"""Warp backend caches shared by the rasterization pipeline.

Rasterization first sorts the Gaussians and computes the total intersection count. The count is read
back on the host to size the pipeline buffers. Warp cannot update a captured CUDA graph, so the
pipeline launches its kernels individually once the count is known. To accelerate the launches as
much as possible, we keep three per-device caches that avoid allocations and kernel repacks.

The launch cache records each kernel as a Warp launch object keyed on kernel and device. A full
wp.launch repacks and validates every argument and costs 5 to 16 us of host time per launch. A
recorded launch replays in about 2 us and repacks only the arguments whose recorded state changed
since the previous call. Most inputs stay fixed across frames, so the diff repacks little.

The scratch cache holds the sort and bin buffers keyed on the workload shape over batch size,
Gaussian count, and tile count. The buffers are reused across renders and only grow, which avoids a
cold reallocation of several gigabytes each frame. A change in workload shape drops the whole entry.
An intersection count above the current capacity reallocates the sort buffers with headroom, at the
cost of a longer compute time on that call.

The readback cache holds one pinned int32 staging element per device for the single
intersection-count copy, fenced with an event. A pageable readback allocates every frame and costs
about 17 us of host time, while the pinned copy and its event sync cost about 6 us.

A recorded launch retains raw pointers into the scratch buffers, so the launch cache and the scratch
cache are purged together whenever the scratch buffers move. The readback cache holds no such
references and is independent of the other two.
"""

from __future__ import annotations

from typing import TypedDict

import warp as wp

_SCRATCH_HEADROOM = 1.25
_scratch_cache: dict[str, _ScratchEntry] = {}  # Persistent grow-only scratch cache
_launch_cache: dict = {}  # Recorded per-kernel launches, keyed on (kernel, device)
_readback_bufs: dict = {}  # Pinned staging element per device for the intersection-count readback


class _ScratchEntry(TypedDict):
    """One device's scratch buffers, keyed on the workload shape signature."""

    sig: tuple
    isect_cap: int
    isect_dtype: type
    isect_ids: wp.array | None  # radix sort key buffer
    gaussian_ids: wp.array | None  # radix sort value buffer, same length as isect_ids
    tile_bins: wp.array  # per-image per-tile bin edges of length B*num_tiles
    depth_mm: wp.array  # per-image [dmin, dmax] for the packed depth quantization


def cached_scratch(
    device: wp.Device | None,
    sig: tuple,
    isect_need: int,
    bins_need: int,
    isect_dtype: type = wp.int64,
) -> _ScratchEntry:
    key = str(device)  # wp.Device itself is unhashable
    entry = _scratch_cache.get(key)
    if entry is None or entry["sig"] != sig:
        # New workload signature. Drop everything first to avoid transient peaks. Purge the recorded
        # launches first, they hold the old array references.
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
        # Grow the buffer with headroom to avoid reallocation for slighly larger counts.
        cap = max(int(isect_need * _SCRATCH_HEADROOM) + 1, entry["isect_cap"])
        # Sort buffers move on realloc, so recorded launches must drop their references to the old
        # buffers before the free below.
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


def cached_launch(
    kernel: wp.Kernel, dim: int, args: list, device: wp.Device | None, block_dim: int = 0
) -> None:
    """Launch a kernel through its recorded per-device launch object.

    Records the launch on first use and afterwards replays it, repacking only the arguments whose
    recorded state changed. FFI callbacks are serialized by Warp's callback lock.

    Entries are keyed on (kernel, device) and the static shape signatures, and drop entries if the
    signature changes.
    """
    entry = _launch_cache.get((kernel, str(device)))
    if entry is None:
        if block_dim:
            launch = wp.launch_tiled(
                kernel, dim=[dim], inputs=args, block_dim=block_dim, device=device, record_cmd=True
            )
        else:
            launch = wp.launch(kernel, dim=dim, inputs=args, device=device, record_cmd=True)
        # State holds the launch dim followed by one entry per argument, arrays as (ptr, shape) and
        # scalars by value.
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


def begin_count_read(src: wp.array, index: int, device: wp.Device | None) -> tuple | int:
    """Enqueue the one-element readback copy and fence it with an event.

    On non-CUDA devices the read is synchronous. Pass the result to fetch_count_read to get the
    result. Count-independent kernels can be launched between the begin and fetch calls, so they
    execute inside the sync bubble without delaying the readback.
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


def fetch_count_read(pending: tuple | int) -> int:
    """Wait for the readback copy and return the count."""
    if isinstance(pending, int):
        return pending
    wp.synchronize_event(pending[2])
    return int(pending[1][0])


def clear_cache() -> None:
    """Release the persistent sort and bin scratch cache buffers on all devices.

    Also purges the recorded launch cache to prevent referencing freed buffers.
    """
    _launch_cache.clear()
    _scratch_cache.clear()
