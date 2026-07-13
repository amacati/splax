"""PLY export round-trips, asset fetching, and the inference/training split parity.

``splax.io.write_ply`` must be the exact inverse of ``splax.io.load_ply``.
Two round-trips assert that:

1. random render-space splats through write_ply then load_ply reproduce the inputs, and
2. a real scene (lego.ply) written to a copy then reloaded renders to the same image
   (fit-free, no training, just the load/write/load/render loop),

plus that ``splax.inference.render`` and ``splax.training.render`` produce the
identical forward image (the split is numerically zero-cost).

``splax.fetch`` is exercised against a local ``http.server`` on an ephemeral
port: download, cache hit, force re-download, ETag-based invalidation, and the
``SPLAX_CACHE`` environment fallback.
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import splax
from splax.io import fetch, load_ply

if TYPE_CHECKING:
    from collections.abc import Iterator


def lookat_viewmats(center: np.ndarray, radius: float, num_views: int) -> jax.Array:
    """World-to-camera matrices orbiting ``center`` (OpenCV convention, +z forward)."""
    mats = []
    for i in range(num_views):
        az = 2 * np.pi * i / num_views
        eye = center + radius * np.array([np.sin(az), 0.3, np.cos(az)])
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])  # rows: cam axes in world
        t = -R @ eye
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, t
        mats.append(m)
    return jnp.asarray(np.stack(mats), jnp.float32)


LEGO_PLY = Path("data/scenes/lego.ply")


class _RenderKw(TypedDict):
    background: jax.Array
    glob_scale: float
    clip_thresh: float


RENDER_KW: _RenderKw = {"background": jnp.ones(3), "glob_scale": 1.0, "clip_thresh": 0.01}


def _render(
    splats: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    viewmat: jax.Array,
    H: int,
    W: int,
) -> jax.Array:
    means, scales, quats, colors, opac = splats
    return splax.inference.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )


def _random_splats(
    seed: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    opac = rng.uniform(0.05, 0.95, (n, 1)).astype(np.float32)
    return means, scales, quats, colors, opac


def test_write_ply_is_load_ply_inverse(tmp_path: Path) -> None:
    """Random splats through write_ply then load_ply reproduce the render-space inputs."""
    means, scales, quats, colors, opac = _random_splats(seed=0, n=5000)
    out = tmp_path / "rand.ply"
    splax.io.write_ply(out, means, scales, quats, colors, opac)

    lm, ls, lq, lc, lo = (np.asarray(x) for x in load_ply(out))

    np.testing.assert_allclose(lm, means, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ls, scales, rtol=1e-5, atol=1e-6)
    # quats are normalized on both sides, compare up to sign is unnecessary since
    # write_ply preserves the stored raw quat direction and load re-normalizes.
    np.testing.assert_allclose(lq, quats, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lc, colors, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lo, opac, rtol=1e-4, atol=1e-4)


def test_ply_render_roundtrip(tmp_path: Path) -> None:
    """Fit-free: load lego.ply, write copy, reload, identical render."""
    splats = load_ply(LEGO_PLY)
    copy = tmp_path / "lego_copy.ply"
    splax.io.write_ply(copy, *splats)
    splats2 = load_ply(copy)

    center = np.asarray(splats[0].mean(axis=0))
    radius = float(np.percentile(np.linalg.norm(np.asarray(splats[0]) - center, axis=-1), 90))
    viewmat = lookat_viewmats(center, radius, 1)[0]

    H = W = 200
    img1 = np.asarray(_render(splats, viewmat, H, W))
    img2 = np.asarray(_render(splats2, viewmat, H, W))
    # Activation round-trip (log/exp, logit/sigmoid) is ULP-level, the render is
    # essentially identical. Splatting's hard 1/255 cull can flip a handful of
    # pixels, so bound by max abs diff rather than requiring bit-exactness.
    assert np.max(np.abs(img1 - img2)) < 1e-3


def test_inference_equals_training_forward() -> None:
    """The split is numerically zero-cost: identical forward image."""
    splats = load_ply(LEGO_PLY)
    center = np.asarray(splats[0].mean(axis=0))
    radius = float(np.percentile(np.linalg.norm(np.asarray(splats[0]) - center, axis=-1), 90))
    viewmat = lookat_viewmats(center, radius, 1)[0]
    means, scales, quats, colors, opac = splats

    H = W = 200
    inf_img = splax.inference.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )
    train_img, _ = splax.training.render(
        means,
        scales,
        quats,
        colors,
        opac,
        viewmat=viewmat,
        img_shape=(H, W),
        f=(float(H), float(H)),
        c=(W // 2, H // 2),
        **RENDER_KW,
    )
    np.testing.assert_array_equal(np.asarray(inf_img), np.asarray(train_img))


@pytest.fixture
def file_server(tmp_path: Path) -> Iterator[tuple[Path, str, list[str]]]:
    """Serve ``tmp_path/srv`` over HTTP on 127.0.0.1 with an ephemeral port.

    Yields (served directory, base url, list of requested paths).
    """
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    requests: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object):
            super().__init__(*args, directory=str(srv_dir), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            requests.append(self.path)
            super().do_GET()

        def end_headers(self) -> None:
            self.send_header("ETag", '"fixed"')  # Constant: content changes are picked up by force.
            super().end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass  # Keep pytest output clean.

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield srv_dir, url, requests
    server.shutdown()
    server.server_close()


def test_fetch_downloads(file_server: tuple[Path, str, list[str]], tmp_path: Path) -> None:
    """First fetch downloads the file into the cache dir with matching bytes."""
    srv_dir, url, _ = file_server
    (srv_dir / "scene.ply").write_bytes(b"splat bytes")
    cache = tmp_path / "cache"

    path = fetch(f"{url}/scene.ply", cache=cache)

    assert path.parent == cache
    assert path.name.endswith("-scene.ply")
    assert path.read_bytes() == b"splat bytes"


def test_fetch_cache_hit(file_server: tuple[Path, str, list[str]], tmp_path: Path) -> None:
    """Second fetch returns the same path without touching the network."""
    srv_dir, url, requests = file_server
    (srv_dir / "scene.ply").write_bytes(b"splat bytes")
    cache = tmp_path / "cache"

    first = fetch(f"{url}/scene.ply", cache=cache)
    assert len(requests) == 1
    second = fetch(f"{url}/scene.ply", cache=cache)

    assert second == first
    assert len(requests) == 1  # No new request on the cache hit.


def test_fetch_force_redownloads(file_server: tuple[Path, str, list[str]], tmp_path: Path) -> None:
    """force=True re-downloads and overwrites the cached copy."""
    srv_dir, url, _ = file_server
    (srv_dir / "scene.ply").write_bytes(b"old bytes")
    cache = tmp_path / "cache"

    path = fetch(f"{url}/scene.ply", cache=cache)
    (srv_dir / "scene.ply").write_bytes(b"new bytes")
    assert fetch(f"{url}/scene.ply", cache=cache).read_bytes() == b"old bytes"

    forced = fetch(f"{url}/scene.ply", cache=cache, force=True)

    assert forced == path
    assert forced.read_bytes() == b"new bytes"


def test_fetch_etag_invalidates_cache(tmp_path: Path) -> None:
    """A cache hit is reused while the remote ETag is unchanged, and refetched when it changes."""
    state = {"body": b"v1 bytes", "etag": '"aaa"', "gets": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["body"])))
            self.send_header("ETag", state["etag"])
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            self._headers()

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            state["gets"] += 1
            self._headers()
            self.wfile.write(state["body"])

        def log_message(self, format: str, *args: object) -> None:
            pass  # Keep pytest output clean.

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/scene.ply"
    cache = tmp_path / "cache"
    try:
        assert fetch(url, cache=cache).read_bytes() == b"v1 bytes"
        assert state["gets"] == 1
        fetch(url, cache=cache)  # Same ETag: cache hit, no new download.
        assert state["gets"] == 1

        state["body"], state["etag"] = b"v2 bytes longer", '"bbb"'
        assert fetch(url, cache=cache).read_bytes() == b"v2 bytes longer"  # Changed ETag: refetch.
        assert state["gets"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_without_etag(tmp_path: Path) -> None:
    """A remote that sends no ETag still downloads, re-fetching on every call."""
    state = {"body": b"no etag bytes", "gets": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()

        def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            self._headers()

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            state["gets"] += 1
            self._headers()
            self.wfile.write(state["body"])

        def log_message(self, format: str, *args: object) -> None:
            pass  # Keep pytest output clean.

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/scene.ply"
    cache = tmp_path / "cache"
    try:
        assert fetch(url, cache=cache).read_bytes() == b"no etag bytes"
        assert state["gets"] == 1
        fetch(url, cache=cache)  # No ETag to compare: download again rather than serve stale.
        assert state["gets"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_env_cache(
    file_server: tuple[Path, str, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit cache the file lands in $SPLAX_CACHE."""
    srv_dir, url, _ = file_server
    (srv_dir / "scene.ply").write_bytes(b"splat bytes")
    env_cache = tmp_path / "env_cache"
    monkeypatch.setenv("SPLAX_CACHE", str(env_cache))

    path = fetch(f"{url}/scene.ply")

    assert path.parent == env_cache
    assert path.read_bytes() == b"splat bytes"
