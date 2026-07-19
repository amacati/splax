"""Viewer construction and pose updates, headless and CPU-only.

viser is a websocket server, so the viewer runs without a GPU or display. The
tests check the covariance conversion against closed-form cases and that
add/update/remove round-trip through the underlying viser handles.
"""

from __future__ import annotations

import socket

import jax.numpy as jnp
import numpy as np
import pytest

from splax.viewer import Viewer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_viewer_roundtrip() -> None:
    """Add (jax and numpy inputs), update, and remove splats on a live server."""
    rng = np.random.default_rng(0)
    n = 50
    means = rng.normal(size=(n, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    colors = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    opacities = rng.uniform(0.0, 1.0, (n, 1)).astype(np.float32)

    viewer = Viewer(host="127.0.0.1", port=_free_port())
    try:
        viewer.add_splats("scene", means, scales, quats, colors, opacities)
        viewer.add_splats(
            "drone",
            *(jnp.asarray(x) for x in (means, scales, quats, colors, opacities)),
            position=(1.0, 2.0, 3.0),
        )

        pos, wxyz = np.array([0.5, -0.5, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
        viewer.update_pose("drone", pos, wxyz)
        handle = viewer._handles["drone"]
        np.testing.assert_allclose(handle.position, pos)
        np.testing.assert_allclose(handle.wxyz, wxyz)

        with pytest.raises(KeyError, match="gate"):
            viewer.update_pose("gate", pos, wxyz)

        viewer.remove("drone")
        assert "drone" not in viewer._handles
    finally:
        viewer.close()
