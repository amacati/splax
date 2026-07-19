"""Web-based splat viewer built on `viser <https://viser.studio>`_.

``Viewer`` wraps a ``viser.ViserServer`` and exposes splats as named rigid objects. ``add_splats``
uploads the gaussians of one object once, and ``update_pose`` moves objects afterwards without
re-uploading.

Viser is an optional dependency that can be installed with ``pip install splax[viewer]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

try:
    import viser
except ImportError as e:
    raise ImportError(
        "splax.viewer requires viser. Install it with `pip install splax[viewer]`."
    ) from e

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    import jax


class Viewer:
    """Splat viewer serving a web client, one scene node per rigid object."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        up_direction: Literal["+x", "+y", "+z", "-x", "-y", "-z"] = "+z",
    ):
        """Start the viser server and configure the scene.

        Args:
            host: Host address to bind the web server to.
            port: Port of the web server.
            up_direction: World up direction, e.g. ``"+z"`` (MuJoCo) or ``"+y"``.
        """
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction(up_direction)
        self._handles: dict[str, viser.GaussianSplatHandle] = {}

    def add_splats(
        self,
        name: str,
        means: jax.Array | np.ndarray,
        scales: jax.Array | np.ndarray,
        quats: jax.Array | np.ndarray,
        colors: jax.Array | np.ndarray,
        opacities: jax.Array | np.ndarray,
        *,
        position: jax.Array | np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        wxyz: jax.Array | np.ndarray | tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Upload one rigid object's gaussians to the viewer under ``name``.

        Args:
            name: Name of the object.
            means: (N, 3) float32 centers in the object frame.
            scales: (N, 3) positive per-axis scales.
            quats: (N, 4) wxyz unit quaternions.
            colors: (N, 3) float32 RGB values.
            opacities: (N, 1) float32 opacity values.
            position: Initial world position of the object.
            wxyz: Initial world orientation of the object as a wxyz quaternion.
        """
        # Rotate covariances into the world frame
        scales = np.asarray(scales, np.float32)
        rot = R.from_quat(np.asarray(quats, np.float32), scalar_first=True).as_matrix()
        covariances = np.einsum("nij,nj,nkj->nik", rot, scales**2, rot).astype(np.float32)
        self._handles[name] = self.server.scene.add_gaussian_splats(
            f"/{name}",
            centers=np.asarray(means, np.float32),
            covariances=covariances,
            rgbs=np.asarray(colors, np.float32),
            opacities=np.asarray(opacities, np.float32),
            position=np.asarray(position, np.float32),
            wxyz=np.asarray(wxyz, np.float32),
        )

    def update_pose(
        self, name: str, position: jax.Array | np.ndarray, wxyz: jax.Array | np.ndarray
    ) -> None:
        """Set the world pose of the object ``name``.

        Args:
            name: Name of the object.
            position: World position of the object.
            wxyz: World orientation of the object as a wxyz quaternion.
        """
        handle = self._handles[name]
        handle.position = np.asarray(position, np.float32)
        handle.wxyz = np.asarray(wxyz, np.float32)

    def remove(self, name: str) -> None:
        """Remove the object ``name`` from the viewer."""
        self._handles.pop(name).remove()

    def close(self) -> None:
        """Stop the web server."""
        self.server.stop()
