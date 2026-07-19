"""Shared camera utilities for scripts, tests, and benchmarks."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import RigidTransform


def look_at(eye: np.ndarray, target: np.ndarray, up: tuple | np.ndarray = (0, 1, 0)) -> np.ndarray:
    """Generate world-to-camera OpenCV matrices looking from ``eye`` to ``target``.

    Args:
        eye: (..., 3) camera positions in world coordinates.
        target: (..., 3) camera targets, broadcastable against ``eye``.
        up: World up direction, must not be parallel to the view direction.

    Returns:
        (..., 4, 4) float32 world-to-camera matrices, at most one batch dimension.
    """
    z = target - eye
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    assert (norm > 0).all(), "eye and target must differ"
    z = z / norm
    x = np.cross(z, up)  # x = right, y = down, so image up (-y) aligns with the world up axis
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    c2w = np.zeros((*z.shape[:-1], 4, 4))
    c2w[..., :3, :3] = np.stack([x, np.cross(z, x), z], axis=-1)
    c2w[..., :3, 3] = eye
    c2w[..., 3, 3] = 1.0
    return RigidTransform.from_matrix(c2w).inv().as_matrix().astype(np.float32)


def nerf_camera(c2w: np.ndarray | list) -> np.ndarray:
    """Convert NeRF blender camera-to-world matrices to world-to-camera view matrices.

    Args:
        c2w: (..., 4, 4) blender camera-to-world matrices, e.g. ``transform_matrix`` entries
            of a NeRF transforms JSON.

    Returns:
        (..., 4, 4) float32 world-to-camera matrices in OpenCV convention, at most one batch
        dimension.
    """
    c2w = np.asarray(c2w, np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])
    return RigidTransform.from_matrix(c2w).inv().as_matrix().astype(np.float32)
