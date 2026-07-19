"""look_at and nerf_camera viewmats against closed-form poses."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import splax


def test_look_at_frame() -> None:
    """The eye maps to the origin and the target onto the +z axis at its distance."""
    eye, target = np.array([1.0, 2.0, 3.0]), np.array([4.0, 2.0, -1.0])
    w2c = splax.utils.look_at(eye, target)
    dist = np.linalg.norm(target - eye)
    np.testing.assert_allclose(w2c @ [*eye, 1.0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(w2c @ [*target, 1.0], [0.0, 0.0, dist, 1.0], atol=1e-6)


def test_look_at_up() -> None:
    """World up maps to image up (-y) for a level view."""
    w2c = splax.utils.look_at(np.zeros(3), np.array([1.0, 0.0, 0.0]), up=(0, 0, 1))
    np.testing.assert_allclose(w2c[:3, :3] @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], atol=1e-6)


def test_look_at_batched() -> None:
    """A batched call with a broadcast target matches per-eye single calls."""
    eyes = np.random.default_rng(0).normal(size=(5, 3))
    target = np.array([0.0, 0.0, 5.0])
    single = np.stack([splax.utils.look_at(eye, target) for eye in eyes])
    np.testing.assert_allclose(splax.utils.look_at(eyes, target), single)


def test_nerf_camera() -> None:
    """A blender pose flips the y and z axes and inverts to world-to-camera."""
    w2c = splax.utils.nerf_camera(np.eye(4))
    np.testing.assert_allclose(w2c, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_nerf_camera_center() -> None:
    """The camera centers of batched rotated and translated poses map to the origin."""
    c2w = np.tile(np.eye(4), (2, 1, 1))
    c2w[:, :3, :3] = Rotation.from_euler("xyz", [[0.3, -0.2, 0.5], [1.1, 0.4, -0.6]]).as_matrix()
    c2w[:, :3, 3] = [[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]]
    centers = np.concatenate([c2w[:, :3, 3], np.ones((2, 1))], axis=1)
    mapped = np.einsum("nij,nj->ni", splax.utils.nerf_camera(c2w), centers)
    np.testing.assert_allclose(mapped, [[0.0, 0.0, 0.0, 1.0]] * 2, atol=1e-6)
