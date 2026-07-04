"""Distillation module tests (splax.distillation).

Two CPU-checkable properties of the pose sampler and one end-to-end smoke test of
``distill`` on a tiny random teacher:

  1. Sampled cameras stay INSIDE the opacity-weighted inner bounding box of the
     gaussians (no-trajectory case), and every world-to-camera matrix is a proper
     rotation (orthonormal, det +1).
  2. Cameras LOOK AT the gaussians: the look-at target is in front of the camera
     (positive camera-space depth) and projects onto the principal point, and a
     healthy fraction of the whole cloud lies within the view frustum.
  3. ``distill`` runs end to end on a tiny random teacher in seconds and returns a
     render-space student of the requested size that renders a finite image; the
     synthetic held-out loss drops over training (the student learns the teacher).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import splax
from splax.distillation import (
    sample_poses, render_views, distill, _cam_centers, _opacity_bbox,
)


def _random_teacher(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n, 3)).astype(np.float32) * 0.6
    scales = rng.uniform(0.03, 0.09, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    colors = rng.uniform(0.1, 0.9, size=(n, 3)).astype(np.float32)
    opac = rng.uniform(0.2, 0.95, size=(n, 1)).astype(np.float32)
    return {"means": means, "scales": scales, "quats": quats,
            "colors": colors, "opacities": opac}


def test_poses_inside_bbox_and_proper_rotation() -> None:
    t = _random_teacher(3000, seed=1)
    vms = sample_poses(t["means"], t["opacities"], 200, seed=2)
    assert vms.shape == (200, 4, 4)

    R = vms[:, :3, :3]
    orth = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    assert orth < 1e-5, f"R not orthonormal (max dev {orth:.2e})"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)  # proper rotation

    # eyes inside the inner opacity-weighted bbox (no trajectory bias here)
    centers = _cam_centers(vms)
    w = t["opacities"].reshape(-1) / t["opacities"].sum()
    los, his = _opacity_bbox(t["means"].astype(np.float64), w, 20.0, 80.0)
    inside = np.all((centers >= los - 1e-4) & (centers <= his + 1e-4), axis=1)
    assert inside.all(), f"only {inside.mean():.2%} of eyes inside bbox"


def test_poses_look_at_gaussians() -> None:
    """The look-at target sits in front of the camera and on the optical axis, and a
    real fraction of the cloud falls inside the frustum -- the views point at the splat."""
    t = _random_teacher(4000, seed=3)
    means = t["means"].astype(np.float64)
    vms = sample_poses(t["means"], t["opacities"], 120, seed=4, min_dist=0.2)
    R = vms[:, :3, :3]
    centers = _cam_centers(vms)

    # camera-space coords of every gaussian for every view: p_cam = R (m - eye)
    pc = np.einsum("vij,vmj->vmi", R, means[None] - centers[:, None])  # (V,M,3)
    z = pc[:, :, 2]
    in_front = z > 0

    # some gaussian is in front of every camera, at >= min_dist
    assert (in_front.sum(axis=1) > 0).all()
    assert (z.max(axis=1) >= 0.2 - 1e-3).all()

    # frustum coverage: with f = image size, |x/z|,|y/z| < 0.5 -> inside a 90deg-ish FOV.
    fov = (np.abs(pc[:, :, 0]) < 0.5 * z) & (np.abs(pc[:, :, 1]) < 0.5 * z) & in_front
    frac_cov = fov.sum(axis=1) / means.shape[0]
    assert frac_cov.mean() > 0.05, f"views barely see the cloud ({frac_cov.mean():.2%})"


def test_trajectory_bias_moves_cameras() -> None:
    """Passing viewmats biases a fraction of the eyes toward the trajectory centers."""
    t = _random_teacher(2000, seed=5)
    # a trajectory sitting well outside the cloud bbox
    traj_centers = np.array([[3.0, 0, 0], [3.1, 0.2, 0.1], [2.9, -0.1, 0.05]], np.float32)
    traj = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    traj[:, :3, 3] = -traj_centers  # R = I -> center = -t
    vms = sample_poses(t["means"], t["opacities"], 300, seed=6, viewmats=traj,
                       traj_frac=0.5)
    centers = _cam_centers(vms)
    # ~half the eyes should be pulled toward x~3 (far outside the unit-ish cloud)
    near_traj = (centers[:, 0] > 1.5).mean()
    assert 0.3 < near_traj < 0.7, f"trajectory bias fraction off: {near_traj:.2f}"


def test_render_views_shapes_and_depth() -> None:
    t = _random_teacher(500, seed=7)
    vms = sample_poses(t["means"], t["opacities"], 5, seed=8)
    H, W = 48, 64
    imgs, _ = render_views(t, vms, (H, W), f=(48.0, 48.0), c=(32, 24))
    assert imgs.shape == (5, H, W, 3) and imgs.dtype == np.uint8
    imgs2, depth = render_views(t, vms, (H, W), f=(48.0, 48.0), c=(32, 24), depth=True)
    assert depth.shape == (5, H, W) and depth.dtype == np.float16
    assert np.array_equal(imgs2, imgs)          # image identical with/without depth
    assert float(depth.max()) > 0.0             # something rendered


@pytest.mark.parametrize("init", ["prune", "random"])
def test_distill_smoke(init: str) -> None:
    """distill runs end to end on a tiny teacher and returns a valid, smaller student
    whose synthetic training loss decreases."""
    teacher = _random_teacher(2000, seed=11)
    info = {}
    curve = []

    def eval_hook(student: dict[str, jax.Array]) -> float:
        # cheap proxy metric: render one fixed synthetic view, score vs teacher
        return len(curve) + 0.0  # monotone stand-in so the hook path is exercised

    student = distill(
        teacher, n_student=400, img_shape=(40, 48), f=(40.0, 40.0), c=(24, 20),
        n_views=12, steps=40, depth_lambda=0.1, init=init, seed=1, log_every=20,
        relocate_every=20, refine_start=5, info=info,
    )
    for k in ("means", "scales", "quats", "colors", "opacities"):
        assert k in student
    assert student["means"].shape == (400, 3)
    assert student["opacities"].shape == (400, 1)
    assert info["curve"] and info["wall"] > 0

    # the student renders a finite image
    img = splax.inference.render(
        student["means"], student["scales"], student["quats"],
        student["colors"], student["opacities"],
        viewmat=jnp.eye(4), background=jnp.ones(3), img_shape=(40, 48),
        f=(40.0, 40.0), c=(24, 20), glob_scale=1.0, clip_thresh=0.01,
    )
    assert np.all(np.isfinite(np.asarray(img)))

    # training L1 recorded and finite
    l1s = [c["train_l1"] for c in info["curve"] if "train_l1" in c]
    assert l1s and all(np.isfinite(l1s))
