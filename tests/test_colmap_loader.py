"""COLMAP loader invariants for the ``colmap`` training-toolkit module.

Checks the pycolmap loader and the point-cloud init produce self-consistent, static shapes with the
right conventions (no GPU and no render needed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from colmap import init_from_points, read_reconstruction
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pathlib import Path


def test_parsers_and_conventions(drone_sparse: Path) -> None:
    cams, imgs, (xyz, rgb, ids, _) = read_reconstruction(drone_sparse)

    assert len(cams) >= 1 and len(imgs) > 0 and xyz.shape[0] > 0
    assert xyz.shape[1] == 3 and rgb.shape == xyz.shape
    assert ids.shape == (xyz.shape[0],)
    # images sorted by name, every image references a known camera
    assert [im["name"] for im in imgs] == sorted(im["name"] for im in imgs)
    assert all(im["camera_id"] in cams for im in imgs)
    # per-image 2D observations: valid point ids reference known points
    known = set(int(p) for p in ids)
    obs = imgs[0]["obs_pid"]
    assert imgs[0]["obs_xy"].shape == (obs.shape[0], 2)
    assert obs.shape[0] == 0 or all(int(p) in known for p in obs[:50])

    # the COLMAP qvec (scalar-first) is a proper rotation (orthonormal, det +1)
    rot = Rotation.from_quat(imgs[0]["qvec"], scalar_first=True).as_matrix()
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)


def test_point_init_static_shapes(drone_sparse: Path) -> None:
    _, _, (xyz, rgb, _, _) = read_reconstruction(drone_sparse)
    n = 8000
    p = init_from_points(xyz[:3000].astype(np.float32), rgb[:3000], n, 0.1, seed=0)
    assert p["means"].shape == (n, 3)
    assert p["log_scales"].shape == (n, 3)
    assert p["quats"].shape == (n, 4)
    assert p["colors_logit"].shape == (n, 3)
    assert p["opac_logit"].shape == (n, 1)
    assert np.all(np.isfinite(np.asarray(p["means"])))
    assert np.all(np.isfinite(np.asarray(p["log_scales"])))
