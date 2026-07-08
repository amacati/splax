"""COLMAP loader invariants for ``scripts/train_colmap.py``.

Requires the drone scene unzipped to ``data/drone/sparse/0``. Checks the
hand-written COLMAP binary parsers and the point-cloud init produce
self-consistent, static shapes with the right conventions (no GPU and no render
needed).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import types

ROOT = Path(__file__).resolve().parent.parent
SPARSE = ROOT / "data" / "drone" / "sparse" / "0"


def _load_module() -> types.ModuleType:
    return importlib.import_module("scripts.train_colmap")


def test_parsers_and_conventions() -> None:
    tc = _load_module()
    cams = tc.read_cameras(SPARSE / "cameras.bin")
    imgs = tc.read_images(SPARSE / "images.bin")
    xyz, rgb, ids, _track_lens = tc.read_points3D(SPARSE / "points3D.bin")

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

    # quat2mat returns a proper rotation (orthonormal, det +1)
    R = tc.quat2mat(imgs[0]["qvec"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_point_init_static_shapes() -> None:
    tc = _load_module()
    xyz, rgb, _ids, _track_lens = tc.read_points3D(SPARSE / "points3D.bin")
    n = 8000
    p = tc.init_from_points(xyz[:3000].astype(np.float32), rgb[:3000], n, 0.1, seed=0)
    assert p["means"].shape == (n, 3)
    assert p["log_scales"].shape == (n, 3)
    assert p["quats"].shape == (n, 4)
    assert p["colors_logit"].shape == (n, 3)
    assert p["opac_logit"].shape == (n, 1)
    assert np.all(np.isfinite(np.asarray(p["means"])))
    assert np.all(np.isfinite(np.asarray(p["log_scales"])))
