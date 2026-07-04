"""Per-image affine exposure correction (survey T5) for ``train_colmap.py``.

All tests here are **pure JAX on CPU** -- they exercise only the affine helpers
(``init_exposure`` / ``apply_exposure``), no renderer, no Warp kernels, no GPU.
``JAX_PLATFORMS=cpu`` is pinned below so they never touch the device even if a GPU
is present. (Importing ``train_colmap`` pulls in ``splax``/Warp at module scope,
but Warp initializes lazily and nothing here launches a kernel.)

The training step itself (``_make_step_exposure``) needs the Warp renderer and so
is *not* unit-tested here; it is covered by the coordinator's drone eval run. The
invariants below are exactly the ones that matter for correctness and honesty:
identity init, correct affine algebra, and off-path parity.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_module() -> types.ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "train_colmap", ROOT / "scripts" / "train_colmap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_init_exposure_is_identity() -> None:
    tc = _load_module()
    ntr = 7
    exp = np.asarray(tc.init_exposure(ntr))
    assert exp.shape == (ntr, 3, 4)
    # every block is [I | 0]
    for i in range(ntr):
        assert np.allclose(exp[i, :, :3], np.eye(3))
        assert np.allclose(exp[i, :, 3], 0.0)


def test_apply_exposure_identity_is_noop() -> None:
    """Identity transform must leave the render bit-identical (off-path parity)."""
    tc = _load_module()
    rng = np.random.default_rng(0)
    img = rng.random((5, 4, 3)).astype(np.float32)
    affine = np.asarray(tc.init_exposure(1))[0]  # (3,4) identity
    out = np.asarray(tc.apply_exposure(img, affine))
    assert np.array_equal(out, img)


def test_apply_exposure_affine_algebra() -> None:
    """Known transform M@rgb + b, applied per pixel, matches an explicit einsum."""
    tc = _load_module()
    rng = np.random.default_rng(1)
    img = rng.random((6, 3, 3)).astype(np.float32)
    M = rng.normal(size=(3, 3)).astype(np.float32)
    b = rng.normal(size=(3,)).astype(np.float32)
    affine = np.concatenate([M, b[:, None]], axis=1)  # (3,4)
    out = np.asarray(tc.apply_exposure(img, affine))
    ref = np.einsum("ij,hwj->hwi", M, img) + b
    assert out.shape == img.shape
    assert np.allclose(out, ref, atol=1e-5)


def test_apply_exposure_scalar_gain_and_offset() -> None:
    """A pure per-channel gain+bias (diagonal M) scales/shifts each channel."""
    tc = _load_module()
    img = np.ones((2, 2, 3), np.float32)
    gain = np.array([0.5, 2.0, 1.0], np.float32)
    bias = np.array([0.1, -0.2, 0.0], np.float32)
    affine = np.concatenate([np.diag(gain), bias[:, None]], axis=1)
    out = np.asarray(tc.apply_exposure(img, affine))
    expect = gain + bias  # img is all ones
    assert np.allclose(out.reshape(-1, 3), expect, atol=1e-6)
