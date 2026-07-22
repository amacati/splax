"""Provide the scene assets the tests read.

The tests need the pretrained lego splat, the lego test-set camera metadata with three of its
views, and the drone COLMAP sparse model. The fixtures download them through ``splax.io.fetch``.
Point ``$SPLAX_TEST_DATA`` at another dataset root to override where they come from.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import imageio.v3 as iio
import pytest

from splax.io import fetch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

BASE = os.environ.get(
    "SPLAX_TEST_DATA",
    "https://huggingface.co/datasets/amacati/splax-test-data/resolve/main",
)


def _read_lego_view(file_path: str) -> np.ndarray:
    """Read the lego test view stored at dataset ``file_path``."""
    return iio.imread(fetch(f"{BASE}/nerf_synthetic/lego/{file_path.lstrip('./')}.png"))


@pytest.fixture(scope="session")
def lego_meta() -> dict:
    """Return the lego test-set camera metadata."""
    return json.loads(fetch(f"{BASE}/nerf_synthetic/lego/transforms_test.json").read_text())


@pytest.fixture(scope="session")
def lego_view() -> Callable[[str], np.ndarray]:
    """Return a reader for the ground-truth image of a lego test view."""
    return _read_lego_view


@pytest.fixture(scope="session")
def lego_ply() -> Path:
    """Return the pretrained lego splat ``.ply``."""
    return fetch(f"{BASE}/scenes/lego.ply")


@pytest.fixture(scope="session")
def drone_sparse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a COLMAP sparse model directory holding the drone reconstruction.

    pycolmap opens the model members by name from a single directory, while the cache stores them
    under hashed names, so the downloads are linked into a directory it can read.
    """
    sparse = tmp_path_factory.mktemp("sparse")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).symlink_to(fetch(f"{BASE}/drone/sparse/0/{name}"))
    return sparse
