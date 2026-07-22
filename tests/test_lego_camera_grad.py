"""Gsplat parity for the camera pose (viewmat) gradient on the lego scene.

Both implementations differentiate a pixelwise MSE against a target rendered at a slightly offset
pose with respect to the viewmat. Covers single-view and batched gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _gsplat_ref as gref
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform as RT

import splax

if TYPE_CHECKING:
    import types
    from pathlib import Path

HEIGHT = WIDTH = 800
TARGET_OFFSET = jnp.array([0.01, -0.01, 0.01, 0.01, -0.01, 0.01])


@pytest.fixture
def gsplat_ref() -> types.ModuleType:
    """Fail the test with a clear reason when gsplat cannot run."""
    gref.require_working()
    return gref


def _lego_scene(lego_meta: dict, lego_ply: Path) -> tuple[tuple[jax.Array, ...], np.ndarray, float]:
    """Load the pretrained lego splat and the frame-0 test pose (viewmat, focal)."""
    gaussians = splax.io.load_ply(lego_ply)
    viewmat = splax.utils.nerf_camera(lego_meta["frames"][0]["transform_matrix"])
    focal = float(0.5 * WIDTH / np.tan(0.5 * lego_meta["camera_angle_x"]))
    return gaussians, viewmat, focal


def test_lego_viewmat_grad_gsplat_parity(
    gsplat_ref: types.ModuleType, lego_meta: dict, lego_ply: Path
) -> None:
    """Single-view and vmap-batched viewmat gradients must match gsplat's autograd."""
    (means, scales, quats, colors, opacities), viewmat, focal = _lego_scene(lego_meta, lego_ply)
    camera_kwargs = {
        "background": jnp.ones(3),
        "img_shape": (HEIGHT, WIDTH),
        "f": (focal, focal),
        "c": (WIDTH / 2, HEIGHT / 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }
    target_viewmat = RT.from_exp_coords(TARGET_OFFSET).as_matrix() @ viewmat

    # three current poses at scaled offsets from the base pose
    tangents = jnp.array([0.25, 0.5, 0.75])[:, None] * TARGET_OFFSET[None, :]
    viewmats = RT.from_exp_coords(tangents).as_matrix() @ viewmat

    def splax_image(viewmat_in: jax.Array) -> jax.Array:
        return splax.render(
            means, scales, quats, colors, opacities, viewmat=viewmat_in, **camera_kwargs
        )[0]

    # render the target in the same framework to reduce noise from framework differences
    splax_target = splax_image(target_viewmat)
    gsplat_target = gsplat_ref.render(
        means, scales, quats, colors, opacities, viewmat=target_viewmat, **camera_kwargs
    )

    def loss(viewmat_in: jax.Array) -> jax.Array:
        return jnp.mean((splax_image(viewmat_in) - splax_target) ** 2)

    splax_single_grad = np.asarray(jax.grad(loss)(viewmats[1]))
    gsplat_single_grad = gsplat_ref.viewmat_grad(
        means,
        scales,
        quats,
        colors,
        opacities,
        viewmat=viewmats[1],
        target=gsplat_target,
        **camera_kwargs,
    )
    assert np.allclose(splax_single_grad, gsplat_single_grad, rtol=1e-3, atol=1e-4)
    assert not np.allclose(splax_single_grad, np.zeros_like(splax_single_grad)), "gradient is zero"

    splax_batch_grads = np.asarray(jax.vmap(jax.grad(loss))(viewmats))
    gsplat_batched_grads = np.asarray(
        [
            gsplat_ref.viewmat_grad(
                means,
                scales,
                quats,
                colors,
                opacities,
                viewmat=viewmats[view],
                target=gsplat_target,
                **camera_kwargs,
            )
            for view in range(3)
        ]
    )
    assert np.allclose(splax_batch_grads, gsplat_batched_grads, rtol=1e-3, atol=1e-4)
