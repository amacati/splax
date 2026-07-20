"""Batched training-step tests (gsplat ``batch_size`` + sqrt-batch LR).

``train_colmap.make_step`` takes a static ``batch`` and ``jax.vmap``-s a
per-view loss over ``batch`` views (loss mean-reduced over the batch, per gsplat), driving
the batch-native backward as one launch. These tests pin the two properties the
batched step must preserve:

  1. **B=1 identity.** The batched builder at ``batch=1`` reproduces the exact
     single-view gradient (the default trainer path, incl. the long 1.5M fit, must not
     move). Checked against a from-scratch reconstruction of the single-view loss.
  2. **Batch = mean-of-views.** A B=2 step's gradient equals the mean of the two
     corresponding B=1 gradients at identical params (pre-optimizer), the defining
     property of averaging the loss over the batch. Also runs under jit (it always does,
     ``make_step`` jits the step) and the duplicate-view sanity (B=2 of one view == B=1).

Gradients are recovered end-to-end through the real ``make_step`` by using a plain SGD
optimizer (lr=1, so ``grad = p − step(p)``), so the actual jitted code path is exercised.
Tolerances follow ``test_depth_reg.py`` / ``test_warp_grad_batched.py`` (atomic-order jitter
in the batched backward is ~1e-4 rel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dm_pix
import jax
import jax.numpy as jnp
import numpy as np
import optax
from train_colmap import init_exposure, make_step

from splax import render_log

if TYPE_CHECKING:
    from collections.abc import Hashable

H = W = 48
INTR = (48.0, 48.0, 24.0, 24.0)
CAMERA = {"img_shape": (H, W), "f": INTR[:2], "c": INTR[2:]}
SSIM_L, OREG, SREG = 0.2, 0.01, 0.01


def _params(n: int = 200, seed: int = 0) -> dict[str, jax.Array]:
    k = jax.random.split(jax.random.key(seed), 5)
    return {
        "means": jax.random.uniform(k[0], (n, 3), minval=-0.6, maxval=0.6),
        "log_scales": jnp.full((n, 3), jnp.log(0.05)),
        "quats": jax.random.normal(k[1], (n, 4)),
        "colors_logit": jax.random.normal(k[2], (n, 3)) * 0.3,
        "opac_logit": jnp.full((n, 1), -1.0),
    }


def _view(seed: int) -> tuple[jax.Array, jax.Array]:
    k = jax.random.split(jax.random.key(100 + seed), 2)
    gt = jax.random.uniform(k[0], (H, W, 3))
    vm = jnp.array(
        [[1, 0, 0, 0.1 * seed], [0, 1, 0, -0.05 * seed], [0, 0, 1, 4.0], [0, 0, 0, 1]], jnp.float32
    )
    return gt, vm


def _sgd_opt(params: dict[str, jax.Array]) -> optax.GradientTransformation:
    # lr=1 SGD so apply_updates(p) = p - grad, so grad = p - step(p)  (linear recovery).
    txs: dict[Hashable, optax.GradientTransformation] = {kk: optax.sgd(1.0) for kk in params}
    return optax.multi_transform(txs, {kk: kk for kk in params})


def _dummy_pts(B: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.zeros((B, 1, 2), jnp.float32),
        jnp.zeros((B, 1), jnp.float32),
        jnp.zeros((B, 1), jnp.float32),
    )


def _recover_grad(
    params: dict[str, jax.Array], batch: int, gts: jax.Array, vms: jax.Array
) -> dict[str, np.ndarray]:
    """Run one real make_step (SGD lr=1, no depth/exposure) and recover grad = p - new."""
    opt = _sgd_opt(params)
    opt_state = opt.init(params)
    step = make_step(opt, H, W, INTR, SSIM_L, OREG, SREG, batch=batch)
    bg = jnp.broadcast_to(jnp.ones(3), (batch, 3))
    new, _os, _l1 = step(params, opt_state, gts, vms, bg, *_dummy_pts(batch))
    return {kk: np.asarray(params[kk] - new[kk]) for kk in params}


def test_b1_matches_pre_t6_single_view() -> None:
    """batch=1 grad == the reconstructed single-view grad (default path frozen)."""
    params = _params(seed=1)
    gt, vm = _view(3)

    def old_loss(
        p: dict[str, jax.Array],
    ) -> jax.Array:  # the single-view loss_fn the trainer uses for one view
        splats = (p["means"], p["log_scales"], p["quats"], p["colors_logit"], p["opac_logit"])
        img, _ = render_log(*splats, viewmat=vm, background=jnp.ones(3), **CAMERA)
        l1 = jnp.mean(jnp.abs(img - gt))
        dssim = 1.0 - dm_pix.ssim(img, gt)
        loss = (1.0 - SSIM_L) * l1 + SSIM_L * dssim
        loss = loss + OREG * jnp.mean(jax.nn.sigmoid(p["opac_logit"]))
        loss = loss + SREG * jnp.mean(jnp.exp(p["log_scales"]))
        return loss

    g_ref = jax.grad(old_loss)(params)
    g_b1 = _recover_grad(params, 1, gt[None], vm[None])
    for kk in params:
        assert np.allclose(np.asarray(g_ref[kk]), g_b1[kk], rtol=1e-4, atol=1e-6), (
            f"{kk}: B=1 grad differs from pre-T6 single-view grad"
        )


def test_b2_grad_equals_mean_of_b1_grads() -> None:
    """A B=2 step's gradient == mean of the two B=1 gradients (loss averaged over batch)."""
    params = _params(seed=2)
    gt0, vm0 = _view(1)
    gt1, vm1 = _view(2)

    g0 = _recover_grad(params, 1, gt0[None], vm0[None])
    g1 = _recover_grad(params, 1, gt1[None], vm1[None])
    gb = _recover_grad(params, 2, jnp.stack([gt0, gt1]), jnp.stack([vm0, vm1]))
    for kk in params:
        mean_g = 0.5 * (g0[kk] + g1[kk])
        assert np.allclose(gb[kk], mean_g, rtol=2e-3, atol=1e-5), (
            f"{kk}: batched grad != mean of per-view grads "
            f"(max dev {np.abs(gb[kk] - mean_g).max():.2e})"
        )


def test_b2_duplicate_view_equals_b1() -> None:
    """B=2 of the same view twice == B=1 of that view (mean of identical = identical)."""
    params = _params(seed=4)
    gt, vm = _view(5)
    g_b1 = _recover_grad(params, 1, gt[None], vm[None])
    g_b2 = _recover_grad(params, 2, jnp.stack([gt, gt]), jnp.stack([vm, vm]))
    for kk in params:
        assert np.allclose(g_b1[kk], g_b2[kk], rtol=2e-3, atol=1e-5), kk


def test_batched_step_runs_under_jit_with_exposure_and_depth() -> None:
    """The batched step traces + runs for B=3 with depth-loss and exposure-opt on."""
    params = _params(n=150, seed=6)
    B = 3
    opt = _sgd_opt(params)
    opt_state = opt.init(params)
    exp_tx = optax.sgd(1.0)
    exp_p = {"exp": init_exposure(8)}
    exp_state = exp_tx.init(exp_p)
    step = make_step(
        opt, H, W, INTR, SSIM_L, OREG, SREG, depth_loss=True, aux_tx=exp_tx, exp_opt=True, batch=B
    )
    gts = jax.random.uniform(jax.random.key(7), (B, H, W, 3))
    vms = jnp.broadcast_to(jnp.eye(4).at[2, 3].set(4.0), (B, 4, 4))
    bg = jnp.broadcast_to(jnp.ones(3), (B, 3))
    vi = jnp.array([0, 3, 5], jnp.int32)
    uv = jax.random.uniform(jax.random.key(8), (B, 4, 2)) * 40 + 4
    depth = jnp.full((B, 4), 4.0)
    mask = jnp.ones((B, 4))
    new_p, _os, new_exp, _es, l1 = step(
        params, opt_state, exp_p, exp_state, gts, vms, bg, vi, uv, depth, mask
    )
    assert np.isfinite(float(l1))
    # only the touched exposure rows (0,3,5) moved, the rest stayed identity.
    moved = np.abs(np.asarray(new_exp["exp"] - exp_p["exp"])).sum((1, 2)) > 0
    assert moved[[0, 3, 5]].all() and not moved[[1, 2, 4, 6, 7]].any()
