"""Parity + invariant checks for splax.mcmc (fixed-shape MCMC training ops).

``compute_relocation`` is checked against a direct Python transcription of the
CUDA ``relocation_kernel`` (gsplat/cuda/csrc/RelocationCUDA.cu, Eq. 9). ``relocate``
and ``inject_noise`` are checked for their fixed-shape invariants (shapes stay
static, dead gaussians teleport onto alive ones, high-opacity gaussians barely
move under noise).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from splax import mcmc


def _cuda_relocation_reference(
    opacities: np.ndarray, scales: np.ndarray, ratios: np.ndarray, n_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Direct transcription of the CUDA relocation_kernel double loop (Eq. 9)."""
    binoms = np.array(
        [
            [math.comb(n, k) if k <= n else 0 for k in range(n_max)]
            for n in range(n_max)
        ],
        np.float64,
    )
    new_opac = np.empty_like(opacities)
    new_scales = np.empty_like(scales)
    for idx in range(len(opacities)):
        n = int(min(max(round(ratios[idx]), 1), n_max))
        no = 1.0 - (1.0 - opacities[idx]) ** (1.0 / n)
        new_opac[idx] = no
        denom = 0.0
        for i in range(1, n + 1):
            for k in range(0, i):
                denom += (
                    binoms[i - 1, k] * ((-1.0) ** k / math.sqrt(k + 1)) * no ** (k + 1)
                )
        new_scales[idx] = (opacities[idx] / denom) * scales[idx]
    return new_opac, new_scales


def test_compute_relocation_matches_cuda_kernel() -> None:
    rng = np.random.default_rng(0)
    n = 200
    opac = rng.uniform(0.01, 0.99, n)
    scales = rng.uniform(0.01, 0.5, (n, 3))
    ratios = rng.integers(1, 12, n).astype(np.float64)

    ref_o, ref_s = _cuda_relocation_reference(opac, scales, ratios, n_max=51)
    binoms = mcmc.make_binoms(51)
    got_o, got_s = mcmc.compute_relocation(
        jnp.asarray(opac, jnp.float32),
        jnp.asarray(scales, jnp.float32),
        jnp.asarray(ratios, jnp.float32),
        binoms,
    )

    np.testing.assert_allclose(np.asarray(got_o), ref_o, rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(got_s), ref_s, rtol=2e-3, atol=1e-5)


def test_compute_relocation_ratio_one_is_identity() -> None:
    """ratio == 1 must pass opacity/scale through unchanged (untouched gaussians)."""
    opac = jnp.array([0.1, 0.5, 0.9], jnp.float32)
    scales = jnp.array(
        [[0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [0.05, 0.1, 0.2]], jnp.float32
    )
    ratios = jnp.ones(3, jnp.float32)
    o, s = mcmc.compute_relocation(opac, scales, ratios, mcmc.make_binoms(51))
    np.testing.assert_allclose(np.asarray(o), np.asarray(opac), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(s), np.asarray(scales), rtol=1e-5)


def test_relocate_teleports_dead_onto_alive() -> None:
    n = 500
    k = jax.random.split(jax.random.key(1), 4)
    means = jax.random.uniform(k[0], (n, 3), minval=-1, maxval=1)
    log_scales = jnp.full((n, 3), jnp.log(0.05))
    quats = jax.random.normal(k[1], (n, 4))
    colors_logit = jax.random.normal(k[2], (n, 3))
    # first 100 dead (opacity ~0), rest alive (opacity ~0.7)
    opac_logit = jnp.concatenate([jnp.full((100, 1), -20.0), jnp.full((400, 1), 0.85)])

    binoms = mcmc.make_binoms(51)
    out, reset = mcmc.relocate(
        k[3],
        means,
        log_scales,
        quats,
        colors_logit,
        opac_logit,
        binoms,
        min_opacity=0.005,
    )

    # shapes are static
    assert out["means"].shape == (n, 3)
    assert out["opac_logit"].shape == (n, 1)
    # every dead gaussian was reset and now has opacity above the dead threshold
    reset = np.asarray(reset)
    assert reset[:100].all()
    new_opac = np.asarray(jax.nn.sigmoid(out["opac_logit"]).reshape(-1))
    assert (new_opac[:100] > 0.005).all()
    # relocated means coincide with some alive source position
    alive_means = np.asarray(means[100:])
    for i in range(100):
        d = np.min(np.linalg.norm(alive_means - np.asarray(out["means"][i]), axis=1))
        assert d < 1e-4


def test_inject_noise_respects_opacity() -> None:
    n = 400
    k = jax.random.split(jax.random.key(2), 2)
    means = jnp.zeros((n, 3))
    log_scales = jnp.full((n, 3), jnp.log(0.1))
    quats = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    # half near-transparent, half near-opaque
    opac_logit = jnp.concatenate([jnp.full((200, 1), -5.0), jnp.full((200, 1), 8.0)])
    moved = mcmc.inject_noise(k[0], means, log_scales, quats, opac_logit, scaler=100.0)
    disp = np.linalg.norm(np.asarray(moved), axis=1)
    # low-opacity gaussians move; high-opacity ones barely move
    assert disp[:200].mean() > 10 * disp[200:].mean() + 1e-6
    assert moved.shape == (n, 3)
