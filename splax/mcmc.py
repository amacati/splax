"""Fixed-budget MCMC training utilities.

This module ports the core mechanics of "3D Gaussian Splatting as Markov Chain Monte Carlo" by
Kheradmand et al. from gsplat to JAX. Every op keeps the gaussian count fixed and works as a mask,
gather, and scatter over the full array, so it composes with ``jax.jit`` and a static optax state
without the densification that grows the gaussian count in the original strategy.

``relocate`` teleports dead gaussians onto alive ones sampled in proportion to opacity. It corrects
opacity and scale so the gaussians that now share a location preserve the original contribution, and
it reports which rows need their optimizer moments reset. ``inject_noise`` perturbs the means with
covariance and opacity weighted Gaussian noise every step, so low-opacity gaussians random-walk to
explore the scene while high-opacity ones stay put.

The relocation math matches gsplat's CUDA relocation kernel, and ``tests/test_mcmc.py`` checks
parity against a direct transcription of it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

_EPS = float(jnp.finfo(jnp.float32).eps)


def make_binoms(n_max: int = 51) -> jax.Array:
    """Build the binomial coefficient lookup table ``b[n, k] = C(n, k)``.

    The table follows the gsplat convention, with zeros for ``k > n``.

    Args:
        n_max: Number of rows and columns of the table.

    Returns:
        Lookup table of shape ``(n_max, n_max)`` as float32.
    """
    b = [[math.comb(n, k) if k <= n else 0 for k in range(n_max)] for n in range(n_max)]
    return jnp.asarray(b, jnp.float32)


def compute_relocation(
    opacities: jax.Array, scales: jax.Array, ratios: jax.Array, binoms: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute the corrected opacity and scale for a relocated multiplicity.

    Each entry receives the opacity and scale that preserve the source gaussian's contribution when
    ``ratio`` gaussians share its location.

    Args:
        opacities: Activation-space opacities of the source gaussians, shape ``(N,)``.
        scales: Activation-space per-axis scales of the sources, shape ``(N, 3)``.
        ratios: Number of gaussians sharing each source's location, shape ``(N,)``.
        binoms: Binomial table from ``make_binoms``.

    Returns:
        Tuple of the new opacities with shape ``(N,)`` and the new scales with shape ``(N, 3)``.
    """
    n_max = binoms.shape[0]
    ratios = jnp.clip(jnp.round(ratios).astype(jnp.int32), 1, n_max)
    new_opac = 1.0 - (1.0 - opacities) ** (1.0 / ratios)
    k = jnp.arange(n_max, dtype=jnp.float32)
    sign_sqrt = ((-1.0) ** k) / jnp.sqrt(k + 1.0)
    cb = jnp.cumsum(binoms, axis=0)
    cbr = cb[ratios - 1]  # (N, n_max)
    powers = new_opac[:, None] ** (k + 1.0)
    denom = jnp.sum(cbr * sign_sqrt[None, :] * powers, axis=1)
    coeff = opacities / denom
    return new_opac, coeff[:, None] * scales


def relocate(
    key: jax.Array,
    means: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    colors_logit: jax.Array,
    opac_logit: jax.Array,
    binoms: jax.Array,
    min_opacity: float = 0.005,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Relocate dead gaussians onto alive ones at a fixed gaussian count.

    A gaussian counts as dead when its activated opacity falls to ``min_opacity`` or below. Every
    gaussian samples a source among the alive ones with probability proportional to opacity. Dead
    gaussians copy the mean, quaternion, and color of their source and take the corrected opacity
    and scale from ``compute_relocation``, while a source chosen by dead gaussians has its own
    opacity and scale corrected for the resulting multiplicity. Alive gaussians that were not chosen
    pass through untouched.

    Args:
        key: PRNG key for the source sampling.
        means: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        colors_logit: Color logits, shape ``(N, 3)``.
        opac_logit: Opacity logits, one entry per gaussian.
        binoms: Binomial table from ``make_binoms``.
        min_opacity: Opacity threshold at or below which a gaussian counts as dead.

    Returns:
        Tuple of the new parameter dict, with the same keys as the inputs, and a boolean mask of
        shape ``(N,)`` marking the rows whose optimizer moments the caller should zero.
    """
    n = means.shape[0]
    opac = jax.nn.sigmoid(opac_logit).reshape(n)
    scales = jnp.exp(log_scales)
    dead = opac <= min_opacity
    # sample a source per gaussian from the alive set, weighted by opacity
    logits = jnp.where(dead, -jnp.inf, jnp.log(opac + _EPS))
    src = jax.random.categorical(key, logits, shape=(n,))
    # counts[s] = number of dead gaussians that chose source s
    counts = jnp.zeros(n, jnp.float32).at[src].add(dead.astype(jnp.float32))
    idx = jnp.arange(n)
    source = jnp.where(dead, src, idx)  # dead pull from src, others self
    ratio = jnp.where(dead, counts[src], counts[idx]) + 1.0
    new_opac, new_scale = compute_relocation(opac[source], scales[source], ratio, binoms)
    new_opac = jnp.clip(new_opac, min_opacity, 1.0 - _EPS)
    new_opac_logit = jnp.log(new_opac / (1.0 - new_opac)).reshape(opac_logit.shape)
    new_log_scales = jnp.log(new_scale)

    reset = dead | (counts > 0)  # rows that actually changed
    m = reset[:, None]
    out = {
        "means": jnp.where(m, means[source], means),
        "quats": jnp.where(m, quats[source], quats),
        "colors_logit": jnp.where(m, colors_logit[source], colors_logit),
        "log_scales": jnp.where(m, new_log_scales, log_scales),
        "opac_logit": jnp.where(reset.reshape(opac_logit.shape), new_opac_logit, opac_logit),
    }
    return out, reset


def inject_noise(
    key: jax.Array,
    means: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    opac_logit: jax.Array,
    scaler: float,
) -> jax.Array:
    """Add covariance and opacity weighted Gaussian noise to the means.

    Each gaussian draws its noise from its own covariance and attenuates it with a steep sigmoid of
    the opacity, so gaussians with low opacity receive close to the full perturbation while opaque
    ones barely move. The caller anneals the strength through ``scaler``, typically as the means
    learning rate at the current step times a noise factor.

    Args:
        key: PRNG key for the noise.
        means: Gaussian centers, shape ``(N, 3)``.
        log_scales: Log of the per-axis scales, shape ``(N, 3)``.
        quats: Rotations as wxyz quaternions, shape ``(N, 4)``.
        opac_logit: Opacity logits, one entry per gaussian.
        scaler: Global noise strength.

    Returns:
        The perturbed means, shape ``(N, 3)``.
    """
    opac = jax.nn.sigmoid(opac_logit).reshape(means.shape[0])
    scales = jnp.exp(log_scales)
    rot = R.from_quat(quats, scalar_first=True).as_matrix()
    m = rot * scales[:, None, :]  # R diag(scale) with Sigma = m m^T
    op_sig = jax.nn.sigmoid(100.0 * ((1.0 - opac) - 0.995))
    noise = jax.random.normal(key, means.shape) * (op_sig * scaler)[:, None]
    noise = jnp.einsum("nij,nkj,nk->ni", m, m, noise)  # Sigma @ noise
    return means + noise
