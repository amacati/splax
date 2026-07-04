"""splax.mcmc: fixed-budget MCMC training utilities (port of gsplat MCMCStrategy).

Ports the essential mechanics of *"3D Gaussian Splatting as Markov Chain Monte
Carlo"* (Kheradmand et al., NeurIPS 2024) as **fixed-shape** JAX ops, so a JAX
pipeline that needs static array shapes (no densification that grows ``N``) can
still get MCMC-style training:

- ``relocate`` teleports dead (low-opacity) gaussians onto alive ones sampled in
  proportion to opacity, correcting opacity/scale so the multiplicity that now
  shares a location preserves the original's contribution (Eq. 9 of the paper),
  and returns a boolean ``reset`` mask (which rows' optimizer moments to zero).
- ``inject_noise`` adds covariance- and opacity-weighted Gaussian noise to the
  means every step (annealed by the caller via ``scaler``), so low-opacity
  gaussians random-walk to explore while high-opacity ones stay put.

Every op is a mask + gather + scatter on the full ``N`` -- no dynamic shapes --
so it composes with ``jax.jit`` and a fixed optax state. Ports the CUDA
``relocation_kernel`` (gsplat/cuda/csrc/RelocationCUDA.cu) and the PyTorch
``relocate`` / ``inject_noise_to_position`` (gsplat/strategy/ops.py) faithfully;
see ``tests/test_mcmc.py`` for the parity check against a direct transcription of
the CUDA Eq. 9 loop.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_EPS = float(jnp.finfo(jnp.float32).eps)


def make_binoms(n_max: int = 51) -> jax.Array:
    """Binomial-coefficient lookup table ``b[n, k] = C(n, k)`` (gsplat convention)."""
    b = [[math.comb(n, k) if k <= n else 0 for k in range(n_max)] for n in range(n_max)]
    return jnp.asarray(b, jnp.float32)


def compute_relocation(
    opacities: jax.Array, scales: jax.Array, ratios: jax.Array, binoms: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Eq. (9) of the MCMC paper: new opacity/scale for a relocated multiplicity.

    Args mirror gsplat's ``compute_relocation`` (the CUDA ``relocation_kernel``):
      opacities (N,)   activation-space opacity of the source gaussians.
      scales    (N, 3) activation-space per-axis scales of the sources.
      ratios    (N,)   how many gaussians now share each source's location.
      binoms  (M, M)   ``make_binoms`` table (``M`` = n_max).

    Returns ``(new_opacities (N,), new_scales (N, 3))``. ``ratio == 1`` is the
    identity, so it is safe to apply to *every* gaussian (untouched ones pass
    through unchanged). Vectorized: the CUDA double loop over ``i in 1..n`` is
    folded into ``cumsum(binoms)`` indexed by ``ratio - 1``.
    """
    n_max = binoms.shape[0]
    ratios = jnp.clip(jnp.round(ratios).astype(jnp.int32), 1, n_max)
    new_opac = 1.0 - (1.0 - opacities) ** (1.0 / ratios)
    k = jnp.arange(n_max, dtype=jnp.float32)
    sign_sqrt = ((-1.0) ** k) / jnp.sqrt(k + 1.0)  # (-1)^k / sqrt(k+1)
    # cb[m, k] = sum_{i=1..m+1} binoms[i-1, k]; sum_{i=1..n} binoms[i-1, k] = cb[n-1, k]
    cb = jnp.cumsum(binoms, axis=0)
    cbr = cb[ratios - 1]  # (N, n_max)
    powers = new_opac[:, None] ** (k + 1.0)  # new_opac^(k+1)
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
    """Relocate dead gaussians onto alive ones (fixed ``N``); port of gsplat ``relocate``.

    Operates on the *raw* training parameters (opacity/scale stored as
    logit/log). Dead = ``sigmoid(opac_logit) <= min_opacity``. Every gaussian
    samples a source among the alive ones with probability proportional to
    opacity; dead gaussians copy their source's mean/quat/color and take the
    Eq.-9-corrected opacity/scale, while a source chosen by ``c`` dead gaussians
    has its own opacity/scale corrected for the resulting multiplicity ``c + 1``.
    Alive, unchosen gaussians pass through untouched (``ratio == 1``).

    Returns ``(new_params_dict, reset_mask)`` where ``new_params_dict`` has the
    same keys as the inputs and ``reset_mask`` (N,) marks rows whose optimizer
    moments the caller should zero (sources + dead copies).
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
    new_opac, new_scale = compute_relocation(
        opac[source], scales[source], ratio, binoms
    )
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
        "opac_logit": jnp.where(
            reset.reshape(opac_logit.shape), new_opac_logit, opac_logit
        ),
    }
    return out, reset


def _quat_to_rotmat(quats: jax.Array) -> jax.Array:
    """Normalized wxyz quaternions -> rotation matrices (N, 3, 3) (gsplat convention)."""
    q = quats / (jnp.linalg.norm(quats, axis=-1, keepdims=True) + _EPS)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return jnp.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def inject_noise(
    key: jax.Array,
    means: jax.Array,
    log_scales: jax.Array,
    quats: jax.Array,
    opac_logit: jax.Array,
    scaler: float,
) -> jax.Array:
    """Add covariance/opacity-weighted Gaussian noise to ``means`` (gsplat ``inject_noise_to_position``).

    ``noise = Sigma @ (randn * op_sigmoid(1 - opacity) * scaler)`` with
    ``Sigma = R diag(scale^2) R^T`` and ``op_sigmoid(x) = sigmoid(100 (x - 0.995))``
    -- low-opacity gaussians get near-full noise, high-opacity ones ~zero. The
    caller anneals via ``scaler = means_lr(step) * noise_lr``. Returns new means.
    """
    opac = jax.nn.sigmoid(opac_logit).reshape(means.shape[0])
    scales = jnp.exp(log_scales)
    rot = _quat_to_rotmat(quats)
    m = rot * scales[:, None, :]  # R diag(scale); Sigma = m m^T
    op_sig = jax.nn.sigmoid(100.0 * ((1.0 - opac) - 0.995))
    noise = jax.random.normal(key, means.shape) * (op_sig * scaler)[:, None]
    noise = jnp.einsum("nij,nkj,nk->ni", m, m, noise)  # Sigma @ noise
    return means + noise


__all__ = ["make_binoms", "compute_relocation", "relocate", "inject_noise"]
