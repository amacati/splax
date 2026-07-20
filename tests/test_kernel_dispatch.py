"""Validate symbolic zeros kernel dispatch.

The projection backward is a single jax.custom_vjp with symbolic_zeros. Its
forward rule records statically which inputs are perturbed (any of
means/scales/quats maps to gaussians, viewmat to viewmat) and the backward rule
branches in Python between three FFI callables, resolved as module globals at
call time so a test can monkeypatch them to record which fired. Asserted here:

  * a viewmat-only grad launches the viewmat accumulator and no gaussian kernel,
  * a gaussian-only grad launches the gaussian kernel and no viewmat accumulator,
  * a joint grad launches the joint kernel only,
  * a plain forward launches none.

The wrappers call through, so the recorded run is also numerically valid.
"""

from __future__ import annotations

from typing import Callable, ParamSpec, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import pytest

import splax
from splax._project import _project


class _PK(TypedDict):
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float


FnParams = ParamSpec("FnParams")
FnReturn = TypeVar("FnReturn")


def _pk(H: int, W: int) -> _PK:
    return {
        "img_shape": (H, W),
        "f": (float(H), float(H)),
        "c": (W // 2, H // 2),
        "glob_scale": 1.0,
        "clip_thresh": 0.01,
    }


def _scene(
    n: int, H: int, W: int, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(seed)
    k = jax.random.split(key, 6)
    means = jax.random.normal(k[0], (n, 3)) * 0.5
    scales = jax.random.uniform(k[1], (n, 3), minval=0.02, maxval=0.08)
    quats = jax.random.normal(k[2], (n, 4))
    quats = quats / jnp.linalg.norm(quats, axis=-1, keepdims=True)
    colors = jax.random.uniform(k[3], (n, 3))
    opac = jax.random.uniform(k[4], (n, 1), minval=0.1, maxval=0.6)
    bg = jax.random.uniform(k[5], (3,))
    vm = jnp.array([[1, 0, 0, 0.2], [0, 1, 0, -0.1], [0, 0, 1, 5], [0, 0, 0, 1]], jnp.float32)
    return means, scales, quats, colors, opac, bg, vm


@pytest.fixture
def record(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Record projection backward call sites.

    Swap the three backward FFI callables for recording wrappers that still
    call through and return the set of names that fired.
    """
    fired: set[str] = set()

    def wrap(name: str, fn: Callable[FnParams, FnReturn]) -> Callable[FnParams, FnReturn]:
        def wrapped(*args: FnParams.args, **kwargs: FnParams.kwargs) -> FnReturn:
            fired.add(name)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        _project,
        "_project_bwd_gaussians_ffi",
        wrap("gaussians", _project._project_bwd_gaussians_ffi),
    )
    monkeypatch.setattr(
        _project, "_project_bwd_viewmat_ffi", wrap("view", _project._project_bwd_viewmat_ffi)
    )
    monkeypatch.setattr(
        _project, "_project_bwd_joint_ffi", wrap("both", _project._project_bwd_joint_ffi)
    )
    return fired


def _render(
    m: jax.Array,
    s: jax.Array,
    q: jax.Array,
    c: jax.Array,
    o: jax.Array,
    bg: jax.Array,
    v: jax.Array,
    H: int,
    W: int,
) -> jax.Array:
    return splax.render(m, s, q, c, o, viewmat=v, background=bg, **_pk(H, W))[0]


def test_viewmat_only_launches_view_kernel(record: set[str]) -> None:
    n, H, W = 400, 64, 64
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)

    def loss(v: jax.Array) -> jax.Array:
        return jnp.sum(_render(means, scales, quats, colors, opac, bg, v, H, W))

    jax.grad(loss)(vm).block_until_ready()
    assert record == {"view"}, record


def test_gaussians_only_launches_gaussian_kernel(record: set[str]) -> None:
    n, H, W = 400, 64, 64
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)

    def loss(m: jax.Array) -> jax.Array:
        return jnp.sum(_render(m, scales, quats, colors, opac, bg, vm, H, W))

    jax.grad(loss)(means).block_until_ready()
    assert record == {"gaussians"}, record


def test_joint_launches_both_kernel(record: set[str]) -> None:
    n, H, W = 400, 64, 64
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)

    def loss(m: jax.Array, v: jax.Array) -> jax.Array:
        return jnp.sum(_render(m, scales, quats, colors, opac, bg, v, H, W))

    gm, gv = jax.grad(loss, argnums=(0, 1))(means, vm)
    gm.block_until_ready()
    gv.block_until_ready()
    assert record == {"both"}, record


def test_forward_launches_no_backward_kernel(record: set[str]) -> None:
    n, H, W = 400, 64, 64
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)
    img = _render(means, scales, quats, colors, opac, bg, vm, H, W)
    img.block_until_ready()
    assert record == set(), record


def test_colors_only_launches_no_projection_kernel(record: set[str]) -> None:
    """Check that colors only gradients skip projection backward kernels."""
    n, H, W = 400, 64, 64
    means, scales, quats, colors, opac, bg, vm = _scene(n, H, W, seed=1)

    def loss(c: jax.Array) -> jax.Array:
        return jnp.sum(_render(means, scales, quats, c, opac, bg, vm, H, W))

    jax.grad(loss)(colors).block_until_ready()
    assert record == set(), record
