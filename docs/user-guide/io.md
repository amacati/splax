# IO

`splax.io` reads and writes 3DGS `.ply` files in render space. The two functions
are exact inverses of each other.

## Loading

`splax.load_ply` reads a 3DGS `.ply` and maps the stored activation-space fields
to the render-space arrays that `render` consumes:

```
scales  = exp(scale_i)
quats   = normalize(rot_i)
colors  = clip(f_dc_i * C0 + 0.5, 0, 1)
opac    = sigmoid(opacity)
```

It returns `(means, scales, quats, colors, opacities)` as float32 JAX arrays with
shapes `(N, 3)`, `(N, 3)`, `(N, 4)`, `(N, 3)`, `(N, 1)`.

```python
means, scales, quats, colors, opacities = splax.load_ply("scene.ply")
```

## Writing

`splax.write_ply` takes the render-space arrays, the same tensors `render`
consumes, and writes the inverse activation-space fields.

```python
splax.write_ply("out.ply", means, scales, quats, colors, opacities)
```

Opacities may be passed as `(N, 1)` or `(N,)`.

## Round-trip and SH degree 0

splax renders spherical harmonics of degree 0 only, a single per-gaussian color.
On write, normals are zeroed and the higher-order SH field `f_rest` is omitted
because `load_ply` reads neither. A file written by `write_ply` therefore
round-trips exactly back through `load_ply`.
