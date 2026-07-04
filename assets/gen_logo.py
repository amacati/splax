"""Render assets/logo.svg from assets/logo_tiles.json.

The tile list is the canonical logo source, designed by hand in the lattice
editor. Each tile is a face on the rhombille lattice the JAX wordmark uses
(lattice steps 25 x 43.3, faces spanned by a = (50, 0), b = (25, -43.3),
c = (25, 43.3), plus the four half-face triangles). The letterforms extend
the JAX logo's design language (github.com/jax-ml/jax, Apache-2.0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

LX = 25.0
LY = 50.0 * math.sin(math.radians(60))
A = (50.0, 0.0)
B = (25.0, -LY)
C = (25.0, LY)


def _add(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
    return (p[0] + q[0], p[1] + q[1])


SHAPES = {
    "ab": [(0.0, 0.0), A, _add(A, B), B],
    "ac": [(0.0, 0.0), A, _add(A, C), C],
    "bc": [(0.0, 0.0), B, _add(B, C), C],
    "ta1": [(0.0, 0.0), A, B],
    "ta2": [A, _add(A, B), B],
    "tc1": [(0.0, 0.0), A, C],
    "tc2": [A, _add(A, C), C],
}
STROKE = "#dce0df"
SW = 1.0


def main() -> None:
    tiles = json.loads(Path("assets/logo_tiles.json").read_text())["tiles"]
    polys = []
    for t in tiles:
        pts = [(t["i"] * LX + ox, t["j"] * LY + oy) for ox, oy in SHAPES[t["t"]]]
        polys.append((t["c"], pts))
    xs = [x for _, p in polys for x, _ in p]
    ys = [y for _, p in polys for _, y in p]
    pad = 12.0
    minx, maxx = min(xs) - pad, max(xs) + pad
    miny, maxy = min(ys) - pad, max(ys) + pad
    body = [
        f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in p)}" '
        f'fill="{c}" stroke="{STROKE}" stroke-width="{SW}" stroke-linejoin="round"/>'
        for c, p in polys
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{minx:.2f} {miny:.2f} '
        f'{maxx - minx:.2f} {maxy - miny:.2f}">\n<title>splax</title>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    Path("assets/logo.svg").write_text(svg)
    print(f"{len(polys)} tiles")


if __name__ == "__main__":
    main()
