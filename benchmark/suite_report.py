"""Build the splax vs gsplat benchmark suite report as a multi-page PDF.

Reads ``reports/benchmark_suite.json`` written by ``bench_suite.py`` and renders a
cover page with the run metadata, one page per scenario with render time, throughput,
and peak-memory curves plus a sample render, and a summary page tabulating the speedup
and memory ratios. The JSON is the source of truth; this module only draws it.

    pixi run -e tests python benchmark/suite_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from matplotlib.axes import Axes

SPLAX_C = "#1b9e77"
GSPLAT_C = "#d95f02"
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports"


def _curve(rows: list[dict], fw: str, key: str) -> np.ndarray:
    return np.array([r[fw][key] for r in rows], float)


def cover_page(pdf: PdfPages, data: dict) -> None:
    """Cover page with the run metadata and the scenes benchmarked."""
    meta = data["meta"]
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5, 0.88, "splax vs gsplat benchmark suite", ha="center", fontsize=26, fontweight="bold"
    )
    fig.text(
        0.5,
        0.83,
        "Forward render throughput, batch scaling, and peak GPU memory",
        ha="center",
        fontsize=13,
        color="#555",
    )
    lines = [
        f"GPU: {meta['gpu']}",
        f"jax {meta['jax_version']}   gsplat {meta['gsplat_version']}   "
        f"torch {meta['torch_version']}",
        f"JAX preallocation off: {meta['no_jax_preallocation']}",
        f"warmup {meta['warmup']}, iters {meta['iters']}, best of {meta['repeat']}",
        f"batches: {', '.join(str(b) for b in meta['batches'])}",
        f"metric: {meta['metric']}",
        f"generated: {meta['generated']}",
    ]
    y = 0.70
    for line in lines:
        fig.text(0.12, y, line, ha="left", fontsize=12)
        y -= 0.035
    y -= 0.02
    fig.text(0.12, y, "Scenarios", ha="left", fontsize=14, fontweight="bold")
    y -= 0.04
    for sc in data["scenarios"]:
        fig.text(0.12, y, f"- {sc['name']}: {sc['description']}", ha="left", fontsize=11)
        y -= 0.032
    y -= 0.02
    for line in _wrap(meta["memory_note"], 110):
        fig.text(0.12, y, line, ha="left", fontsize=9.5, color="#666")
        y -= 0.028
    pdf.savefig(fig)
    plt.close(fig)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def scenario_page(pdf: PdfPages, base: Path, sc: dict) -> None:
    """One page per scenario: sample render plus time, throughput, and memory curves."""
    rows = sc["rows"]
    batches = np.array([r["batch"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{sc['name']}  -  {sc['n_gaussians']:,} gaussians, "
        f"{sc['img_shape'][0]}x{sc['img_shape'][1]}, {sc['cameras']} cameras",
        fontsize=14,
        fontweight="bold",
    )

    ax_img = axes[0, 0]
    ax_img.axis("off")
    img_path = base / sc["sample_render"]
    if img_path.exists():
        ax_img.imshow(iio.imread(img_path))
    ax_img.set_title("splax sample render (view 0)", fontsize=10)

    ax_t = axes[0, 1]
    ax_t.plot(batches, _curve(rows, "splax", "time_ms"), "o-", color=SPLAX_C, label="splax")
    ax_t.plot(batches, _curve(rows, "gsplat", "time_ms"), "s-", color=GSPLAT_C, label="gsplat")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("batch size (cameras)")
    ax_t.set_ylabel("render time per call (ms)")
    ax_t.set_title("Render time vs batch")

    ax_thru = axes[1, 0]
    ax_thru.plot(
        batches, _curve(rows, "splax", "throughput_ips"), "o-", color=SPLAX_C, label="splax"
    )
    ax_thru.plot(
        batches, _curve(rows, "gsplat", "throughput_ips"), "s-", color=GSPLAT_C, label="gsplat"
    )
    ax_thru.set_xscale("log", base=2)
    ax_thru.set_xlabel("batch size (cameras)")
    ax_thru.set_ylabel("throughput (images / s)")
    ax_thru.set_title("Throughput vs batch")

    ax_m = axes[1, 1]
    ax_m.plot(
        batches, _curve(rows, "splax", "peak_bytes") / 1e6, "o-", color=SPLAX_C, label="splax (jax)"
    )
    ax_m.plot(
        batches,
        _curve(rows, "gsplat", "peak_bytes") / 1e6,
        "s-",
        color=GSPLAT_C,
        label="gsplat (torch)",
    )
    ax_m.set_xscale("log", base=2)
    ax_m.set_xlabel("batch size (cameras)")
    ax_m.set_ylabel("peak allocator memory (MB)")
    ax_m.set_title("Peak GPU memory vs batch")

    for ax in (ax_t, ax_thru, ax_m):
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf: PdfPages, data: dict) -> None:
    """Table of gsplat/splax time speedup and splax/gsplat memory ratio per batch."""
    scenarios = data["scenarios"]
    batches = data["meta"]["batches"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    fig.suptitle("Summary: splax relative to gsplat", fontsize=14, fontweight="bold")

    def table(ax: Axes, key: str, title: str, fmt: str) -> None:
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=12)
        col_labels = ["scenario"] + [f"b={b}" for b in batches]
        cells = []
        for sc in scenarios:
            by_batch = {r["batch"]: r for r in sc["rows"]}
            row = [sc["name"]]
            for b in batches:
                r = by_batch.get(b)
                row.append(format(r[key], fmt) if r and r[key] is not None else "-")
            cells.append(row)
        tbl = ax.table(cellText=cells, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)

    table(
        ax1,
        "speedup_gsplat_over_splax",
        "Time speedup  (gsplat ms / splax ms, >1 means splax faster)",
        ".2f",
    )
    table(
        ax2,
        "mem_ratio_splax_over_gsplat",
        "Peak memory ratio  (splax bytes / gsplat bytes, <1 means splax leaner)",
        ".2f",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def build_report(data: dict, out: Path) -> None:
    """Render the full PDF report from the loaded benchmark data."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        cover_page(pdf, data)
        for sc in data["scenarios"]:
            scenario_page(pdf, out.parent, sc)
        summary_page(pdf, data)


def main() -> None:
    """Build the PDF from an existing benchmark JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=OUT_DIR / "benchmark_suite.json")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "benchmark_suite.pdf")
    args = ap.parse_args()
    data = json.loads(args.json.read_text())
    build_report(data, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
