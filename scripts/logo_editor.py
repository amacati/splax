"""Web-based lattice editor for the splax logo.

Serves the editor at http://127.0.0.1:<port> and writes every edit straight back to the JSON file.

Controls (also shown in the toolbar):
  1-7                  select face shape (ab, ac, bc, ta1, ta2, tc1, tc2)
  q w e r t y u i o    select fill color
  left click           place the selected face at the nearest lattice anchor
  right click          delete the face under the cursor
  drag on empty space  rubber-band select tiles (shift adds to the selection)
  shift+click          toggle a tile in the selection
  drag a selected tile move the whole selection (snapped to the lattice)
  arrow keys           move the selection one lattice period
  delete / backspace   delete the selection
  ctrl+c / x / v       copy / cut the selection, paste at the mouse cursor
  escape               clear the selection
  middle drag / wheel  pan / zoom the canvas (f fits the view)
  z / shift+z          undo / redo (ctrl+z / ctrl+shift+z work too)

Usage:
  pixi run python scripts/logo_editor.py [--file docs/assets/logo_tiles.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ("ab", "ac", "bc", "ta1", "ta2", "tc1", "tc2")

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>splax logo editor</title>
<style>
  body { margin: 0; font: 13px system-ui, sans-serif; display: flex;
         flex-direction: column; height: 100vh; }
  #bar { display: flex; align-items: center; gap: 6px; padding: 8px 12px;
         border-bottom: 1px solid #dce0df; flex-wrap: wrap; }
  #bar button { border: 1px solid #b8bfbd; border-radius: 4px; background: #fff;
                padding: 2px 6px; cursor: pointer; display: flex;
                flex-direction: column; align-items: center; gap: 1px; }
  #bar button.sel { border: 2px solid #000; padding: 1px 5px; }
  #bar button .key { color: #667; font-size: 10px; }
  .swatch { width: 26px; height: 20px; border-radius: 2px; }
  #file { margin-left: auto; color: #99a; font-size: 11px; }
  #status { color: #667; }
  #hint { color: #99a; font-size: 11px; width: 100%; }
  svg { flex: 1; cursor: crosshair; }
  polygon.sel { stroke: #111; stroke-width: 2.5; }
</style>
</head>
<body>
<div id="bar"></div>
<svg id="cv"></svg>
<script>
"use strict";
const LX = 25, LY = 50 * Math.sin(Math.PI / 3);
const A = [50, 0], B = [25, -LY], C = [25, LY];
const add = (p, q) => [p[0] + q[0], p[1] + q[1]];
const SHAPES = {
  ab: [[0, 0], A, add(A, B), B],
  ac: [[0, 0], A, add(A, C), C],
  bc: [[0, 0], B, add(B, C), C],
  ta1: [[0, 0], A, B],
  ta2: [A, add(A, B), B],
  tc1: [[0, 0], A, C],
  tc2: [A, add(A, C), C],
};
const SHAPE_KEYS = Object.fromEntries(Object.keys(SHAPES).map((s, n) => [n + 1, s]));
const PALETTE = ["#5e97f6", "#3367d6", "#2a56c6", "#26a69a", "#00796b",
                 "#00695c", "#ea80fc", "#9c27b0", "#6a1b9a"];
const COLOR_KEYS = Object.fromEntries([..."qwertyuio"].map((k, n) => [k, PALETTE[n]]));
const STROKE = "#dce0df";

let filePath = __DEFAULT_FILE__;
let tiles = [], undoStack = [], redoStack = [];
let shape = "ab", color = PALETTE[0];
const selected = new Set();
let drag = null, pan = null, view = null;
let clipboard = [], cursor = null;

const cv = document.getElementById("cv");
const bar = document.getElementById("bar");
const NS = "http://www.w3.org/2000/svg";
const dotsG = document.createElementNS(NS, "g");
const tilesG = document.createElementNS(NS, "g");
const preview = document.createElementNS(NS, "polygon");
const band = document.createElementNS(NS, "rect");
dotsG.style.pointerEvents = "none";
preview.style.pointerEvents = "none";
preview.setAttribute("opacity", "0.45");
preview.setAttribute("visibility", "hidden");
band.style.pointerEvents = "none";
band.setAttribute("fill", "rgba(94,151,246,0.15)");
band.setAttribute("stroke", "#5e97f6");
band.setAttribute("stroke-dasharray", "4 3");
band.setAttribute("visibility", "hidden");
cv.append(dotsG, tilesG, preview, band);

function tilePts(t) {
  return SHAPES[t.t].map(([ox, oy]) => `${t.i * LX + ox},${t.j * LY + oy}`).join(" ");
}

function centroid(t) {
  const pts = SHAPES[t.t];
  const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  return [t.i * LX + cx, t.j * LY + cy];
}

function pointInPoly(px, py, pts) {
  let inside = false;
  for (let a = 0, b = pts.length - 1; a < pts.length; b = a++) {
    const [x1, y1] = pts[a], [x2, y2] = pts[b];
    if ((y1 > py) !== (y2 > py) && px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
      inside = !inside;
  }
  return inside;
}

// Anchor whose face (of the selected shape) contains the cursor; in the gaps
// where no face of that shape does, the face whose centroid is nearest. Faces
// anchor at their left corner, so snapping to the nearest anchor would jump to
// the next face as soon as the cursor leaves a face's left half.
function snapFace(x, y) {
  const pts0 = SHAPES[shape];
  const cx = pts0.reduce((a, p) => a + p[0], 0) / pts0.length;
  const cy = pts0.reduce((a, p) => a + p[1], 0) / pts0.length;
  const ci = Math.round((x - cx) / LX), cj = Math.round((y - cy) / LY);
  let best = null, bestD = Infinity;
  for (let i = ci - 3; i <= ci + 3; i++) {
    for (let j = cj - 2; j <= cj + 2; j++) {
      if (((i + j) % 2 + 2) % 2 !== 1) continue;
      const pts = pts0.map(([px, py]) => [i * LX + px, j * LY + py]);
      if (pointInPoly(x, y, pts)) return [i, j];
      const dd = (i * LX + cx - x) ** 2 + (j * LY + cy - y) ** 2;
      if (dd < bestD) { bestD = dd; best = [i, j]; }
    }
  }
  return best;
}

// Lattice translations must keep i + j parity, so di + dj is snapped to even.
function snapDelta(dx, dy) {
  let di = Math.round(dx / LX), dj = Math.round(dy / LY);
  if (((di + dj) % 2 + 2) % 2 !== 0) {
    const cands = [[di + 1, dj], [di - 1, dj], [di, dj + 1], [di, dj - 1]];
    const d = ([ci, cj]) => (ci * LX - dx) ** 2 + (cj * LY - dy) ** 2;
    [di, dj] = cands.reduce((a, b) => d(a) < d(b) ? a : b);
  }
  return [di, dj];
}

function fitView() {
  const ii = tiles.map(t => t.i), jj = tiles.map(t => t.j);
  const i0 = Math.min(0, ...ii) - 2, i1 = Math.max(30, ...ii) + 4;
  const j0 = Math.min(0, ...jj) - 2, j1 = Math.max(6, ...jj) + 3;
  view = { x: i0 * LX, y: j0 * LY, w: (i1 - i0) * LX, h: (j1 - j0) * LY };
}

function setViewBox() {
  cv.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
}

function render() {
  if (!view) fitView();
  setViewBox();
  dotsG.innerHTML = "";
  const i0 = Math.floor(view.x / LX) - 1, i1 = Math.ceil((view.x + view.w) / LX) + 1;
  const j0 = Math.floor(view.y / LY) - 1, j1 = Math.ceil((view.y + view.h) / LY) + 1;
  if ((i1 - i0) * (j1 - j0) < 20000) {  // skip the grid when zoomed way out
    for (let gi = i0; gi <= i1; gi++) {
      for (let gj = j0; gj <= j1; gj++) {
        if (((gi + gj) % 2 + 2) % 2 !== 1) continue;
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", gi * LX);
        dot.setAttribute("cy", gj * LY);
        dot.setAttribute("r", "1.2");
        dot.setAttribute("fill", "#c5cac8");
        dotsG.append(dot);
      }
    }
  }
  tilesG.innerHTML = "";
  tiles.forEach((t, n) => {
    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("points", tilePts(t));
    poly.setAttribute("fill", t.c);
    poly.setAttribute("stroke", STROKE);
    poly.setAttribute("stroke-width", "1");
    poly.setAttribute("stroke-linejoin", "round");
    poly.dataset.idx = n;
    if (selected.has(t)) poly.classList.add("sel");
    tilesG.append(poly);
  });
}

function setStatus(msg) { document.getElementById("status").textContent = msg; }
function setFileLabel() { document.getElementById("file").textContent = filePath; }

async function save() {
  try {
    const resp = await fetch("/tiles", {
      method: "POST", body: JSON.stringify({ tiles, path: filePath }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    setStatus(`${tiles.length} tiles · saved`);
  } catch (err) {
    setStatus(`SAVE FAILED: ${err.message}`);
  }
}

function commit() { save(); render(); }
function pushUndo() { undoStack.push(JSON.stringify(tiles)); redoStack = []; }
function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(tiles));
  tiles = JSON.parse(undoStack.pop());
  selected.clear();
  commit();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(tiles));
  tiles = JSON.parse(redoStack.pop());
  selected.clear();
  commit();
}

function moveSelection(di, dj) {
  for (const t of selected) { t.i += di; t.j += dj; }
  const keys = new Set([...selected].map(t => `${t.t},${t.i},${t.j}`));
  tiles = tiles.filter(t => selected.has(t) || !keys.has(`${t.t},${t.i},${t.j}`));
}

async function loadFile() {
  const p = prompt("Load tiles JSON from path:", filePath);
  if (!p) return;
  const resp = await fetch(`/tiles?path=${encodeURIComponent(p)}`);
  if (!resp.ok) { alert(await resp.text()); return; }
  tiles = (await resp.json()).tiles;
  filePath = p;
  selected.clear();
  undoStack = [];
  redoStack = [];
  fitView();
  render();
  setStatus(`${tiles.length} tiles loaded`);
  setFileLabel();
}

async function saveAs() {
  const p = prompt("Save tiles JSON to path:", filePath);
  if (!p) return;
  filePath = p;
  setFileLabel();
  await save();
}

function svgPoint(e) {
  return new DOMPoint(e.clientX, e.clientY).matrixTransform(cv.getScreenCTM().inverse());
}

cv.addEventListener("mousedown", e => {
  if (e.button === 1) {  // middle button: pan
    e.preventDefault();
    pan = { m: cv.getScreenCTM().inverse(), cx: e.clientX, cy: e.clientY,
            x0: view.x, y0: view.y };
    preview.setAttribute("visibility", "hidden");
    return;
  }
  if (e.button !== 0) return;
  e.preventDefault();
  const p = svgPoint(e);
  const idx = e.target.dataset?.idx;
  const t = idx !== undefined ? tiles[+idx] : null;
  if (!e.shiftKey && t && selected.has(t)) drag = { p0: p, mode: "move", di: 0, dj: 0 };
  else drag = { p0: p, p1: p, mode: "band", tile: t, shift: e.shiftKey };
  drag.moved = false;
});

cv.addEventListener("mousemove", e => {
  const p = svgPoint(e);
  cursor = p;
  if (drag) return;
  const [i, j] = snapFace(p.x, p.y);
  preview.setAttribute("points", tilePts({ t: shape, i, j }));
  preview.setAttribute("fill", color);
  preview.setAttribute("visibility", "visible");
});
cv.addEventListener("mouseleave", () => preview.setAttribute("visibility", "hidden"));

window.addEventListener("mousemove", e => {
  if (pan) {
    const p0 = new DOMPoint(pan.cx, pan.cy).matrixTransform(pan.m);
    const p1 = new DOMPoint(e.clientX, e.clientY).matrixTransform(pan.m);
    view.x = pan.x0 - (p1.x - p0.x);
    view.y = pan.y0 - (p1.y - p0.y);
    setViewBox();
    return;
  }
  if (!drag) return;
  const p = svgPoint(e);
  const dx = p.x - drag.p0.x, dy = p.y - drag.p0.y;
  if (!drag.moved && dx * dx + dy * dy < 16) return;
  drag.moved = true;
  preview.setAttribute("visibility", "hidden");
  if (drag.mode === "move") {
    [drag.di, drag.dj] = snapDelta(dx, dy);
    for (const poly of tilesG.children) {
      if (selected.has(tiles[+poly.dataset.idx]))
        poly.setAttribute("transform", `translate(${drag.di * LX} ${drag.dj * LY})`);
    }
  } else {
    drag.p1 = p;
    band.setAttribute("x", Math.min(p.x, drag.p0.x));
    band.setAttribute("y", Math.min(p.y, drag.p0.y));
    band.setAttribute("width", Math.abs(dx));
    band.setAttribute("height", Math.abs(dy));
    band.setAttribute("visibility", "visible");
  }
});

window.addEventListener("mouseup", e => {
  if (pan && e.button === 1) { pan = null; render(); return; }
  if (!drag || e.button !== 0) return;
  const d = drag;
  drag = null;
  band.setAttribute("visibility", "hidden");
  if (d.mode === "move") {
    for (const poly of tilesG.children) poly.removeAttribute("transform");
    if (d.moved && (d.di || d.dj)) {
      pushUndo();
      moveSelection(d.di, d.dj);
      commit();
    }
    return;
  }
  if (d.moved) {  // rubber-band select: tiles whose centroid is in the box
    if (!d.shift) selected.clear();
    const x0 = Math.min(d.p0.x, d.p1.x), x1 = Math.max(d.p0.x, d.p1.x);
    const y0 = Math.min(d.p0.y, d.p1.y), y1 = Math.max(d.p0.y, d.p1.y);
    for (const t of tiles) {
      const [cx, cy] = centroid(t);
      if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) selected.add(t);
    }
    render();
    setStatus(`${selected.size} selected`);
  } else if (d.shift) {
    if (!d.tile) return;
    selected.has(d.tile) ? selected.delete(d.tile) : selected.add(d.tile);
    render();
    setStatus(`${selected.size} selected`);
  } else if (selected.size) {  // click-off deselects instead of placing
    selected.clear();
    render();
    setStatus("");
  } else {
    const [i, j] = snapFace(d.p0.x, d.p0.y);
    pushUndo();
    tiles = tiles.filter(t => t.t !== shape || t.i !== i || t.j !== j);
    tiles.push({ t: shape, i, j, c: color });
    commit();
  }
});

cv.addEventListener("wheel", e => {  // scroll: zoom around the cursor
  e.preventDefault();
  const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const p = svgPoint(e);
  view.x = p.x - (p.x - view.x) * f;
  view.y = p.y - (p.y - view.y) * f;
  view.w *= f;
  view.h *= f;
  render();
}, { passive: false });

cv.addEventListener("contextmenu", e => {
  e.preventDefault();
  const idx = e.target.dataset?.idx;
  if (idx === undefined) return;
  pushUndo();
  selected.delete(tiles[+idx]);
  tiles.splice(+idx, 1);
  commit();
});

function buildBar() {
  for (const [key, s] of Object.entries(SHAPE_KEYS)) {
    const btn = document.createElement("button");
    btn.dataset.shape = s;
    const pts = SHAPES[s];
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const [w, h] = [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
    btn.innerHTML =
      `<svg width="34" height="22" viewBox="${Math.min(...xs) - 2} ${Math.min(...ys) - 2} ` +
      `${w + 4} ${h + 4}"><polygon points="${pts.map(p => p.join(",")).join(" ")}" ` +
      `fill="#dce0df" stroke="#8a9490"/></svg><span class="key">${key}:${s}</span>`;
    btn.onclick = () => { shape = s; refreshBar(); };
    bar.append(btn);
  }
  for (const [key, c] of Object.entries(COLOR_KEYS)) {
    const btn = document.createElement("button");
    btn.dataset.color = c;
    btn.innerHTML = `<span class="swatch" style="background:${c}"></span>` +
      `<span class="key">${key}</span>`;
    btn.onclick = () => { color = c; refreshBar(); };
    bar.append(btn);
  }
  const actions = [["undo (z)", undo], ["redo (shift+z)", redo],
                   ["load\\u2026", loadFile], ["save as\\u2026", saveAs]];
  for (const [label, fn] of actions) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.onclick = fn;
    bar.append(btn);
  }
  const file = document.createElement("span");
  file.id = "file";
  const status = document.createElement("span");
  status.id = "status";
  const hint = document.createElement("span");
  hint.id = "hint";
  hint.textContent = "left click: place · right click: delete · drag: select · " +
    "shift+click: toggle · drag selection: move · arrows: move selection · " +
    "del: delete selection · esc: deselect · ctrl+c/x/v: copy/cut/paste at " +
    "cursor · middle drag: pan · wheel: zoom · f: fit · 1-7: shape · q-o: color";
  bar.append(file, status, hint);
  refreshBar();
  setFileLabel();
}

function refreshBar() {
  for (const btn of bar.querySelectorAll("button")) {
    btn.classList.toggle("sel",
      btn.dataset.shape === shape || btn.dataset.color === color);
  }
  preview.setAttribute("fill", color);
}

const MOVES = { ArrowLeft: [-2, 0], ArrowRight: [2, 0],
                ArrowUp: [0, -2], ArrowDown: [0, 2] };

document.addEventListener("keydown", e => {
  if (e.key === "Escape") { selected.clear(); render(); setStatus(""); return; }
  if ((e.ctrlKey || e.metaKey) && (e.key === "c" || e.key === "x") && selected.size) {
    clipboard = [...selected].map(t => ({ ...t }));
    if (e.key === "x") {
      pushUndo();
      tiles = tiles.filter(t => !selected.has(t));
      selected.clear();
      commit();
    } else {
      setStatus(`${clipboard.length} copied`);
    }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "v" && clipboard.length) {
    e.preventDefault();
    let di = 2, dj = 2;  // fallback offset when the cursor is off-canvas
    if (cursor) {  // paste at the cursor, snapped to a valid lattice shift
      const cs = clipboard.map(centroid);
      const cx = cs.reduce((a, c) => a + c[0], 0) / cs.length;
      const cy = cs.reduce((a, c) => a + c[1], 0) / cs.length;
      [di, dj] = snapDelta(cursor.x - cx, cursor.y - cy);
    }
    pushUndo();
    const pasted = clipboard.map(t => ({ ...t, i: t.i + di, j: t.j + dj }));
    const keys = new Set(pasted.map(t => `${t.t},${t.i},${t.j}`));
    tiles = tiles.filter(t => !keys.has(`${t.t},${t.i},${t.j}`));
    tiles.push(...pasted);
    selected.clear();
    pasted.forEach(t => selected.add(t));
    commit();
    return;
  }
  if ((e.key === "Delete" || e.key === "Backspace") && selected.size) {
    pushUndo();
    tiles = tiles.filter(t => !selected.has(t));
    selected.clear();
    commit();
    return;
  }
  if (MOVES[e.key] && selected.size) {
    e.preventDefault();
    pushUndo();
    moveSelection(...MOVES[e.key]);
    commit();
    return;
  }
  if (e.key === "z" || e.key === "Z") { e.shiftKey ? redo() : undo(); return; }
  if (e.key === "y") { redo(); return; }
  if (e.key === "f") { fitView(); render(); return; }
  if (SHAPE_KEYS[e.key]) shape = SHAPE_KEYS[e.key];
  else if (COLOR_KEYS[e.key]) color = COLOR_KEYS[e.key];
  else return;
  refreshBar();
});

buildBar();
fetch(`/tiles?path=${encodeURIComponent(filePath)}`).then(r => r.json()).then(d => {
  tiles = d.tiles;
  render();
  setStatus(`${tiles.length} tiles loaded`);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    """Serve the logo editor page and tile endpoints."""

    tile_file: Path

    def _reply(self, code: int, body: bytes, ctype: str) -> None:
        """Send an HTTP response with explicit content metadata."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, path: str | None) -> Path:
        """Resolve a requested tile path against the repository root."""
        if path is None:
            return self.tile_file
        p = Path(path).expanduser()
        return p if p.is_absolute() else ROOT / p

    def do_GET(self) -> None:  # noqa: N802
        """Serve editor HTML or tile JSON."""
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            page = PAGE.replace("__DEFAULT_FILE__", json.dumps(str(self.tile_file)))
            self._reply(200, page.encode(), "text/html; charset=utf-8")
        elif url.path == "/tiles":
            query = urllib.parse.parse_qs(url.query)
            file = self._resolve(query["path"][0] if "path" in query else None)
            if not file.exists():
                self._reply(404, f"{file} not found".encode(), "text/plain")
                return
            try:
                body = file.read_bytes()
                _validate(json.loads(body))
            except (KeyError, TypeError, ValueError) as e:
                self._reply(400, f"{file}: {e}".encode(), "text/plain")
                return
            self._reply(200, body, "application/json")
        else:
            self._reply(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        """Persist updated tile JSON from the editor."""
        if self.path != "/tiles":
            self._reply(404, b"not found", "text/plain")
            return
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            data = json.loads(raw)
            tiles = _validate(data)
            file = self._resolve(data.get("path"))
            file.write_text(json.dumps({"tiles": tiles}, indent=1))
        except (KeyError, TypeError, ValueError, OSError) as e:
            self._reply(400, str(e).encode(), "text/plain")
            return
        self._reply(200, b'{"ok": true}', "application/json")

    def log_message(self, format: str, *args: object) -> None:
        """Silence default HTTP request logging."""
        pass


def _validate(data: dict) -> list[dict]:
    """Validate and normalize tile payload data."""
    tiles = []
    for t in data["tiles"]:
        ok = (
            t["t"] in SHAPES
            and isinstance(t["i"], int)
            and isinstance(t["j"], int)
            and re.fullmatch(r"#[0-9a-f]{6}", t["c"])
        )
        if not ok:
            raise ValueError(f"bad tile: {t}")
        tiles.append({"t": t["t"], "i": t["i"], "j": t["j"], "c": t["c"]})
    return tiles


def main() -> None:
    """Run the local logo tile editor server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=ROOT / "docs/assets/logo_tiles.json")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()
    Handler.tile_file = args.file.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    logger.info(f"editing {args.file}\nserving at {url}\nctrl+c to stop")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
