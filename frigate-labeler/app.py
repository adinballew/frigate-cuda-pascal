#!/usr/bin/env python3
"""Tiny local YOLO labeler for Frigate custom-model review images.

Serves FrontDoor/Backyard images and writes YOLO txt labels next to them.
No cloud, no external dependencies.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ALLOWED_CAMERAS = {"FrontDoor", "Backyard"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_LABELS = [
    "person", "package", "car", "truck", "van", "dog", "cat", "bird",
    "bicycle", "motorcycle", "backpack", "suitcase", "waste_bin",
]

HTML = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Frigate Package Labeler</title>
<style>
:root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --muted:#8c95a3; --text:#e8edf2; --accent:#63d297; --danger:#ff6b6b; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }
header { display:flex; align-items:center; gap:12px; padding:10px 14px; background:#11151c; border-bottom:1px solid #272c35; position:sticky; top:0; z-index:2; }
header h1 { font-size:16px; margin:0; }
header .stat { color:var(--muted); font-size:13px; }
main { display:grid; grid-template-columns: 1fr 320px; min-height: calc(100vh - 45px); }
#stageWrap { display:flex; align-items:center; justify-content:center; padding:12px; overflow:auto; }
canvas { max-width:100%; max-height:calc(100vh - 80px); background:#050607; border:1px solid #2c3340; cursor:crosshair; }
aside { background:var(--panel); border-left:1px solid #272c35; padding:12px; overflow:auto; }
button, select, input { background:#222834; color:var(--text); border:1px solid #3a4352; border-radius:8px; padding:8px 10px; }
button { cursor:pointer; }
button:hover { border-color:#5b6678; }
button.primary { background:#1f6f48; border-color:#2e9a68; }
button.danger { background:#5a2222; border-color:#8a3333; }
button.full { width:100%; margin:4px 0; }
.row { display:flex; gap:8px; align-items:center; margin:8px 0; flex-wrap:wrap; }
.row > * { flex:1; }
.small { color:var(--muted); font-size:12px; line-height:1.35; }
.card { border:1px solid #2c3340; border-radius:10px; padding:10px; margin:10px 0; background:#12161d; }
#boxes { max-height:220px; overflow:auto; }
.boxItem { display:flex; justify-content:space-between; gap:8px; padding:6px; border-bottom:1px solid #2a303a; font-size:13px; }
.boxItem button { padding:2px 6px; flex:0; }
.kbd { font-family:ui-monospace, SFMono-Regular, monospace; color:#c4ccda; }
.notice { color:#ffd166; }
.ok { color:var(--accent); }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-left:0; border-top:1px solid #272c35; } canvas { max-height:70vh; } }
</style>
</head>
<body>
<header>
  <h1>Frigate Package Labeler</h1>
  <span class="stat" id="counter">loading…</span>
</header>
<main>
  <section id="stageWrap"><canvas id="canvas"></canvas></section>
  <aside>
    <div class="card">
      <div class="small">Image</div>
      <div id="imageName" style="word-break:break-all">—</div>
      <div class="small" id="imageMeta">—</div>
    </div>
    <div class="card">
      <label class="small">Class for new boxes</label>
      <select id="classSelect"></select>
      <div class="small" style="margin-top:8px">Draw: click-drag on image. Delete: use list. Default class is <b>package</b>.</div>
    </div>
    <div class="card">
      <button class="primary full" id="saveBtn">Save labels + Next</button>
      <button class="full" id="emptyBtn">Mark Empty + Next</button>
      <div class="row"><button id="prevBtn">Prev</button><button id="nextBtn">Skip/Next</button></div>
      <button class="danger full" id="clearBtn">Clear boxes</button>
      <div class="small notice" id="status"></div>
    </div>
    <div class="card">
      <div class="small">Boxes</div>
      <div id="boxes"></div>
    </div>
    <div class="card small">
      <div><span class="kbd">S</span> save + next</div>
      <div><span class="kbd">E</span> mark empty + next</div>
      <div><span class="kbd">N</span> next, <span class="kbd">P</span> previous</div>
      <div><span class="kbd">Backspace</span> delete last box</div>
    </div>
  </aside>
</main>
<script>
const token = new URLSearchParams(location.search).get('token') || localStorage.getItem('frigateLabelerToken') || '';
if (token) localStorage.setItem('frigateLabelerToken', token);
const auth = () => token ? `?token=${encodeURIComponent(token)}` : '';
const withAuth = (url) => url + (url.includes('?') ? '&' : '?') + (token ? `token=${encodeURIComponent(token)}` : '');
let images = [], labels = [], idx = 0, boxes = [], img = new Image(), scale = 1;
let drawing = false, start = null, draft = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const cls = document.getElementById('classSelect');
const statusEl = document.getElementById('status');

function setStatus(s, ok=false) { statusEl.textContent = s || ''; statusEl.className = ok ? 'small ok' : 'small notice'; }
function current() { return images[idx]; }
function labelName(id) { return labels[id] || `class_${id}`; }

async function api(path, opts={}) {
  const r = await fetch(withAuth(path), opts);
  if (!r.ok) throw new Error(await r.text());
  return r.headers.get('content-type')?.includes('application/json') ? r.json() : r.text();
}

async function init() {
  const meta = await api('/api/meta');
  labels = meta.labels;
  labels.forEach((name, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${i}: ${name}`; cls.appendChild(o); });
  const packageId = labels.indexOf('package'); if (packageId >= 0) cls.value = packageId;
  images = await api('/api/images');
  if (!images.length) { setStatus('No images found. Collect candidates first.'); return; }
  load(0);
}

async function load(newIdx) {
  idx = Math.max(0, Math.min(images.length - 1, newIdx));
  const item = current();
  document.getElementById('counter').textContent = `${idx+1}/${images.length}`;
  document.getElementById('imageName').textContent = item.name;
  document.getElementById('imageMeta').textContent = `${item.camera} · ${item.bucket} · ${item.status}`;
  boxes = (await api(`/api/labels/${encodeURIComponent(item.camera)}/${encodeURIComponent(item.name)}?bucket=${encodeURIComponent(item.bucket)}`)).boxes || [];
  img = new Image();
  img.onload = () => { resizeCanvas(); draw(); renderBoxes(); setStatus(''); };
  img.src = withAuth(item.url);
}

function resizeCanvas() {
  const maxW = Math.min(window.innerWidth - (window.innerWidth > 900 ? 360 : 24), img.naturalWidth);
  const maxH = Math.min(window.innerHeight - 85, img.naturalHeight);
  scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
}

function toCanvasBox(b) { return { x:(b.x-b.w/2)*img.naturalWidth*scale, y:(b.y-b.h/2)*img.naturalHeight*scale, w:b.w*img.naturalWidth*scale, h:b.h*img.naturalHeight*scale }; }
function fromCanvasRect(x1,y1,x2,y2) {
  const x = Math.max(0, Math.min(x1,x2)) / scale;
  const y = Math.max(0, Math.min(y1,y2)) / scale;
  const w = Math.abs(x2-x1) / scale;
  const h = Math.abs(y2-y1) / scale;
  if (w < 8 || h < 8) return null;
  return { class_id:Number(cls.value), x:(x+w/2)/img.naturalWidth, y:(y+h/2)/img.naturalHeight, w:w/img.naturalWidth, h:h/img.naturalHeight };
}

function draw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  for (const b of boxes) drawBox(b, '#63d297');
  if (draft) drawRect(draft.x, draft.y, draft.w, draft.h, '#ffd166', 'new');
}
function confidenceLabel(b) { return (b.confidence !== null && b.confidence !== undefined) ? ` ${Math.round(Number(b.confidence)*100)}%` : ''; }
function boxDisplayLabel(b) { return `${labelName(b.class_id)}${confidenceLabel(b)}`; }
function drawBox(b, color) { const r = toCanvasBox(b); drawRect(r.x,r.y,r.w,r.h,color,boxDisplayLabel(b)); }
function drawRect(x,y,w,h,color,text) {
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.strokeRect(x,y,w,h);
  ctx.fillStyle='rgba(0,0,0,.65)'; ctx.fillRect(x, Math.max(0,y-20), Math.max(60, text.length*8), 20);
  ctx.fillStyle=color; ctx.font='13px sans-serif'; ctx.fillText(text, x+4, Math.max(13,y-6));
}
function pos(ev) { const r=canvas.getBoundingClientRect(); return {x:ev.clientX-r.left, y:ev.clientY-r.top}; }
canvas.addEventListener('mousedown', ev => { drawing=true; start=pos(ev); draft={x:start.x,y:start.y,w:0,h:0}; });
canvas.addEventListener('mousemove', ev => { if(!drawing) return; const p=pos(ev); draft={x:Math.min(start.x,p.x),y:Math.min(start.y,p.y),w:Math.abs(p.x-start.x),h:Math.abs(p.y-start.y)}; draw(); });
canvas.addEventListener('mouseup', ev => { if(!drawing) return; drawing=false; const p=pos(ev); const b=fromCanvasRect(start.x,start.y,p.x,p.y); draft=null; if(b) boxes.push(b); draw(); renderBoxes(); });

function renderBoxes() {
  const el = document.getElementById('boxes'); el.innerHTML = '';
  boxes.forEach((b,i) => { const d=document.createElement('div'); d.className='boxItem'; d.innerHTML=`<span>${i+1}. ${boxDisplayLabel(b)} ${(b.w*100).toFixed(1)}×${(b.h*100).toFixed(1)}%</span>`; const del=document.createElement('button'); del.textContent='×'; del.onclick=()=>{boxes.splice(i,1); draw(); renderBoxes();}; d.appendChild(del); el.appendChild(d); });
  if (!boxes.length) el.innerHTML = '<div class="small">No boxes.</div>';
}

async function save(empty=false) {
  const item=current();
  await api(`/api/labels/${encodeURIComponent(item.camera)}/${encodeURIComponent(item.name)}?bucket=${encodeURIComponent(item.bucket)}`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({boxes: empty ? [] : boxes})
  });
  setStatus(empty ? 'Saved empty.' : `Saved ${boxes.length} box(es).`, true);
  if (idx < images.length-1) setTimeout(()=>load(idx+1), 150);
}

document.getElementById('saveBtn').onclick = () => save(false);
document.getElementById('emptyBtn').onclick = () => save(true);
document.getElementById('nextBtn').onclick = () => load(idx+1);
document.getElementById('prevBtn').onclick = () => load(idx-1);
document.getElementById('clearBtn').onclick = () => { boxes=[]; draw(); renderBoxes(); };
window.addEventListener('resize', () => { if(img.naturalWidth) { resizeCanvas(); draw(); }});
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'SELECT' || ev.target.tagName === 'INPUT') return;
  if (ev.key === 's' || ev.key === 'S') save(false);
  if (ev.key === 'e' || ev.key === 'E') save(true);
  if (ev.key === 'n' || ev.key === 'N') load(idx+1);
  if (ev.key === 'p' || ev.key === 'P') load(idx-1);
  if (ev.key === 'Backspace') { boxes.pop(); draw(); renderBoxes(); ev.preventDefault(); }
});
init().catch(e => setStatus(String(e)));
</script>
</body>
</html>'''

class App(BaseHTTPRequestHandler):
    server_version = "FrigateLabeler/0.1"

    @property
    def data_root(self) -> Path:
        return self.server.data_root  # type: ignore[attr-defined]

    @property
    def token(self) -> str:
        return self.server.token  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.client_address[0]} - {fmt % args}")

    def _authorized(self) -> bool:
        if not self.token:
            return True
        q = parse_qs(urlparse(self.path).query)
        return q.get("token", [""])[0] == self.token

    def _deny(self):
        self.send_response(HTTPStatus.FORBIDDEN)
        self.end_headers()
        self.wfile.write(b"Forbidden\n")

    def _json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text: str, status=HTTPStatus.OK, ctype="text/plain; charset=utf-8"):
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _labels(self) -> list[str]:
        p = self.data_root / "labels.txt"
        if p.exists():
            labels = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            return labels or DEFAULT_LABELS
        return DEFAULT_LABELS

    def _image_path(self, bucket: str, camera: str, name: str) -> Path | None:
        if camera not in ALLOWED_CAMERAS or "/" in name or ".." in name:
            return None
        candidates = [
            self.data_root / bucket / camera / "images" / name,
            self.data_root / bucket / camera / name,
        ]
        for p in candidates:
            try:
                p.relative_to(self.data_root)
            except ValueError:
                continue
            if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                return p
        return None

    def _label_path_for_image(self, image: Path) -> Path:
        # Expected review layout: <bucket>/<camera>/images/foo.jpg -> <bucket>/<camera>/labels/foo.txt
        if image.parent.name == "images":
            label_dir = image.parent.parent / "labels"
        else:
            label_dir = image.parent / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir / f"{image.stem}.txt"

    def _list_images(self):
        out = []
        for bucket in ["review", "active_learning", "incoming"]:
            root = self.data_root / bucket
            if not root.exists():
                continue
            for camera in sorted(ALLOWED_CAMERAS):
                for base in [root / camera / "images", root / camera]:
                    if not base.exists():
                        continue
                    for img in sorted(base.iterdir()):
                        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                            continue
                        lab = self._label_path_for_image(img)
                        status = "labeled" if lab.exists() and lab.read_text(encoding="utf-8").strip() else ("empty" if lab.exists() else "new")
                        out.append({
                            "bucket": bucket,
                            "camera": camera,
                            "name": img.name,
                            "status": status,
                            "url": f"/image/{bucket}/{camera}/{img.name}",
                        })
        order = {"new": 0, "empty": 1, "labeled": 2}
        return sorted(out, key=lambda x: (order.get(x["status"], 9), x["camera"], x["name"]))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._text(HTML, ctype="text/html; charset=utf-8")
        if not self._authorized():
            return self._deny()
        if path == "/api/meta":
            return self._json({"labels": self._labels(), "cameras": sorted(ALLOWED_CAMERAS)})
        if path == "/api/images":
            return self._json(self._list_images())
        if path.startswith("/image/"):
            parts = path.split("/", 4)
            if len(parts) != 5:
                return self._text("Bad image path\n", HTTPStatus.BAD_REQUEST)
            _, _, bucket, camera, name = parts
            img = self._image_path(unquote(bucket), unquote(camera), unquote(name))
            if not img:
                return self._text("Not found\n", HTTPStatus.NOT_FOUND)
            data = img.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(img.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/api/labels/"):
            parts = path.split("/", 4)
            if len(parts) != 5:
                return self._text("Bad labels path\n", HTTPStatus.BAD_REQUEST)
            _, _, _, camera, name = parts
            bucket = parse_qs(parsed.query).get("bucket", ["review"])[0]
            img = self._image_path(unquote(bucket), unquote(camera), unquote(name))
            if not img:
                return self._text("Not found\n", HTTPStatus.NOT_FOUND)
            lab = self._label_path_for_image(img)
            boxes = []
            if lab.exists():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    parts2 = line.split()
                    if len(parts2) == 5:
                        try:
                            confidence = None
                            for tok in parts2[5:]:
                                if tok.startswith("#"):
                                    tok = tok[1:]
                                if "=" not in tok:
                                    continue
                                key, value = tok.split("=", 1)
                                if key.lower() in {"conf", "confidence", "score"}:
                                    try:
                                        confidence = float(value)
                                    except ValueError:
                                        pass
                            boxes.append({"class_id": int(parts2[0]), "x": float(parts2[1]), "y": float(parts2[2]), "w": float(parts2[3]), "h": float(parts2[4]), "confidence": confidence})
                        except ValueError:
                            pass
            return self._json({"boxes": boxes})
        return self._text("Not found\n", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self._authorized():
            return self._deny()
        path = parsed.path
        if not path.startswith("/api/labels/"):
            return self._text("Not found\n", HTTPStatus.NOT_FOUND)
        parts = path.split("/", 4)
        if len(parts) != 5:
            return self._text("Bad labels path\n", HTTPStatus.BAD_REQUEST)
        _, _, _, camera, name = parts
        bucket = parse_qs(parsed.query).get("bucket", ["review"])[0]
        img = self._image_path(unquote(bucket), unquote(camera), unquote(name))
        if not img:
            return self._text("Not found\n", HTTPStatus.NOT_FOUND)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        boxes = payload.get("boxes", [])
        labels = self._labels()
        lines = []
        for b in boxes:
            cid = int(b["class_id"])
            if cid < 0 or cid >= len(labels):
                return self._text("Bad class id\n", HTTPStatus.BAD_REQUEST)
            vals = [float(b[k]) for k in ["x", "y", "w", "h"]]
            if not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                return self._text("Bad box coordinates\n", HTTPStatus.BAD_REQUEST)
            lines.append(f"{cid} {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}")
        lab = self._label_path_for_image(img)
        lab.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        return self._json({"ok": True, "label": str(lab), "boxes": len(lines)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--data", type=Path, default=Path("/workspace/frigate_custom_model"))
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"data path not found: {args.data}")
    token = os.environ.get("LABELER_TOKEN", "")
    server = ThreadingHTTPServer((args.host, args.port), App)
    server.data_root = args.data
    server.token = token
    print(f"Frigate labeler on http://{args.host}:{args.port}/ data={args.data} token={'set' if token else 'off'}", flush=True)
    server.serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
