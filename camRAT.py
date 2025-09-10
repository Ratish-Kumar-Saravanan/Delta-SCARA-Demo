import logging
import time
from threading import Lock

import cv2
import mss
import numpy as np
import pygetwindow as gw
from flask import Flask, Response, make_response, render_template_string, request

app = Flask(__name__)

# ---- Config ----
FPS = 10
JPEG_QUALITY = 70
WINDOW_TITLE = "DMV-Viewer"   # substring that must appear in the window title
BIND_IP = "0.0.0.0"      # LAN IP to bind (use "0.0.0.0" to listen on all)
PORT = 8081

# ---- ROI settings ----
# Choose cropping mode: "relative" (fractions 0..1) or "absolute" (pixels)
ROI_MODE = "relative"

# Absolute crop (x, y, w, h) in *window* pixels (fallback/reference)
ABS_ROI = (1454, 540, 460, 377)

# Relative crop (left, top, right, bottom) as fractions of the window bbox
# Replace with your calibration values if you want them as default:
REL_ROI = (0.748713, 0.518234, 0.985582, 0.880038)

# Optional border clamp (pixels) after ROI (avoids tiny border artifacts)
ROI_CLAMP = 0

# Runtime overrides (so you can change ROI via /set_roi without restarting)
ROI_RUNTIME_MODE = None        # "absolute" or "relative"
ROI_RUNTIME_ABS  = None        # (x,y,w,h)
ROI_RUNTIME_REL  = None        # (l,t,r,b)

lock = Lock()

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------- Window / ROI helpers ----------------

def find_window_bbox():
    """Find the on-screen bounding box for a window with title containing WINDOW_TITLE."""
    wins = [w for w in gw.getWindowsWithTitle(WINDOW_TITLE) if WINDOW_TITLE in w.title]
    if not wins:
        raise ValueError(f"Window '{WINDOW_TITLE}' not found")
    w = wins[0]
    if w.isMinimized:
        raise ValueError(f"Window '{w.title}' is minimized")
    left, top, right, bottom = w.left, w.top, w.right, w.bottom
    return {"title": w.title, "top": top, "left": left, "width": right - left, "height": bottom - top}

def _roi_from_absolute(bbox, abs_roi):
    """Absolute (x,y,w,h) inside the window bbox -> ROI dict for mss."""
    L, T, W, H = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
    x, y, w, h = [int(v) for v in abs_roi]
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return {"left": L + x, "top": T + y, "width": w, "height": h}

def _roi_from_relative(bbox, rel_roi):
    """Relative (l,t,r,b) fractions (0..1) -> ROI dict for mss."""
    L, T, W, H = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
    l, t, r, b = [float(v) for v in rel_roi]
    l = max(0.0, min(1.0, l)); t = max(0.0, min(1.0, t))
    r = max(0.0, min(1.0, r)); b = max(0.0, min(1.0, b))
    x0 = int(L + W * min(l, r)); y0 = int(T + H * min(t, b))
    x1 = int(L + W * max(l, r)); y1 = int(T + H * max(t, b))
    w  = max(1, x1 - x0); h = max(1, y1 - y0)
    return {"left": x0, "top": y0, "width": w, "height": h}

def _apply_roi_to_bbox(bbox):
    """Return ROI dict (top,left,width,height) using runtime override if present."""
    mode = (ROI_RUNTIME_MODE or ROI_MODE or "relative").lower()
    if mode == "absolute":
        abs_roi = ROI_RUNTIME_ABS if ROI_RUNTIME_ABS else ABS_ROI
        roi = _roi_from_absolute(bbox, abs_roi)
    else:
        rel_roi = ROI_RUNTIME_REL if ROI_RUNTIME_REL else REL_ROI
        roi = _roi_from_relative(bbox, rel_roi)

    if ROI_CLAMP > 0:
        roi["left"]  += ROI_CLAMP
        roi["top"]   += ROI_CLAMP
        roi["width"]  = max(1, roi["width"]  - 2 * ROI_CLAMP)
        roi["height"] = max(1, roi["height"] - 2 * ROI_CLAMP)
    return roi

def grab_frame(sct):
    """Grab a cropped frame of the target window, or return an error frame and the exception."""
    try:
        bbox_win = find_window_bbox()
        bbox_roi = _apply_roi_to_bbox(bbox_win)
        shot = sct.grab(bbox_roi)
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        return img, None
    except Exception as e:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, str(e), (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, e

def grab_full_window(sct):
    """Grab the full window image as BGR, plus its bbox dict."""
    bbox_win = find_window_bbox()
    shot = sct.grab({k: bbox_win[k] for k in ("top", "left", "width", "height")})
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    return img, bbox_win

def encode_jpeg(img):
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return jpg.tobytes()

def mjpeg_generator():
    frame_interval = 1.0 / max(FPS, 1)
    with mss.mss() as sct:
        while True:
            t0 = time.time()
            img, err = grab_frame(sct)
            if err:
                app.logger.warning("Frame error: %s", err)
            try:
                frame = encode_jpeg(img)
            except Exception as e:
                app.logger.error("JPEG encode error: %s", e)
                time.sleep(0.2)
                continue

            with lock:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                )

            dt = time.time() - t0
            if dt < frame_interval:
                time.sleep(frame_interval - dt)


# ---------------- Views ----------------

# Full-screen viewer (fills screen; no black bars via object-fit:cover)
@app.route("/")
def index():
    return render_template_string("""
    <!doctype html><title>DMV-Viewer (Cover)</title>
    <style>html,body{margin:0;height:100%}
    img{width:100vw;height:100vh;object-fit:cover;display:block;background:#000}</style>
    <img src="/stream.mjpg" alt="DMV-Viewer stream">
    """)

@app.route("/stream.mjpg")
def stream():
    resp = Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# Single snapshot of the current cropped frame
@app.route("/snapshot.jpg")
def snapshot():
    with mss.mss() as sct:
        img, _ = grab_frame(sct)
        jpg = encode_jpeg(img)
    r = make_response(jpg)
    r.headers["Content-Type"] = "image/jpeg"
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Content-Disposition"] = "inline; filename=snapshot.jpg"
    return r

# Full window snapshot (for measuring & debugging)
@app.route("/full_window.jpg")
def full_window_jpg():
    with mss.mss() as sct:
        img, _ = grab_full_window(sct)
        jpg = encode_jpeg(img)
    r = make_response(jpg)
    r.headers["Content-Type"] = "image/jpeg"
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return r

# Grid helper drawn over the full window + current ROI outline (green)
@app.route("/snapshot_grid.jpg")
def snapshot_grid():
    try:
        with mss.mss() as sct:
            base, bbox_win = grab_full_window(sct)
    except Exception:
        with mss.mss() as sct:
            base, _ = grab_frame(sct)

    h, w = base.shape[:2]
    step = max(50, min(200, w // 12))
    for x in range(0, w, step):
        cv2.line(base, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(base, str(x), (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(base, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(base, str(y), (3, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    try:
        bbox_win = find_window_bbox()
        roi = _apply_roi_to_bbox(bbox_win)
        # Convert screen coords to window-local for drawing on 'base'
        winL, winT = bbox_win["left"], bbox_win["top"]
        x = roi["left"] - winL
        y = roi["top"] - winT
        ww = roi["width"]
        hh = roi["height"]
        cv2.rectangle(base, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.putText(base, f"ROI {ww}x{hh}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception:
        pass

    jpg = encode_jpeg(base)
    r = make_response(jpg)
    r.headers["Content-Type"] = "image/jpeg"
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return r

# JS-free fallback snapshot "video" (works in old IE/WebView engines)
@app.route("/view_meta")
def view_meta():
    refresh_sec = 0.2
    ts = int(time.time() * 1000)
    return render_template_string(f"""
    <!doctype html><title>DMV-Viewer (Meta Refresh)</title>
    <meta http-equiv="refresh" content="{refresh_sec}; url=/view_meta">
    <style>html,body{{height:100%;margin:0;background:#111;display:grid;place-items:center}}
    img{{max-width:100vw;max-height:100vh}}
    .note{{position:fixed;top:10px;left:10px;color:#aaa;font:14px/1.4 system-ui,sans-serif}}</style>
    <div class="note">Fallback: meta-refresh ~{1/refresh_sec:.1f} fps</div>
    <img src="/snapshot.jpg?ts={ts}" alt="DMV snapshot">
    """)

# Some legacy containers prefer <object> for MJPEG
@app.route("/view_object")
def view_object():
    return render_template_string("""
    <!doctype html><title>DMV-Viewer (Object MJPEG)</title>
    <style>html,body{height:100%;margin:0;background:#111} object{width:100%;height:100%}</style>
    <object data="/stream.mjpg" type="image/jpeg"></object>
    """)

# Interactive calibrator (click 2 corners; see/copy ABS/REL; apply live)
@app.route("/measure")
def measure():
    return render_template_string("""
    <!doctype html>
    <title>ROI Calibrator</title>
    <style>
      html,body{height:100%;margin:0;background:#111;color:#ddd;font:14px/1.4 system-ui,sans-serif}
      .wrap{display:grid;grid-template-columns:1fr 360px;gap:12px;height:100%}
      .pane{display:grid;place-items:center;overflow:auto}
      #img{max-width:100%;height:auto;cursor:crosshair;background:#000}
      .side{padding:12px}
      pre{white-space:pre-wrap;background:#1e1e1e;padding:10px;border-radius:8px}
      .btn{display:inline-block;padding:6px 10px;margin-right:8px;border:1px solid #444;border-radius:8px;cursor:pointer}
      .btn:hover{background:#222}
      .row{margin-bottom:8px}
      code{color:#a6e22e}
    </style>
    <div class="wrap">
      <div class="pane">
        <img id="img" src="/full_window.jpg?ts="+Date.now() alt="full window">
      </div>
      <div class="side">
        <div class="row">
          <span class="btn" id="refresh">Refresh</span>
          <span class="btn" id="reset">Reset clicks</span>
        </div>
        <div class="row">
          <strong>Mouse:</strong>
          <div>pixel: <code id="pix">(x,y)</code></div>
          <div>relative: <code id="rel">(l,t)</code></div>
        </div>
        <div class="row">
          <strong>Clicks:</strong>
          <div>1️⃣ top-left: <code id="p1">–</code></div>
          <div>2️⃣ bottom-right: <code id="p2">–</code></div>
        </div>
        <div class="row">
          <strong>Suggested ROI:</strong>
          <pre id="out">Click two corners on the image to compute ABS_ROI and REL_ROI...</pre>
        </div>
        <div class="row">
          <strong>Apply now (optional):</strong><br>
          <span class="btn" id="applyAbs">Apply ABS_ROI</span>
          <span class="btn" id="applyRel">Apply REL_ROI</span>
          <div id="applyMsg"></div>
        </div>
      </div>
    </div>
    <script>
    const img = document.getElementById('img');
    const pix = document.getElementById('pix');
    const rel = document.getElementById('rel');
    const p1El = document.getElementById('p1');
    const p2El = document.getElementById('p2');
    const out = document.getElementById('out');
    const applyMsg = document.getElementById('applyMsg');
    let naturalW = 0, naturalH = 0;
    let click1 = null, click2 = null;

    function updMouse(e){
      if (!naturalW || !naturalH) return;
      const r = img.getBoundingClientRect();
      let x = Math.round((e.clientX - r.left) * (naturalW / r.width));
      let y = Math.round((e.clientY - r.top)  * (naturalH / r.height));
      x = Math.max(0, Math.min(naturalW-1, x));
      y = Math.max(0, Math.min(naturalH-1, y));
      const lx = (x / naturalW).toFixed(4);
      const ty = (y / naturalH).toFixed(4);
      pix.textContent = `(${x}, ${y})`;
      rel.textContent = `(${lx}, ${ty})`;
    }

    function showOutput(){
      if (!click1 || !click2) { out.textContent = "Click two corners on the image to compute ABS_ROI and REL_ROI..."; return; }
      const x0 = Math.min(click1.x, click2.x);
      const y0 = Math.min(click1.y, click2.y);
      const x1 = Math.max(click1.x, click2.x);
      const y1 = Math.max(click1.y, click2.y);
      const w  = x1 - x0;
      const h  = y1 - y0;
      const l = (x0 / naturalW).toFixed(6);
      const t = (y0 / naturalH).toFixed(6);
      const r = (x1 / naturalW).toFixed(6);
      const b = (y1 / naturalH).toFixed(6);
      out.textContent =
`ABS_ROI = (${x0}, ${y0}, ${w}, ${h})
REL_ROI = (${l}, ${t}, ${r}, ${b})

Paste into config:
  ROI_MODE = "absolute"
  ABS_ROI  = (${x0}, ${y0}, ${w}, ${h})

or
  ROI_MODE = "relative"
  REL_ROI  = (${l}, ${t}, ${r}, ${b})
`;
    }

    img.addEventListener('mousemove', updMouse);
    img.addEventListener('click', (e)=>{
      if (!naturalW || !naturalH) return;
      const r = img.getBoundingClientRect();
      let x = Math.round((e.clientX - r.left) * (naturalW / r.width));
      let y = Math.round((e.clientY - r.top)  * (naturalH / r.height));
      if (!click1) { click1 = {x,y}; p1El.textContent = `(${x}, ${y})`; }
      else { click2 = {x,y}; p2El.textContent = `(${x}, ${y})`; }
      showOutput();
    });

    document.getElementById('reset').onclick = ()=>{
      click1 = click2 = null; p1El.textContent='–'; p2El.textContent='–'; showOutput();
    };
    document.getElementById('refresh').onclick = ()=>{
      img.src = '/full_window.jpg?ts=' + Date.now();
    };

    async function apply(mode){
      if (!click1 || !click2) { applyMsg.textContent = "Click two corners first."; return; }
      const x0 = Math.min(click1.x, click2.x);
      const y0 = Math.min(click1.y, click2.y);
      const x1 = Math.max(click1.x, click2.x);
      const y1 = Math.max(click1.y, click2.y);
      const qs = new URLSearchParams({mode});
      if (mode === 'absolute') {
        qs.set('x', x0); qs.set('y', y0); qs.set('w', x1 - x0); qs.set('h', y1 - y0);
      } else {
        qs.set('l', (x0/naturalW)); qs.set('t', (y0/naturalH));
        qs.set('r', (x1/naturalW)); qs.set('b', (y1/naturalH));
      }
      const res = await fetch('/set_roi?' + qs.toString());
      const data = await res.json();
      applyMsg.textContent = data.message || 'OK';
    }
    document.getElementById('applyAbs').onclick = ()=>apply('absolute');
    document.getElementById('applyRel').onclick = ()=>apply('relative');

    img.addEventListener('load', ()=>{
      const probe = new Image();
      probe.onload = ()=>{ naturalW = probe.naturalWidth; naturalH = probe.naturalHeight; };
      probe.src = img.src;
    });
    </script>
    """)

# Apply ROI at runtime (no restart)
@app.route("/set_roi")
def set_roi():
    global ROI_RUNTIME_MODE, ROI_RUNTIME_ABS, ROI_RUNTIME_REL
    mode = (request.args.get("mode") or "").lower()
    if mode == "absolute":
        x = int(float(request.args.get("x", 0)))
        y = int(float(request.args.get("y", 0)))
        w = int(float(request.args.get("w", 1)))
        h = int(float(request.args.get("h", 1)))
        ROI_RUNTIME_MODE = "absolute"
        ROI_RUNTIME_ABS  = (x, y, w, h)
        ROI_RUNTIME_REL  = None
        return {"ok": True, "message": f"Applied ABS_ROI=({x},{y},{w},{h})"}
    elif mode == "relative":
        l = float(request.args.get("l", 0))
        t = float(request.args.get("t", 0))
        r = float(request.args.get("r", 1))
        b = float(request.args.get("b", 1))
        ROI_RUNTIME_MODE = "relative"
        ROI_RUNTIME_REL  = (l, t, r, b)
        ROI_RUNTIME_ABS  = None
        return {"ok": True, "message": f"Applied REL_ROI=({l:.6f},{t:.6f},{r:.6f},{b:.6f})"}
    else:
        return {"ok": False, "message": "Specify mode=absolute or mode=relative"}, 400

# Health check
@app.route("/health")
def health():
    try:
        bbox_win = find_window_bbox()
        bbox_roi = _apply_roi_to_bbox(bbox_win)
        return {
            "ok": True,
            "window": bbox_win["title"],
            "window_bbox": {k: bbox_win[k] for k in ("left", "top", "width", "height")},
            "roi_bbox": bbox_roi,
            "mode": ROI_RUNTIME_MODE or ROI_MODE,
            "runtime_abs": ROI_RUNTIME_ABS,
            "runtime_rel": ROI_RUNTIME_REL,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "window_title": WINDOW_TITLE}, 200


# ---------------- App start ----------------

if __name__ == "__main__":
    app.run(host=BIND_IP, port=PORT, threaded=True)
