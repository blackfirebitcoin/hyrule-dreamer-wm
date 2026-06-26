#!/usr/bin/env python3
"""Live interactive play harness for the hyrule-dreamer world model.

aiohttp serves an arcade frontend + a /ws WebSocket that streams generated frames
and accepts D-pad/keyboard actions. Runtime-switchable sampler: Heun or DPM8
(DPM-Solver++(2M)). Directional-only. Seed menu: real spawn screens + synthetic
OOD "stress" patterns (encoded through the frozen tokenizer) that the WM diffuses
from. Inference-only — no training/mutation. Press SAVE to export the last 30s.

    python play.py \
        --wm weights/hyrule_dreamer_wm.pt \
        --tokenizer weights/f4_ego_tokenizer.pt \
        --seeds-dir assets/seeds

then open http://localhost:9300 . Needs a CUDA GPU for smooth (~10fps) play.
"""
from __future__ import annotations
import argparse, asyncio, base64, io, json, random, sys, threading, time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from aiohttp import web, WSMsgType
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.seqwm_causal.solvers import sample_clip_solver
from src.seqwm_causal import diffusion as dfn
from infer import load_wm, load_tokenizer, decode_latents

SOLVER_MAP = {"heun": ("heun", 8), "dpm8": ("dpmpp2m", 8)}
ACTIONS = {0: "noop", 5: "Up", 6: "Down", 7: "Left", 8: "Right"}


# ----------------------------------------------------------------- stress seeds
def stress_patterns(n=64) -> dict:
    """64x64x3 uint8 OOD test images the WM has never seen."""
    y, x = np.mgrid[0:n, 0:n]
    p = {}
    p["checkerboard"] = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    p["fine grating"] = (((x % 4) < 2) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    r = np.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2)
    p["rings"] = ((np.sin(r / 2.2) > 0) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    ang = np.arctan2(y - n / 2, x - n / 2)
    p["spokes"] = (((np.sin(ang * 12) > 0)) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    bars = np.zeros((n, n, 3), np.uint8)
    cols = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
            (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
    for i, c in enumerate(cols):
        bars[:, i * n // 8:(i + 1) * n // 8] = c
    p["color bars"] = bars
    rng = np.random.default_rng(7)
    p["RGB static"] = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    p["diagonals"] = ((((x + y) % 10) < 5) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    p["spiral"] = ((np.sin(r / 2 + ang * 4) > 0) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    herm = np.full((n, n, 3), 255, np.uint8)
    herm[(x % 12 < 9) & (y % 12 < 9)] = 0
    p["hermann grid"] = herm
    grad = np.zeros((n, n, 3), np.uint8)
    grad[..., 0] = x * 4 % 256; grad[..., 1] = y * 4 % 256; grad[..., 2] = (x + y) * 2 % 256
    p["gradient"] = grad
    return p


class Dream:
    def __init__(self, model, decoder, cfg, edm, device, seeds, n_ctx=8, sigma_max=8.0):
        self.model, self.decoder, self.cfg, self.edm = model, decoder, cfg, edm
        self.device, self.n_ctx, self.sigma_max = device, n_ctx, sigma_max
        self.seeds = seeds
        self.by_id = {s["id"]: s for s in seeds}
        self.current_action = 0
        self.current_solver = "dpm8"
        self.project = False          # VAE latent re-projection (drift stabilizer)
        self.frame_idx = 0
        self.epoch = 0
        self.lock = threading.Lock()
        self.cuda = device.type == "cuda"
        self.reset(None)

    @torch.no_grad()
    def _project(self, z):
        """Pull a committed latent back onto the valid-f4 manifold: decode then
        re-encode (fp32). Reduces closed-loop off-manifold drift; verified to
        keep the dream alive (obedience preserved) rather than freeze it."""
        img = self.decoder.decode(z.unsqueeze(0).float()).clamp(-1, 1)
        mean, _ = self.decoder.encode(img)
        return mean[0]

    def reset(self, seed_id):
        if seed_id is None or seed_id not in self.by_id:
            reals = [s["id"] for s in self.seeds if s["kind"] == "real"]
            choices = [r for r in reals if r != getattr(self, "seed_id", None)] or reals
            seed_id = random.choice(choices)
        s = self.by_id[seed_id]
        with self.lock:                       # atomic vs an in-flight step()
            self.epoch += 1
            self.seed_id, self.label = seed_id, s["label"]
            self.lat = deque([s["ctx"][i] for i in range(self.n_ctx)], maxlen=96)
            self.act = deque(list(s["act"]), maxlen=96)
            self.current_action = 0           # spawn at rest; no carried-over input
            self.frame_idx = 0

    @torch.no_grad()
    def step(self) -> np.ndarray:
        with self.lock:
            ep = self.epoch
            ctx = torch.stack(list(self.lat)[-self.n_ctx:], dim=0).unsqueeze(0)
            acts = list(self.act)[-self.n_ctx:] + [self.current_action]
            action = self.current_action
        act_t = torch.tensor([acts], dtype=torch.long, device=self.device)
        solver, steps = SOLVER_MAP[self.current_solver]
        with torch.autocast("cuda", enabled=self.cuda):
            out = sample_clip_solver(self.model, ctx, act_t, self.cfg, self.edm,
                                     self.n_ctx, steps, self.sigma_max, self.device,
                                     solver, horizon=1)
        new_lat = out[0, self.n_ctx].detach()
        if self.project:                      # manifold re-projection (toggle)
            new_lat = self._project(new_lat)
        with self.lock:
            if ep != self.epoch:              # a reset happened mid-generation
                return self._decode(self.lat[-1])   # discard stale frame, show fresh seed
            self.lat.append(new_lat)
            self.act.append(action)
            self.frame_idx += 1
        return self._decode(new_lat)

    def _decode(self, lat) -> np.ndarray:
        return decode_latents(self.decoder, lat.unsqueeze(0).to(self.device), self.device)[0]


def png_b64(hwc: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(hwc).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def build_seeds(model_f4, data_dir, n_seeds, device, n_ctx=8):
    """Simple seed loader: take the first real latent frame of N clips and prime
    the dream from that still frame (spawn at rest). No RAM/window filtering."""
    seeds, previews = [], {}
    files = sorted(Path(data_dir).glob("*.pt"))[:max(1, n_seeds)]
    for i, f in enumerate(files):
        clip = torch.load(f, map_location="cpu", weights_only=False)
        lat = clip["obs_latent"].float()
        still = lat[n_ctx - 1:n_ctx].repeat(n_ctx, 1, 1, 1).to(device)
        seeds.append({"id": f"real_{i}", "label": f"screen {i}",
                      "kind": "real", "ctx": still, "act": [0] * n_ctx})
    cuda = device.type == "cuda"
    for name, hwc in stress_patterns().items():
        x = torch.from_numpy(hwc).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        mean, _ = model_f4.encode(x.to(device))            # fp32, one-time
        ctx = mean[0].unsqueeze(0).repeat(n_ctx, 1, 1, 1)
        seeds.append({"id": f"stress_{name}", "label": name, "kind": "stress",
                      "ctx": ctx, "act": [0] * n_ctx})
    for s in seeds:
        fr = decode_latents(model_f4, s["ctx"][-1:].to(device), device)[0]
        previews[s["id"]] = png_b64(fr)
    return seeds, previews


# --------------------------------------------------------------------------- web
async def gen_loop(app):
    dream: Dream = app["dream"]
    loop = asyncio.get_event_loop()
    ema_dt = None
    while True:
        if not app["clients"]:
            await asyncio.sleep(0.08); continue
        t0 = time.time()
        frame = await loop.run_in_executor(app["executor"], dream.step)
        dt = time.time() - t0
        app["rec"].append(frame)              # rolling record buffer for clip export
        ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
        msg = json.dumps({"t": "frame", "img": png_b64(frame),
                          "fps": round(1.0 / max(ema_dt, 1e-6), 1),
                          "solver": dream.current_solver, "label": dream.label,
                          "idx": dream.frame_idx})
        for q in list(app["clients"]):
            if q.full():
                try: q.get_nowait()
                except Exception: pass
            q.put_nowait(msg)


async def ws_handler(request):
    app = request.app
    dream: Dream = app["dream"]
    ws = web.WebSocketResponse(); await ws.prepare(request)
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    app["clients"].add(q)
    await ws.send_str(json.dumps({"t": "hello", "solver": dream.current_solver, "label": dream.label, "project": dream.project}))

    async def sender():
        while not ws.closed:
            await ws.send_str(await q.get())
    send_task = asyncio.ensure_future(sender())
    try:
        async for m in ws:
            if m.type != WSMsgType.TEXT: continue
            d = json.loads(m.data); t = d.get("t")
            if t == "act":
                a = int(d.get("a", 0))
                if a in ACTIONS: dream.current_action = a
            elif t == "solver":
                s = d.get("s")
                if s in SOLVER_MAP: dream.current_solver = s
            elif t == "project":
                dream.project = bool(d.get("on", False))
            elif t == "reset":
                dream.reset(d.get("seed"))
                await ws.send_str(json.dumps({"t": "hello", "solver": dream.current_solver, "label": dream.label, "project": dream.project}))
    finally:
        send_task.cancel(); app["clients"].discard(q)
    return ws


async def index(request):
    return web.Response(text=PAGE, content_type="text/html")


async def seeds_json(request):
    sds = request.app["dream"].seeds
    prev = request.app["previews"]
    return web.json_response([{"id": s["id"], "label": s["label"], "kind": s["kind"],
                              "preview": prev[s["id"]]} for s in sds])


async def save_clip(request):
    """Render the last N generated frames to an MP4 and return it for download.
    The playback fps is fixed (default 10) so '300 frames' == a 30s clip
    regardless of how fast the model was generating."""
    app = request.app
    n = int(request.query.get("n", 300))
    fps = int(request.query.get("fps", 10))
    scale = int(request.query.get("scale", 4))
    frames = list(app["rec"])[-n:]
    if not frames:
        return web.json_response({"error": "nothing recorded yet"}, status=400)
    up = [np.repeat(np.repeat(f, scale, 0), scale, 1) for f in frames]
    out = ROOT / "out" / "hyrule_play_clip.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, up, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return web.FileResponse(out, headers={
        "Content-Disposition": 'attachment; filename="hyrule_dream_clip.mp4"'})


async def on_start(app):
    app["executor"] = None
    app["gen"] = asyncio.ensure_future(gen_loop(app))


async def on_stop(app):
    app["gen"].cancel()


def build_app(dream, previews):
    app = web.Application()
    app["dream"] = dream; app["clients"] = set(); app["previews"] = previews
    app["rec"] = deque(maxlen=1200)        # ~120s of generated frames at 10fps
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/seeds", seeds_json)
    app.router.add_get("/save", save_clip)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    return app


PAGE = r"""<!DOCTYPE html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>HYRULE DREAMER</title><style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%;width:100%;background:#05060a;color:#9ef;font-family:'Courier New',monospace;overflow:hidden;touch-action:none}
#wrap{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:8px}
h1{font-size:18px;letter-spacing:3px;margin:2px;color:#ffd34d;text-shadow:0 0 8px #ff8c00,0 0 2px #fff}
#stage{position:relative;background:#000;border:3px solid #2af;border-radius:8px;box-shadow:0 0 18px #08f,inset 0 0 18px #024;overflow:hidden;width:min(94vw,58vh);aspect-ratio:1}
#cv{width:100%;height:100%;image-rendering:pixelated;image-rendering:crisp-edges;display:block}
#scan{position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(transparent 0 2px,rgba(0,0,0,.22) 2px 3px);mix-blend-mode:multiply}
#controls{display:flex;gap:30px;align-items:center;justify-content:center;width:100%}
#dpad{display:grid;grid-template-columns:repeat(3,56px);grid-template-rows:repeat(3,56px);gap:5px}
.d{border:1px solid #357;border-radius:9px;background:#0c1830;display:flex;align-items:center;justify-content:center;font-size:24px;color:#8cf}
.d.press{background:#28e;color:#fff;box-shadow:0 0 12px #28e}
#side{display:flex;flex-direction:column;gap:7px;align-items:stretch;min-width:122px}
.btn{padding:9px 12px;border:1px solid #46c;border-radius:7px;background:#0b1422;color:#9ef;font-weight:bold;font-size:13px;cursor:pointer;text-align:center}
.btn.on{background:#1d6;color:#021;border-color:#1d6;box-shadow:0 0 10px #1d6}
#fps{font-size:12px;color:#7fffa0;text-align:center}#scr{font-size:11px;color:#678;text-align:center}
@media(orientation:landscape){h1{position:fixed;top:3px;left:50%;transform:translateX(-50%);font-size:13px;margin:0;z-index:5}
  #stage{width:auto;height:min(90vh,62vw)}#controls{display:contents}
  #dpad{position:fixed;left:16px;bottom:16px;grid-template-columns:repeat(3,60px);grid-template-rows:repeat(3,60px)}
  #side{position:fixed;right:16px;bottom:16px;min-width:128px}}
#menu{position:fixed;inset:0;background:rgba(2,4,8,.93);z-index:20;display:none;flex-direction:column;padding:14px;overflow:auto}
#menu.show{display:flex}
#menu h2{color:#ffd34d;text-align:center;margin:6px;font-size:16px;letter-spacing:2px}
.grp{color:#6af;font-size:12px;margin:8px 4px 2px;border-bottom:1px solid #234;grid-column:1/-1}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:10px}
.seed{border:1px solid #245;border-radius:8px;background:#0a1220;padding:5px;text-align:center;cursor:pointer}
.seed.stress{border-color:#a36}
.seed img{width:100%;aspect-ratio:1;image-rendering:pixelated;border-radius:4px;display:block}
.seed span{font-size:10px;color:#9cf;display:block;margin-top:3px}
#close{align-self:center;margin-top:10px}
</style></head><body><div id=wrap>
<h1>⚔ HYRULE DREAMER ⚔</h1>
<div id=stage><canvas id=cv width=64 height=64></canvas><div id=scan></div></div>
<div id=controls>
  <div id=dpad><div></div><div class=d data-a=5>▲</div><div></div>
    <div class=d data-a=7>◀</div><div class=d data-a=0>•</div><div class=d data-a=8>▶</div>
    <div></div><div class=d data-a=6>▼</div><div></div></div>
  <div id=side><div id=fps>-- fps</div>
    <button class="btn" id=heun>HEUN · 15</button>
    <button class="btn on" id=dpm8>DPM8 · 8</button>
    <button class="btn" id=proj>✨ STABILIZE</button>
    <button class="btn" id=reset>↺ NEW DREAM</button>
    <button class="btn" id=seedbtn>▦ SEEDS</button>
    <button class="btn" id=save>⏺ SAVE 30s</button>
    <div id=scr>--</div></div>
</div></div>
<div id=menu><h2>CHOOSE A DREAM SEED</h2><div id=grid></div>
  <button class="btn" id=close>✕ CLOSE</button></div>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),cx=cv.getContext('2d');
let ws,cur=0,solver='dpm8',seedsLoaded=false;
function setSolver(s){solver=s;$('heun').classList.toggle('on',s=='heun');$('dpm8').classList.toggle('on',s=='dpm8');send({t:'solver',s});}
function send(o){if(ws&&ws.readyState==1)ws.send(JSON.stringify(o));}
function setAct(a){if(a===cur)return;cur=a;send({t:'act',a});}
$('heun').onclick=()=>setSolver('heun');
$('dpm8').onclick=()=>setSolver('dpm8');
$('reset').onclick=()=>{send({t:'reset'});};
let projOn=false;
$('proj').onclick=()=>{projOn=!projOn;$('proj').classList.toggle('on',projOn);send({t:'project',on:projOn});};
$('seedbtn').onclick=()=>{loadSeeds();$('menu').classList.add('show');};
$('save').onclick=()=>{const b=$('save');const o=b.textContent;b.textContent='⏳ saving…';
  fetch('/save?n=300&fps=10').then(r=>r.blob()).then(bl=>{const u=URL.createObjectURL(bl);
  const a=document.createElement('a');a.href=u;a.download='hyrule_dream_clip.mp4';a.click();
  URL.revokeObjectURL(u);b.textContent=o;}).catch(()=>{b.textContent=o;});};
$('close').onclick=()=>$('menu').classList.remove('show');
function draw(b64){const im=new Image();im.onload=()=>{cx.imageSmoothingEnabled=false;cx.drawImage(im,0,0,64,64);};im.src='data:image/png;base64,'+b64;}
async function loadSeeds(){if(seedsLoaded)return;seedsLoaded=true;
  const sds=await (await fetch('/seeds')).json();
  const real=sds.filter(s=>s.kind=='real'),stress=sds.filter(s=>s.kind=='stress');
  const cell=s=>`<div class="seed ${s.kind}" data-id="${s.id}"><img src="data:image/png;base64,${s.preview}"><span>${s.label}</span></div>`;
  $('grid').innerHTML='<div class=grp>REAL SCREENS</div>'+real.map(cell).join('')+'<div class=grp>STRESS / OOD SEEDS (diffuses to gibberish — keeps dreaming)</div>'+stress.map(cell).join('');
  document.querySelectorAll('.seed').forEach(el=>el.onclick=()=>{send({t:'reset',seed:el.dataset.id});$('menu').classList.remove('show');});}
const pressed=[];
function refresh(){setAct(pressed.length?+pressed[pressed.length-1].dataset.a:0);}
function bind(el){const on=e=>{e.preventDefault();if(!pressed.includes(el)){el.classList.add('press');pressed.push(el);}refresh();};
  const off=e=>{e.preventDefault();el.classList.remove('press');const i=pressed.indexOf(el);if(i>=0)pressed.splice(i,1);refresh();};
  el.addEventListener('touchstart',on,{passive:false});el.addEventListener('touchend',off,{passive:false});el.addEventListener('touchcancel',off,{passive:false});
  el.addEventListener('mousedown',on);el.addEventListener('mouseup',off);el.addEventListener('mouseleave',e=>{if(pressed.includes(el))off(e);});}
document.querySelectorAll('.d[data-a]').forEach(bind);
const KMAP={ArrowUp:5,KeyW:5,ArrowDown:6,KeyS:6,ArrowLeft:7,KeyA:7,ArrowRight:8,KeyD:8};const kd=[];
addEventListener('keydown',e=>{const a=KMAP[e.code];if(a!==undefined){e.preventDefault();if(!kd.includes(a))kd.push(a);setAct(kd[kd.length-1]);}});
addEventListener('keyup',e=>{const a=KMAP[e.code];if(a!==undefined){const i=kd.indexOf(a);if(i>=0)kd.splice(i,1);setAct(kd.length?kd[kd.length-1]:0);}});
function connect(){ws=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.onmessage=ev=>{const d=JSON.parse(ev.data);
    if(d.t=='frame'){draw(d.img);$('fps').textContent=d.fps+' fps';$('scr').textContent=d.label;}
    if(d.t=='hello'){setSolver(d.solver);$('scr').textContent=d.label;
      if(d.project!==undefined){projOn=d.project;$('proj').classList.toggle('on',projOn);}}};
  ws.onclose=()=>setTimeout(connect,800);}
connect();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True, type=Path, help="world-model checkpoint")
    ap.add_argument("--tokenizer", required=True, type=Path, help="tokenizer checkpoint")
    ap.add_argument("--seeds-dir", type=Path, default=ROOT / "assets" / "seeds",
                    help="directory of .pt seed clips")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--sigma-data", type=float, default=0.179)
    ap.add_argument("--sigma-ctx", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_tokenizer(args.tokenizer, device)
    edm = dfn.EDMConfig(sigma_data=args.sigma_data, sigma_ctx=args.sigma_ctx)
    model, cfg = load_wm(args.wm, device, weights=args.weights)
    seeds, previews = build_seeds(decoder, args.seeds_dir, args.n_seeds, device)
    dream = Dream(model, decoder, cfg, edm, device, seeds)
    print(f"[play] loaded; {len(seeds)} seeds ({sum(s['kind']=='real' for s in seeds)} real + "
          f"{sum(s['kind']=='stress' for s in seeds)} stress); device={device}", flush=True)

    if args.smoke:
        for sid in (seeds[0]["id"], "stress_checkerboard"):
            dream.reset(sid)
            for s in ("heun", "dpm8"):
                dream.current_solver = s; dream.current_action = 8
                t0 = time.time(); f = dream.step(); dt = time.time() - t0
                print(f"[smoke] seed={sid} {s}: {f.shape} {dt*1000:.0f}ms", flush=True)
        print("[smoke] OK", flush=True); return

    app = build_app(dream, previews)
    print(f"[play] serving http://{args.host}:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
