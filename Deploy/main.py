"""
API Deteksi Jenis Sampah (YOLO11 via ONNX Runtime)

Versi ringan untuk hosting RAM 512 MB: tanpa PyTorch dan Ultralytics.
Endpoint: /  /health  /v1/predict  /v1/stats  /docs

Jalankan lokal:
    uvicorn main:app --reload
"""
import ast
import ctypes
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

# ----------------------------- Konfigurasi -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.onnx"))
# Model cepat (opsional): hasil export dengan imgsz lebih kecil, dipakai kamera langsung.
FAST_MODEL_PATH = os.getenv("FAST_MODEL_PATH", str(BASE_DIR / "models" / "best_416.onnx"))
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.347"))  # ambang terbaik dari kurva F1
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
MAX_DET = 300
MAX_UPLOAD_MB = 10
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))                 # sisi terpanjang gambar yang diproses (px)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(40_000_000)))    # tolak gambar dengan resolusi ekstrem
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))        # jumlah gambar diproses bersamaan
ORT_THREADS = int(os.getenv("ORT_THREADS", "1"))              # naikkan ke 2 jika paket hosting punya 2 vCPU
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
VERSION = "1.3.0"

# Dipakai hanya jika file ONNX tidak menyimpan nama kelas.
# Urutan HARUS sama dengan metadata model (cek /health atau data.yaml hasil training).
FALLBACK_NAMES = ["B3", "anorganik", "organik", "residu"]

# Saran penanganan per jenis sampah
TIPS = {
    "organik": "Sampah organik: dapat dikomposkan. Buang ke tempat sampah organik (hijau).",
    "anorganik": "Sampah anorganik: dapat didaur ulang. Bersihkan lalu buang ke tempat sampah anorganik (kuning).",
    "residu": "Sampah residu: tidak dapat didaur ulang atau dikompos. Buang ke tempat sampah residu.",
    "B3": "Sampah B3 (bahan berbahaya dan beracun): jangan dicampur sampah biasa. Serahkan ke tempat pengumpulan khusus.",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

# ----------------------------- Hemat memori -----------------------------
# Hanya MAX_CONCURRENT gambar yang diproses sekaligus, sisanya mengantre.
_gate = threading.BoundedSemaphore(MAX_CONCURRENT)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def trim_memory():
    """Minta glibc mengembalikan memori bekas ke sistem."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


# ----------------------------- Statistik (monitoring) -----------------------------
def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))], 1)


class Stats:
    """Statistik di memori: hilang saat restart, dan hanya akurat dengan satu worker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.detections = 0
        self.inference = deque(maxlen=500)  # inference_ms 500 request terakhir

    def count_request(self, status_code: int):
        with self.lock:
            self.requests += 1
            if status_code >= 400:
                self.errors += 1

    def add_inference(self, ms: float, n_det: int):
        with self.lock:
            self.inference.append(ms)
            self.detections += n_det

    def snapshot(self):
        with self.lock:
            inf = list(self.inference)
            return {
                "uptime_s": round(time.time() - self.started),
                "requests": self.requests,
                "errors": self.errors,
                "error_rate": round(self.errors / self.requests, 4) if self.requests else 0.0,
                "detections_total": self.detections,
                "memory_mb": rss_mb(),
                "inference_ms": {
                    "samples": len(inf),
                    "avg": round(sum(inf) / len(inf), 1) if inf else None,
                    "p50": percentile(inf, 50),
                    "p95": percentile(inf, 95),
                    "p99": percentile(inf, 99),
                },
            }


stats = Stats()


# ----------------------------- Model -----------------------------
def load_names(sess: ort.InferenceSession) -> dict:
    """Ultralytics menyimpan nama kelas di metadata ONNX, misalnya "{0: 'B3', ...}"."""
    raw = sess.get_modelmeta().custom_metadata_map.get("names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return {int(k): str(v) for k, v in parsed.items()}
        except (ValueError, SyntaxError):
            pass
    return {i: n for i, n in enumerate(FALLBACK_NAMES)}


class Engine:
    """Satu model ONNX beserta ukuran input dan nama kelasnya."""

    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_THREADS
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False   # cegah memori membengkak setelah banyak request
        opts.enable_mem_pattern = False
        self.path = path
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.h = h if isinstance(h, int) else 640
        self.w = w if isinstance(w, int) else 640
        self.fp16 = "float16" in inp.type
        self.names = load_names(self.session)
        # warm-up supaya request pertama tidak lambat
        self.session.run(None, {self.input_name: np.zeros((1, 3, self.h, self.w), dtype=self.dtype)})

    @property
    def dtype(self):
        return np.float16 if self.fp16 else np.float32


engine: Optional[Engine] = None
fast_engine: Optional[Engine] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Model dimuat sekali saat server menyala, bukan di setiap request."""
    global engine, fast_engine
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"File model tidak ditemukan: {MODEL_PATH}")
    engine = Engine(MODEL_PATH)
    if os.path.exists(FAST_MODEL_PATH) and os.path.abspath(FAST_MODEL_PATH) != os.path.abspath(MODEL_PATH):
        fast_engine = Engine(FAST_MODEL_PATH)
    log.info(json.dumps({
        "event": "startup", "version": VERSION, "model": f"{engine.w}x{engine.h}",
        "fast_model": f"{fast_engine.w}x{fast_engine.h}" if fast_engine else None, "memory_mb": rss_mb(),
    }))
    yield


# ----------------------------- Aplikasi -----------------------------
app = FastAPI(
    title="API Deteksi Jenis Sampah",
    description="Deteksi sampah organik, anorganik, residu, dan B3 menggunakan YOLO11 (ONNX Runtime).",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Beri setiap request sebuah ID, ukur durasinya, dan catat satu baris log JSON."""
    rid = re.sub(r"[^A-Za-z0-9_.-]", "", request.headers.get("x-request-id", ""))[:64] or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(json.dumps({"event": "unhandled_error", "request_id": rid, "path": request.url.path}))
        response = JSONResponse({"detail": "Terjadi kesalahan pada server."}, status_code=500)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    if request.url.path == "/v1/predict":
        stats.count_request(response.status_code)
    log.info(json.dumps({
        "event": "request", "request_id": rid, "method": request.method,
        "path": request.url.path, "status": response.status_code, "duration_ms": duration_ms,
    }))
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "model_not_loaded",
        "version": VERSION,
        "model": os.path.basename(MODEL_PATH),
        "model_input": f"{engine.w}x{engine.h}" if engine else None,
        "fast_model": os.path.basename(FAST_MODEL_PATH) if fast_engine else None,
        "classes": list(engine.names.values()) if engine else [],
        "default_conf": DEFAULT_CONF,
        "memory_mb": rss_mb(),
    }


@app.get("/v1/stats")
def get_stats():
    """Ringkasan monitoring: jumlah request, error rate, memori, dan latensi inferensi (p50/p95/p99)."""
    return stats.snapshot()


# ----------------------------- Praproses & pascaproses -----------------------------
def load_image(data: bytes) -> Image.Image:
    """Buka gambar dengan hemat memori: JPEG didekode langsung dalam ukuran kecil."""
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if w * h > MAX_PIXELS:
        raise ValueError(f"Resolusi gambar terlalu besar (maks {MAX_PIXELS // 1_000_000} megapiksel).")
    if img.format == "JPEG":
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    return img


def preprocess(image: Image.Image, eng: Engine):
    """Letterbox ke ukuran input model (padding abu-abu 114), RGB, 0-1, NCHW."""
    w0, h0 = image.size
    r = min(eng.h / h0, eng.w / w0)
    nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (eng.w, eng.h), (114, 114, 114))
    px, py = (eng.w - nw) // 2, (eng.h - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
    return arr.astype(eng.dtype), r, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    """NMS lintas kelas (setara agnostic_nms=True)."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def postprocess(out: np.ndarray, conf_thr: float, r: float, px: int, py: int, w0: int, h0: int):
    pred = np.asarray(out[0], dtype=np.float32)  # (4+nc, N) untuk YOLO11
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                           # -> (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all.max(axis=1)
    m = conf >= conf_thr
    if not m.any():
        return []
    pred, cls, conf = pred[m], cls[m], conf[m]
    if len(conf) > MAX_DET * 4:                 # batasi kandidat sebelum NMS
        top = conf.argsort()[::-1][: MAX_DET * 4]
        pred, cls, conf = pred[top], cls[top], conf[top]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r   # kembalikan ke koordinat gambar asli
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    keep = nms(boxes, conf, IOU_THRESHOLD)[:MAX_DET]
    return [(int(cls[i]), float(conf[i]), boxes[i]) for i in keep]


# ----------------------------- Endpoint prediksi -----------------------------
@app.post("/v1/predict")
def predict(
    request: Request,
    file: UploadFile = File(..., description="Gambar sampah (JPG/PNG/WEBP, maks 10 MB)"),
    conf: float = Query(DEFAULT_CONF, ge=0.05, le=0.95, description="Ambang confidence"),
    fast: bool = Query(False, description="Pakai model cepat (resolusi input lebih kecil) jika tersedia"),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Format tidak didukung. Gunakan JPG, PNG, atau WEBP.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)  # baca maksimal limit+1 byte agar file raksasa tidak memenuhi RAM
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi {MAX_UPLOAD_MB} MB.")

    eng = fast_engine if (fast and fast_engine is not None) else engine

    with _gate:
        try:
            image = load_image(data)
        except ValueError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")
        finally:
            del data

        start = time.perf_counter()
        tensor, r, px, py = preprocess(image, eng)
        out = eng.session.run(None, {eng.input_name: tensor})[0]
        found = postprocess(out, conf, r, px, py, image.width, image.height)
        inference_ms = (time.perf_counter() - start) * 1000
        img_w, img_h = image.width, image.height
        del tensor, out, image
        trim_memory()

    detections = []
    for c, score, b in found:
        label = eng.names.get(c, str(c))
        x1, y1, x2, y2 = [round(float(v), 1) for v in b]
        detections.append(
            {
                "label": label,
                "confidence": round(score, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "tip": TIPS.get(label, ""),
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    stats.add_inference(inference_ms, len(detections))

    return {
        "request_id": request.state.request_id,
        "count": len(detections),
        "detections": detections,
        "image": {"width": img_w, "height": img_h},
        "conf_threshold": conf,
        "model_input": f"{eng.w}x{eng.h}",
        "inference_ms": round(inference_ms, 1),
        "message": "OK" if detections else "Tidak ada sampah terdeteksi pada ambang confidence ini.",
    }