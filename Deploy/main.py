"""
API Deteksi Jenis Sampah (YOLO11) - versi deployment.

Satu server melayani frontend (/) dan API (/health, /v1/predict, /v1/stats).
Konfigurasi lewat environment variable, lihat bagian Konfigurasi.
"""
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

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# ----------------------------- Konfigurasi -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.onnx"))
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.347"))  # puncak kurva F1 hasil training
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
VERSION = "1.1.0"

TIPS = {
    "organik": "Sampah organik: dapat dikomposkan. Buang ke tempat sampah organik (hijau).",
    "anorganik": "Sampah anorganik: dapat didaur ulang. Bersihkan lalu buang ke tempat sampah anorganik (kuning).",
    "residu": "Sampah residu: tidak dapat didaur ulang atau dikompos. Buang ke tempat sampah residu.",
    "B3": "Sampah B3 (bahan berbahaya dan beracun): jangan dicampur sampah biasa. Serahkan ke tempat pengumpulan khusus.",
}

# ----------------------------- Logging & statistik -----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))], 1)


class Stats:
    """Statistik di memori (hilang saat restart; gunakan satu worker)."""

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
                "inference_ms": {
                    "samples": len(inf),
                    "avg": round(sum(inf) / len(inf), 1) if inf else None,
                    "p50": percentile(inf, 50),
                    "p95": percentile(inf, 95),
                    "p99": percentile(inf, 99),
                },
            }


stats = Stats()
infer_lock = threading.Lock()  # satu inferensi pada satu waktu: aman dan tidak berebut CPU
model: Optional[YOLO] = None


# ----------------------------- Aplikasi -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Model dimuat sekali saat server menyala, bukan di setiap request."""
    global model
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"File model tidak ditemukan: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.predict(Image.new("RGB", (IMG_SIZE, IMG_SIZE)), imgsz=IMG_SIZE, iou=0.5, agnostic_nms=True, verbose=False)  # warm-up
    log.info(json.dumps({"event": "startup", "model": os.path.basename(MODEL_PATH), "version": VERSION}))
    yield


app = FastAPI(
    title="API Deteksi Jenis Sampah",
    description="Deteksi sampah organik, anorganik, residu, dan B3 menggunakan YOLO11.",
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
    page = BASE_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "API Deteksi Jenis Sampah aktif. Dokumentasi: /docs"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "version": VERSION,
        "model": os.path.basename(MODEL_PATH),
        "classes": list(model.names.values()) if model else [],
        "default_conf": DEFAULT_CONF,
        "uptime_s": stats.snapshot()["uptime_s"],
    }


@app.get("/v1/stats")
def get_stats():
    """Ringkasan monitoring: jumlah request, error rate, dan latensi inferensi (p50/p95/p99)."""
    return stats.snapshot()


@app.post("/v1/predict")
def predict(
    request: Request,
    file: UploadFile = File(..., description="Gambar sampah (JPG/PNG/WEBP, maks 10 MB)"),
    conf: float = Query(DEFAULT_CONF, ge=0.05, le=0.95, description="Ambang confidence"),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Format tidak didukung. Gunakan JPG, PNG, atau WEBP.")

    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi {MAX_UPLOAD_MB} MB.")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    start = time.perf_counter()
    try:
        with infer_lock:
            result = model.predict(image, conf=conf, imgsz=IMG_SIZE, iou=0.5, agnostic_nms=True, verbose=False)[0]
    except Exception:
        log.exception(json.dumps({"event": "inference_error", "request_id": request.state.request_id}))
        raise HTTPException(status_code=500, detail="Gagal memproses gambar.")
    inference_ms = (time.perf_counter() - start) * 1000

    detections = []
    for box in result.boxes:
        label = result.names[int(box.cls)]
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "label": label,
            "confidence": round(float(box.conf), 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "tip": TIPS.get(label, ""),
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    stats.add_inference(inference_ms, len(detections))

    return {
        "request_id": request.state.request_id,
        "count": len(detections),
        "detections": detections,
        "image": {"width": image.width, "height": image.height},
        "conf_threshold": conf,
        "inference_ms": round(inference_ms, 1),
        "message": "OK" if detections else "Tidak ada sampah terdeteksi pada ambang confidence ini.",
    }
