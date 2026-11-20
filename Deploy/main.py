"""
API Deteksi Jenis Sampah (YOLO11 via ONNX Runtime)

Versi ringan untuk hosting gratis (RAM 512 MB): tanpa PyTorch dan Ultralytics.
Endpoint dan format respons sama dengan versi sebelumnya, jadi index.html tidak perlu diubah.

Jalankan lokal:
    uvicorn main:app --reload
Dokumentasi interaktif:
    http://127.0.0.1:8000/docs
"""
import ast
import ctypes
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

# ----------------------------- Konfigurasi -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.onnx"))
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.347"))  # ambang terbaik dari kurva F1
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
MAX_DET = 300
MAX_UPLOAD_MB = 10
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))                 # sisi terpanjang gambar yang diproses (px)
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(40_000_000)))    # tolak gambar dengan resolusi ekstrem
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))        # jumlah gambar diproses bersamaan
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Dipakai hanya jika file ONNX tidak menyimpan nama kelas. Urutan HARUS sama dengan data.yaml.
FALLBACK_NAMES = ["organik", "anorganik", "residu", "B3"]

# Saran penanganan per jenis sampah (nama kelas sesuai data.yaml)
TIPS = {
    "organik": "Sampah organik: dapat dikomposkan. Buang ke tempat sampah organik (hijau).",
    "anorganik": "Sampah anorganik: dapat didaur ulang. Bersihkan lalu buang ke tempat sampah anorganik (kuning).",
    "residu": "Sampah residu: tidak dapat didaur ulang atau dikompos. Buang ke tempat sampah residu.",
    "B3": "Sampah B3 (bahan berbahaya dan beracun): jangan dicampur sampah biasa. Serahkan ke tempat pengumpulan khusus.",
}

# ----------------------------- Hemat memori -----------------------------
# Hanya MAX_CONCURRENT gambar yang diproses sekaligus, sisanya mengantre. Tanpa ini, beberapa
# request bersamaan (mis. kamera + upload) menumpuk memori dan bisa mematikan kontainer 512 MB.
_gate = threading.BoundedSemaphore(MAX_CONCURRENT)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def trim_memory():
    """Minta glibc mengembalikan memori bekas ke sistem (Python jarang melakukannya sendiri)."""
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


# ----------------------------- Model -----------------------------
session: Optional[ort.InferenceSession] = None
input_name = ""
in_h, in_w = 640, 640
in_fp16 = False
names: dict = {}


def load_names(sess: ort.InferenceSession) -> dict:
    """Ultralytics menyimpan nama kelas di metadata ONNX, misalnya "{0: 'organik', ...}"."""
    raw = sess.get_modelmeta().custom_metadata_map.get("names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return {int(k): str(v) for k, v in parsed.items()}
        except (ValueError, SyntaxError):
            pass
    return {i: n for i, n in enumerate(FALLBACK_NAMES)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Model dimuat sekali saat server menyala, bukan di setiap request."""
    global session, input_name, in_h, in_w, in_fp16, names
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"File model tidak ditemukan: {MODEL_PATH}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))  # hemat memori/CPU di paket gratis
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False   # cegah memori membengkak setelah banyak request
    opts.enable_mem_pattern = False
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    input_name = inp.name
    h, w = inp.shape[2], inp.shape[3]
    in_h = h if isinstance(h, int) else 640
    in_w = w if isinstance(w, int) else 640
    in_fp16 = "float16" in inp.type
    names = load_names(session)
    # warm-up supaya request pertama tidak lambat
    session.run(None, {input_name: np.zeros((1, 3, in_h, in_w), dtype=np.float16 if in_fp16 else np.float32)})
    yield


# ----------------------------- Aplikasi -----------------------------
app = FastAPI(
    title="API Deteksi Jenis Sampah",
    description="Deteksi sampah organik, anorganik, residu, dan B3 menggunakan YOLO11 (ONNX Runtime).",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk produksi, ganti dengan domain frontend-mu
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok" if session is not None else "model_not_loaded",
        "model": os.path.basename(MODEL_PATH),
        "classes": list(names.values()),
        "default_conf": DEFAULT_CONF,
        "memory_mb": rss_mb(),
    }


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


def preprocess(image: Image.Image):
    """Letterbox ke ukuran input model (padding abu-abu 114), RGB, 0-1, NCHW."""
    w0, h0 = image.size
    r = min(in_h / h0, in_w / w0)
    nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (in_w, in_h), (114, 114, 114))
    px, py = (in_w - nw) // 2, (in_h - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
    return (arr.astype(np.float16) if in_fp16 else arr), r, px, py


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
    file: UploadFile = File(..., description="Gambar sampah (JPG/PNG/WEBP, maks 10 MB)"),
    conf: float = Query(DEFAULT_CONF, ge=0.05, le=0.95, description="Ambang confidence"),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Format tidak didukung. Gunakan JPG, PNG, atau WEBP.")

    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi {MAX_UPLOAD_MB} MB.")

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
        tensor, r, px, py = preprocess(image)
        out = session.run(None, {input_name: tensor})[0]
        found = postprocess(out, conf, r, px, py, image.width, image.height)
        inference_ms = (time.perf_counter() - start) * 1000
        img_w, img_h = image.width, image.height
        del tensor, out, image
        trim_memory()

    detections = []
    for c, score, b in found:
        label = names.get(c, str(c))
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

    return {
        "count": len(detections),
        "detections": detections,
        "image": {"width": img_w, "height": img_h},
        "conf_threshold": conf,
        "inference_ms": round(inference_ms, 1),
        "message": "OK" if detections else "Tidak ada sampah terdeteksi pada ambang confidence ini.",
    }
