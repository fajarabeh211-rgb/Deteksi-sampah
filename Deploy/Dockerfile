FROM python:3.12-slim-bookworm

# Library sistem yang dibutuhkan OpenCV (dipakai Ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces menjalankan container sebagai user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    PORT=7860
WORKDIR $HOME/app

# PyTorch versi CPU (jauh lebih kecil daripada versi CUDA), lalu dependensi lain
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Kode, frontend, dan bobot model
COPY --chown=user main.py index.html ./
COPY --chown=user models ./models

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','7860'))"

# Satu worker: model dimuat sekali dan statistik /v1/stats tetap konsisten
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-1024} --workers 1
