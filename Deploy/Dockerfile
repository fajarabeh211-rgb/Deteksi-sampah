FROM python:3.12-slim-bookworm

# Jalankan container sebagai user biasa (bukan root)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=1024
WORKDIR $HOME/app

# Dependensi (ringan: tanpa PyTorch dan Ultralytics)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Kode, frontend, dan model ONNX
COPY --chown=user main.py index.html ./
COPY --chown=user models ./models

EXPOSE 1024

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','1024'))"

# Satu worker: model dimuat sekali
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-1024} --workers 1
