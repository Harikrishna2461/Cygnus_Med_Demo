# Vein Naming System -- web app + BioMedParse segmentation (GPU). The Qwen model server runs in
# its OWN container (see docker-compose.yml, service "qwen"); this image only holds the app.
#
# Build context is ../Code  (docker-compose.yml sets it), not this folder.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) GPU PyTorch (CUDA 12.8 wheels) -- pinned to the versions the project was tested with.
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Everything else. `opencv-python` -> headless (no display server in a container).
COPY requirements.txt .
RUN sed -i 's/^opencv-python>=/opencv-python-headless>=/' requirements.txt \
    && pip install -r requirements.txt

# 3) App code + vendored BiomedParse source. Model WEIGHTS are mounted at runtime (see compose).
COPY backend ./backend
COPY frontend ./frontend
COPY BiomedParse_Assets ./BiomedParse_Assets
COPY sample_data ./sample_data

ENV CMED_HOST=0.0.0.0 \
    CMED_PORT=7862 \
    LLAMA_SERVER_URL=http://qwen:8081 \
    HF_HOME=/hf_cache

WORKDIR /app/backend
EXPOSE 7862

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://127.0.0.1:7862/api/health || exit 1

CMD ["python3", "app.py"]
