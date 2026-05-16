FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/runpod-volume/ollama

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

WORKDIR /app

COPY requirements-runpod.txt .
RUN pip3 install --break-system-packages -r requirements-runpod.txt

COPY mejorar_subtitulos.py .
COPY runpod_handler.py .

# === PRE-CARGA DEL MODELO QWEN3.5 ===
RUN ollama serve & \
    sleep 12 && \
    ollama pull qwen3.5:32b && \
    pkill ollama || true

CMD ["python3", "-u", "runpod_handler.py"]