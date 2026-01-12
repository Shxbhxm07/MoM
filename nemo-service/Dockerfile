FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Clone NeMo
RUN git clone https://github.com/NVIDIA/NeMo.git /app/NeMo

WORKDIR /app/NeMo

# Run NeMo installation script
RUN bash docker/common/install_dep.sh --library all || true

# Install NeMo from source
RUN pip install --no-cache-dir ".[all]" --break-system-packages || \
    pip install --no-cache-dir ".[asr]" --break-system-packages

# Copy service file
COPY service.py /app/service.py

WORKDIR /app

EXPOSE 8003

CMD ["python", "service.py"]
