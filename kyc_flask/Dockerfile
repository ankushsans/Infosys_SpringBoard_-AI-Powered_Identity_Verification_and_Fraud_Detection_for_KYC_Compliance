# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for Tesseract + OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="KYC Compliance AI"
LABEL description="Flask API for KYC document classification & GNN anomaly detection"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py           ./
COPY wsgi.py          ./
COPY export_models.py ./
COPY utils/           ./utils/
COPY templates/       ./templates/

# Model weights are mounted at runtime — do NOT bake into the image
# (they can be large and should be updated independently of the code)
RUN mkdir -p models uploads

# Non-root user for security
RUN groupadd -r kyc && useradd -r -g kyc kyc \
    && chown -R kyc:kyc /app
USER kyc

EXPOSE 5000

# Gunicorn with 4 workers; adjust --workers based on available vCPUs
CMD ["gunicorn", \
     "--bind",         "0.0.0.0:5000", \
     "--workers",      "4", \
     "--worker-class", "sync", \
     "--timeout",      "120", \
     "--access-logfile", "-", \
     "--error-logfile",  "-", \
     "wsgi:app"]
