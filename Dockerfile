# NeuroSense AI API — Dockerfile for Hugging Face Spaces & Cloud Deployment
FROM python:3.10-slim

# System dependencies for OpenCV, scientific computing, and medical imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user

# Set working directory and ownership
WORKDIR /app
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install lightweight CPU-only PyTorch first (drastically reduces image size & build time)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY --chown=user:user neurosense/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY --chown=user:user . /app

# Ensure directories for outputs and uploads exist with write permissions
RUN mkdir -p /app/outputs/heatmaps /app/uploads /app/neurosense/checkpoints

# Hugging Face Spaces routes to port 7860 by default
EXPOSE 7860

# Launch FastAPI with uvicorn
CMD ["uvicorn", "neurosense.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
