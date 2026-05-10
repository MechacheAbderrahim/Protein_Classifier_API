# Python 3.11 image used to run the FastAPI inference service.
# Build this image with: docker buildx build --platform linux/amd64 --load -t protein-api:latest .
# We use linux/amd64 because EDTSurf_linux is an x86-64 Linux binary.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and make logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application directory inside the container.
WORKDIR /app

# System libraries commonly needed by Open3D / scientific Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache dependency installation.
# If only the application code changes, this layer is reused.
# If requirements.txt changes, Docker reruns the dependency installation below.
COPY requirements.txt .

# Install Python dependencies for the API and inference pipeline.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project files: API, model code, weights, and Linux EDTSurf binary.
COPY . .

# Use the Linux EDTSurf binary inside Docker.
# Locally, utils/EDTSurf can stay as the macOS binary.
# Inside the Linux container, we replace it with utils/EDTSurf_linux so main.py
# can keep using the same default path: utils/EDTSurf.
RUN cp utils/EDTSurf_linux utils/EDTSurf \
    && chmod +x utils/EDTSurf

# FastAPI will listen on port 8000 inside the container.
EXPOSE 8000

# Start the API server.
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
