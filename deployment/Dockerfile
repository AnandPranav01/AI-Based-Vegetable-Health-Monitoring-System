# ---------------------------------------------------------------------------
# Food Spoilage Detection API — Docker image
# ---------------------------------------------------------------------------
# Build:  docker build -t food-spoilage-api .
# Run:    docker run -p 8000:8000 food-spoilage-api
# ---------------------------------------------------------------------------

FROM python:3.10-slim

# System libraries required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to exploit Docker layer caching
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy project source
COPY configs/   /app/configs/
COPY models/    /app/models/
COPY src/       /app/src/
COPY deployment/app.py /app/deployment/app.py

# Non-root user for least-privilege execution
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

# Note: deployment.app:app matches the path deployment/app.py
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
