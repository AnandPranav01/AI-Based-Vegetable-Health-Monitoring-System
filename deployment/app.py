"""
Food Spoilage Detection API

FastAPI application providing REST endpoints for the two-stage food
spoilage detection system (YOLOv8 detection + ResNet50 classification).

Endpoints:
    GET  /health      — Liveness / readiness probe
    POST /predict     — Full pipeline (YOLO + ResNet) on an uploaded image
    POST /classify    — ResNet-only classification for a pre-cropped image

Local usage:
    uvicorn deployment.app:app --host 0.0.0.0 --port 8000

Docker usage (from project root):
    docker build -t food-spoilage-api .
    docker run -p 8000:8000 food-spoilage-api
"""

import io
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
# Works both locally (deployment/app.py → parent = project root)
# and in Docker (COPY . /app/ → /app/deployment/app.py → parent = /app)
_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.inference import FoodSpoilagePipeline  # noqa: E402
from src.resnet.predict_resnet import ResNetPredictor      # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("food_spoilage_api")

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.yaml"
_RESNET_PATH = PROJECT_ROOT / "models" / "resnet_spoilage.pt"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
class _AppState:
    pipeline: Optional[FoodSpoilagePipeline] = None
    resnet: Optional[ResNetPredictor] = None


_state = _AppState()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load models at startup; release them on shutdown."""
    logger.info("Loading models (this may take ~30 s on first run) …")
    try:
        _state.pipeline = FoodSpoilagePipeline(
            config_path=str(_CONFIG_PATH) if _CONFIG_PATH.exists() else None
        )
        _state.resnet = ResNetPredictor(
            model_path=str(_RESNET_PATH),
            config_path=str(_CONFIG_PATH) if _CONFIG_PATH.exists() else None,
        )
        logger.info("Models loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load models: {exc}")
        raise

    yield  # application is running

    logger.info("Shutting down — releasing models.")
    _state.pipeline = None
    _state.resnet = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Food Spoilage Detection API",
    description=(
        "Two-stage food spoilage detection: "
        "YOLOv8 locates food items in an image, "
        "ResNet50 classifies each item as **fresh** or **spoiled**."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class DetectionItem(BaseModel):
    detection_id: int
    object_class: str
    bbox: List[float]               # [x1, y1, x2, y2]
    object_confidence: float
    spoilage_status: str            # "fresh" | "spoiled"
    spoilage_confidence: float
    freshness_percentage: float     # 0-100
    combined_confidence: float


class PipelineResponse(BaseModel):
    image_name: str
    processing_time_s: float
    total_detections: int
    fresh_count: int
    spoiled_count: int
    fresh_percentage: float
    spoiled_percentage: float
    detections: List[DetectionItem]


class ClassifyResponse(BaseModel):
    classification: str             # "fresh" | "spoiled"
    confidence: float
    freshness_percentage: float
    probabilities: dict             # {"fresh": float, "spoiled": float}


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


def _read_image(upload: UploadFile) -> Image.Image:
    """Validate content-type & size, then decode to a PIL RGB image."""
    if upload.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{upload.content_type}'. "
                f"Accepted: {sorted(_ALLOWED_TYPES)}"
            ),
        )
    raw = upload.file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(raw) // 1024} KB). Maximum is 20 MB.",
        )
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded file as an image.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health():
    """
    Liveness and readiness probe.

    Returns `pipeline_ready` and `resnet_ready` flags so orchestrators
    (Kubernetes, Docker Compose health-checks) can confirm the models
    have been loaded before routing traffic.
    """
    return {
        "status": "ok",
        "pipeline_ready": _state.pipeline is not None,
        "resnet_ready": _state.resnet is not None,
    }


@app.post("/predict", response_model=PipelineResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Food scene image (JPEG / PNG / WebP / BMP)"),
):
    """
    Run the **full two-stage pipeline** on an uploaded image.

    - **Stage 1** — YOLOv8 detects and localises food items.
    - **Stage 2** — ResNet50 classifies each crop as *fresh* or *spoiled*.

    Returns per-item results together with a summary of fresh/spoiled counts.
    """
    if _state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline model is not initialised. Retry after startup completes.",
        )

    pil_image = _read_image(file)
    img_array = np.array(pil_image)

    t0 = time.perf_counter()
    try:
        raw = _state.pipeline.process_image(img_array, save_visualization=False)
    except Exception as exc:
        logger.exception("Pipeline inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )
    elapsed = round(time.perf_counter() - t0, 3)

    summary = raw.get("summary", {})
    metadata = raw.get("metadata", {})

    detections = [
        DetectionItem(
            detection_id=d["detection_id"],
            object_class=d["object_class"],
            bbox=d["bbox"],
            object_confidence=round(d["object_confidence"], 4),
            spoilage_status=d["spoilage_status"],
            spoilage_confidence=round(d["spoilage_confidence"], 4),
            freshness_percentage=round(d.get("freshness_percentage", 0.0), 2),
            combined_confidence=round(d["combined_confidence"], 4),
        )
        for d in raw.get("detections", [])
    ]

    return PipelineResponse(
        image_name=metadata.get("image_name", file.filename or "uploaded"),
        processing_time_s=elapsed,
        total_detections=summary.get("total_detections", 0),
        fresh_count=summary.get("fresh_count", 0),
        spoiled_count=summary.get("spoiled_count", 0),
        fresh_percentage=round(summary.get("fresh_percentage", 0.0), 1),
        spoiled_percentage=round(summary.get("spoiled_percentage", 0.0), 1),
        detections=detections,
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify(
    file: UploadFile = File(
        ...,
        description="Pre-cropped single food item image (JPEG / PNG / WebP / BMP)",
    ),
):
    """
    Classify a **single pre-cropped food item** as *fresh* or *spoiled*.

    Use this endpoint when you already have a tight crop around one food item
    and do not need the YOLO detection step.
    """
    if _state.resnet is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ResNet model is not initialised. Retry after startup completes.",
        )

    pil_image = _read_image(file)
    try:
        result = _state.resnet.predict(pil_image, return_confidence=True)
    except Exception as exc:
        logger.exception("ResNet classification error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {exc}",
        )

    return ClassifyResponse(
        classification=result["class"],
        confidence=round(result["confidence"], 4),
        freshness_percentage=round(result.get("freshness_percentage", 0.0), 2),
        probabilities={
            k: round(v, 4) for k, v in result.get("probabilities", {}).items()
        },
    )
