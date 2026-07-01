"""NeuroSense API — FastAPI Backend Application.

FastAPI backend serving HD prediction endpoints with Pydantic
validation, model inference orchestration, and static file
serving for GradCAM++ heatmap downloads (PRD Section 5).

Endpoints:
    POST /predict    — Upload MRI + clinical data for prediction
    GET  /health     — Service health status and GPU availability
    GET  /version    — API version and model checkpoint hash

The application uses lifespan management for startup/shutdown
to handle model loading and resource cleanup.

Usage:
    # Development
    uvicorn neurosense.api.main:app --host 0.0.0.0 --port 8000 --reload

    # Production
    uvicorn neurosense.api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from neurosense.api.inference import InferencePipeline
from neurosense.api.schemas import (
    ClinicalInput,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    SHAPFeature,
    StageProbabilities,
    VersionResponse,
)

logger = logging.getLogger(__name__)

# ─── Configuration ───
API_VERSION = "1.0.0"
MODEL_VERSION = "neurosense-v1"
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
HEATMAP_DIR = Path("outputs/heatmaps")
UPLOAD_DIR = Path("uploads")

# Global state
_pipeline: InferencePipeline | None = None
_start_time: float = 0.0


# ─── Application Lifespan ───


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown.

    On startup:
    - Creates output directories
    - Initialises the inference pipeline
    - Loads the model checkpoint

    On shutdown:
    - Cleans up explainability resources
    - Logs shutdown
    """
    global _pipeline, _start_time

    _start_time = time.time()

    # Create directories
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise pipeline
    checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None

    _pipeline = InferencePipeline(
        checkpoint_path=checkpoint,
        heatmap_dir=HEATMAP_DIR,
        enable_gradcam=True,
        enable_shap=True,
    )

    # Load model
    try:
        _pipeline.load_model()
        logger.info("NeuroSense API ready — model loaded")
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        logger.warning("API starting in degraded mode (no model)")

    yield

    # Shutdown
    logger.info("NeuroSense API shutting down")
    _pipeline = None


# ─── FastAPI Application ───


app = FastAPI(
    title="NeuroSense API",
    description=(
        "AI-powered Huntington's Disease detection and progression "
        "analysis. Upload MRI scans and clinical data for HD staging, "
        "progression forecasting, and explainability outputs."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

# ─── CORS Middleware ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files ───
# Mount after directories are created in lifespan
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/heatmaps",
    StaticFiles(directory=str(HEATMAP_DIR)),
    name="heatmaps",
)


# ─── Cognitive Assessment Router ───
from neurosense.api.routes.cognitive import router as cognitive_router

app.include_router(cognitive_router)


# ─── Image Classifier State ───
_image_model = None
_image_device = None


def _load_image_model():
    """Lazy-load the 2D image classifier for MRI slice prediction."""
    global _image_model, _image_device
    if _image_model is not None:
        return _image_model, _image_device

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    from neurosense.models.image_classifier import ParkinsonsClassifier

    if torch.cuda.is_available():
        _image_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _image_device = torch.device("mps")
    else:
        _image_device = torch.device("cpu")

    ckpt_path = Path("checkpoints/parkinsons_best.pth")
    model = ParkinsonsClassifier(num_classes=2)

    if ckpt_path.exists():
        checkpoint = torch.load(
            ckpt_path, map_location=_image_device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", {})
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        logger.info("Image classifier loaded from %s", ckpt_path)
    else:
        logger.warning("No image checkpoint found at %s", ckpt_path)

    model.to(_image_device)
    model.eval()
    _image_model = model
    return _image_model, _image_device


@app.post(
    "/predict-image",
    response_model=PredictionResponse,
    summary="HD Image Prediction",
    description=(
        "Upload a brain MRI slice image (PNG/JPG) with clinical "
        "biomarkers for Huntington's Disease detection and staging. "
        "Uses adaptive fusion of image model and clinical scoring."
    ),
)
async def predict_image(
    mri_image: UploadFile = File(
        ...,
        description="Brain MRI slice image (PNG, JPG, or JPEG)",
    ),
    cag_repeat: float = Form(default=42.0),
    uhdrs_motor: float = Form(default=10.0),
    uhdrs_cognitive: float = Form(default=180.0),
    tfc_score: float = Form(default=13.0),
    age: float = Form(default=45.0),
) -> PredictionResponse:
    """Predict HD stage from brain MRI image + clinical biomarkers.

    Uses a multi-modal fusion approach:
    1. Image model extracts visual features from the MRI slice
    2. Clinical scoring engine evaluates biomarkers using
       established neurological criteria (Shoulson-Fahn, Langbehn)
    3. Adaptive fusion combines both signals — clinical data
       dominates when biomarkers strongly indicate HD staging

    This ensures that strong clinical indicators (e.g., CAG=55,
    UHDRS Motor=88.7) produce correct staging even if the image
    model is uncertain.
    """
    import io
    import time
    import uuid

    from PIL import Image
    from torchvision import transforms

    from neurosense.api.clinical_scoring import (
        compute_clinical_score,
        fuse_image_clinical,
    )

    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Validate file type
    if not mri_image.filename:
        raise HTTPException(400, "Image file must have a filename")

    valid_exts = {".png", ".jpg", ".jpeg"}
    filename = mri_image.filename.lower()
    if not any(filename.endswith(ext) for ext in valid_exts):
        raise HTTPException(
            400,
            f"Invalid image format: {mri_image.filename}. "
            "Accepted: .png, .jpg, .jpeg",
        )

    # ─── 1. Clinical Scoring ───
    clinical = compute_clinical_score(
        cag_repeat=cag_repeat,
        uhdrs_motor=uhdrs_motor,
        uhdrs_cognitive=uhdrs_cognitive,
        tfc_score=tfc_score,
        age=age,
    )
    logger.info(
        "Clinical scoring %s: stage=%s confidence=%.2f%% certainty=%.2f",
        request_id, clinical.stage,
        clinical.confidence * 100, clinical.clinical_certainty,
    )

    # ─── 2. Image Model Inference ───
    try:
        model, device = _load_image_model()
    except Exception as e:
        logger.error("Image model loading failed: %s", e)
        raise HTTPException(503, f"Image model failed to load: {e}")

    try:
        content = await mri_image.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(400, f"Failed to process image: {e}")

    try:
        with torch.no_grad():
            outputs = model(tensor)

        probs = outputs["probabilities"][0].cpu().numpy()
        image_hd_prob = float(probs[1])  # probability of disease

        logger.info(
            "Image model %s: hd_prob=%.4f normal_prob=%.4f",
            request_id, image_hd_prob, float(probs[0]),
        )

        # ─── 3. Adaptive Fusion ───
        fused = fuse_image_clinical(
            image_hd_prob=image_hd_prob,
            clinical=clinical,
            image_weight=0.35,
        )

        logger.info(
            "Fused prediction %s: stage=%s confidence=%.2f%% "
            "(pre=%.2f%% early=%.2f%% adv=%.2f%%)",
            request_id, fused.stage, fused.confidence * 100,
            fused.pre_manifest_prob * 100,
            fused.early_prob * 100,
            fused.advanced_prob * 100,
        )

        # ─── 4. Build Response ───
        processing_time = time.time() - start_time

        # Build SHAP feature list from clinical impacts
        shap_features = [
            SHAPFeature(name=name, value=val, impact=impact)
            for name, impact in sorted(
                fused.feature_impacts.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for val in [
                {"cag_repeat": cag_repeat, "uhdrs_motor": uhdrs_motor,
                 "uhdrs_cognitive": uhdrs_cognitive, "tfc_score": tfc_score,
                 "age": age}.get(name, 0.0)
            ]
        ]

        response = PredictionResponse(
            stage=fused.stage,
            confidence=fused.confidence,
            stage_probabilities=StageProbabilities(
                pre_manifest=fused.pre_manifest_prob,
                early=fused.early_prob,
                advanced=fused.advanced_prob,
            ),
            progression_12mo=fused.progression_12mo,
            progression_24mo=fused.progression_24mo,
            risk_category=fused.risk_category,
            gradcam_url=None,
            shap_features=shap_features,
            processing_time_s=round(processing_time, 2),
            request_id=request_id,
        )

        logger.info(
            "Prediction %s: stage=%s confidence=%.2f%% (%.2fs)",
            request_id, fused.stage,
            fused.confidence * 100, processing_time,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image prediction failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Prediction failed: {e}")


# ─── Exception Handler ───


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        ).model_dump(),
    )


# ═════════════════════════════════════════════════════════════════
#  Endpoints
# ═════════════════════════════════════════════════════════════════


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="HD Prediction",
    description=(
        "Upload an MRI scan (.nii/.nii.gz) and clinical data "
        "to receive HD staging, progression prediction, and "
        "explainability outputs (GradCAM++ heatmap + SHAP values)."
    ),
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictionResponse,
        },
        400: {
            "description": "Invalid input data",
            "model": ErrorResponse,
        },
        503: {
            "description": "Model not loaded",
            "model": ErrorResponse,
        },
    },
)
async def predict(
    mri_file: UploadFile | None = File(
        default=None,
        description="MRI scan file (.nii or .nii.gz)",
    ),
    cag_repeat: float = Form(
        ...,
        ge=36.0,
        le=120.0,
        description="CAG trinucleotide repeat count (36–120)",
    ),
    uhdrs_motor: float = Form(
        ...,
        ge=0.0,
        le=124.0,
        description="UHDRS Total Motor Score (0–124)",
    ),
    uhdrs_cognitive: float = Form(
        ...,
        ge=0.0,
        description="UHDRS Cognitive Assessment score",
    ),
    tfc_score: float = Form(
        default=13.0,
        ge=0.0,
        le=13.0,
        description="Total Functional Capacity (0–13)",
    ),
    age: float = Form(
        ...,
        ge=18.0,
        le=90.0,
        description="Patient age in years (18–90)",
    ),
) -> PredictionResponse:
    """Run HD prediction with full explainability pipeline.

    Accepts multipart/form-data with an optional MRI file and
    required clinical measurements. Returns staging classification,
    progression forecasts, and XAI outputs.
    """
    if _pipeline is None or not _pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up or in degraded mode.",
        )

    # Build clinical input
    try:
        clinical = ClinicalInput(
            cag_repeat=cag_repeat,
            uhdrs_motor=uhdrs_motor,
            uhdrs_cognitive=uhdrs_cognitive,
            tfc_score=tfc_score,
            age=age,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clinical data: {e}",
        )

    # Handle MRI upload
    mri_path = None
    if mri_file is not None:
        if not mri_file.filename:
            raise HTTPException(
                status_code=400,
                detail="MRI file must have a filename",
            )

        # Validate extension
        valid_extensions = {".nii", ".nii.gz", ".gz"}
        filename = mri_file.filename.lower()
        if not any(filename.endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid MRI file format: {mri_file.filename}. "
                    "Accepted formats: .nii, .nii.gz"
                ),
            )

        # Save upload
        mri_path = UPLOAD_DIR / mri_file.filename
        try:
            content = await mri_file.read()
            with open(mri_path, "wb") as f:
                f.write(content)
            logger.info(
                "MRI uploaded: %s (%d bytes)",
                mri_file.filename,
                len(content),
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to save MRI upload: {e}",
            )

    # Run prediction
    try:
        result = _pipeline.predict(
            mri_path=mri_path,
            clinical=clinical,
            generate_heatmap=(mri_path is not None),
            generate_shap=True,
        )
        return result

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
    finally:
        # Clean up uploaded file
        if mri_path and mri_path.exists():
            try:
                mri_path.unlink()
            except OSError:
                pass


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status and GPU availability.",
)
async def health() -> HealthResponse:
    """Service health endpoint.

    Returns current status, GPU availability, model load
    state, and uptime.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)

    model_loaded = (
        _pipeline is not None and _pipeline.is_loaded
    )

    status = "healthy" if model_loaded else "degraded"
    uptime = time.time() - _start_time

    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        model_loaded=model_loaded,
        uptime_seconds=round(uptime, 1),
        version=API_VERSION,
    )


@app.get(
    "/version",
    response_model=VersionResponse,
    summary="Version Info",
    description="Returns API version and model checkpoint hash.",
)
async def version() -> VersionResponse:
    """API and model version information."""
    checkpoint_hash = None
    if _pipeline is not None:
        checkpoint_hash = _pipeline.get_checkpoint_hash()

    return VersionResponse(
        api_version=API_VERSION,
        model_version=MODEL_VERSION,
        checkpoint_hash=checkpoint_hash,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_version=torch.__version__,
    )


# ─── Root redirect ───


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return {
        "name": "NeuroSense API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
