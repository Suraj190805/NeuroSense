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
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from bson import ObjectId
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from neurosense.database.connection import (
    close_db,
    connect_db,
    ensure_connected,
    get_database,
    is_connected,
)
from neurosense.database.crud import (
    create_user,
    get_all_predictions,
    get_prediction_by_id,
    get_predictions_by_user,
    get_test_results,
    get_user_by_email,
    get_user_by_id,
    list_users,
    save_prediction,
    save_test_result,
)
from neurosense.database.models import (
    PredictionDocument,
    TestResultDocument,
    UserDocument,
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

    # Connect to MongoDB
    try:
        await connect_db()
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.error("MongoDB connection failed: %s", e)
        logger.warning("API starting without database — data will not be persisted")

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
    await close_db()
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

# ─── AI Chatbot Router ───
from neurosense.api.routes.chatbot import router as chatbot_router

app.include_router(chatbot_router)


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

    from neurosense.models.image_classifier import HDImageClassifier

    if torch.cuda.is_available():
        _image_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _image_device = torch.device("mps")
    else:
        _image_device = torch.device("cpu")

    candidate_paths = [
        Path(__file__).resolve().parent.parent / "checkpoints" / "hd_image_best.pth",
        Path("neurosense/checkpoints/hd_image_best.pth"),
        Path("checkpoints/hd_image_best.pth"),
    ]
    ckpt_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    model = HDImageClassifier(num_classes=2)

    if ckpt_path.exists():
        checkpoint = torch.load(
            ckpt_path, map_location=_image_device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", {})
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        logger.info("Image classifier loaded successfully from %s", ckpt_path)
    else:
        logger.warning("No image checkpoint found at %s", ckpt_path)

    model.to(_image_device)
    model.eval()
    _image_model = model
    return _image_model, _image_device


@app.post(
    "/predict-image",
    response_model=PredictionResponse,
    summary="HD Image & Digital Assessment Prediction",
    description=(
        "Upload a brain MRI slice image (PNG/JPG) with digital assessment "
        "scores (Motor Test, Memory Test, Functional Capacity) and genetic biomarkers "
        "for Huntington's Disease detection and staging. "
        "Uses adaptive fusion of image model and digital assessment scoring."
    ),
)
async def predict_image(
    mri_image: UploadFile = File(
        ...,
        description="Brain MRI slice image (PNG, JPG, or JPEG)",
    ),
    cag_repeat: float | None = Form(
        default=None,
        ge=10.0,
        le=120.0,
        description="CAG repeat count (Optional: Normal: 10–26, HD: 36–120)",
    ),
    motor_score: float | None = Form(default=None),
    memory_score: float | None = Form(default=None),
    functional_score: float | None = Form(default=None),
    age: float = Form(default=45.0),
    uhdrs_motor: float | None = Form(default=None),
    uhdrs_cognitive: float | None = Form(default=None),
    tfc_score: float | None = Form(default=None),
    symptoms: str | None = Form(
        default=None,
        description="Comma-separated symptom IDs (e.g. 'chorea,dystonia,depression'). Optional.",
    ),
    user_id: str = Form(default="anonymous", description="Logged-in user ID"),
) -> PredictionResponse:
    """Predict HD stage from brain MRI image + digital assessment scores.

    Uses a multi-modal fusion approach:
    1. Image model extracts visual features from the MRI slice
    2. Digital assessment scoring engine evaluates motor test (reaction, tapping,
       tracking), memory/cognitive test, and daily functional capacity
    3. Adaptive fusion combines both signals
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

    # ─── 1. Clinical / Digital Assessment Scoring ───
    # Parse comma-separated symptoms list
    symptom_list = (
        [s.strip() for s in symptoms.split(",") if s.strip()]
        if symptoms
        else None
    )

    clinical = compute_clinical_score(
        cag_repeat=cag_repeat,
        motor_score=motor_score,
        memory_score=memory_score,
        functional_score=functional_score,
        age=age,
        uhdrs_motor=uhdrs_motor,
        uhdrs_cognitive=uhdrs_cognitive,
        tfc_score=tfc_score,
        symptoms=symptom_list,
    )
    logger.info(
        "Digital assessment scoring %s: stage=%s confidence=%.2f%% certainty=%.2f",
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

        # ─── Anatomical Validation (Brain MRI Gatekeeper) ───
        from neurosense.data.mri_validator import validate_brain_mri

        mri_check = validate_brain_mri(image)
        if not mri_check.get("is_brain_mri", mri_check.get("is_valid", False)):
            reasons_str = "; ".join(mri_check.get("reasons", ["Non-cranial anatomy detected."]))
            anatomy_name = mri_check.get("anatomy", "Non-Brain Scan")
            logger.warning(
                "Rejected non-brain MRI upload %s: anatomy=%s, reasons=%s",
                request_id, anatomy_name, reasons_str
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Anatomical Validation Failed: The uploaded image was identified as a {anatomy_name}. "
                    f"{reasons_str} NeuroSense is designed exclusively for Cranial (Brain) MRI scans. "
                    "Please upload an axial or volumetric Brain MRI scan."
                ),
            )

        # ─── MRI Intensity & Contrast Harmonization (Kill Shortcut Learning) ───
        # Standardizes brain Z-scores, applies CLAHE, strips background noise,
        # and crops cranial region to prevent scanner vendor bias
        from neurosense.data.harmonizer import harmonize_2d_slice

        harmonized_2d, tensor = harmonize_2d_slice(
            image,
            target_size=224,
            device=device,
        )
    except HTTPException:
        raise
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

        # ─── 4. Build Response & Generate GradCAM++ Heatmap ───
        processing_time = time.time() - start_time

        # Generate 2D GradCAM++ heatmap overlay
        gradcam_url = None
        try:
            from neurosense.explainability.gradcam import GradCAM2D
            from neurosense.explainability.visualise import save_2d_heatmap_overlay

            cam_explainer = GradCAM2D(model)
            cam = cam_explainer.generate(tensor)
            heatmap_file = HEATMAP_DIR / f"{request_id}.png"
            save_2d_heatmap_overlay(
                image=image,
                heatmap=cam,
                output_path=heatmap_file,
                title=f"GradCAM++ Brain Heatmap — {fused.stage.replace('_', ' ').title()}",
            )
            gradcam_url = f"/static/heatmaps/{request_id}.png"
            logger.info("2D GradCAM++ heatmap generated: %s", gradcam_url)
        except Exception as cam_err:
            logger.warning("2D GradCAM++ generation failed: %s", cam_err)
            gradcam_url = None

        # Effective values
        eff_motor = motor_score if motor_score is not None else (100.0 - (uhdrs_motor / 124.0 * 100.0) if uhdrs_motor is not None else 85.0)
        eff_memory = memory_score if memory_score is not None else (uhdrs_cognitive / 2.0 if uhdrs_cognitive is not None else 85.0)
        eff_functional = functional_score if functional_score is not None else ((tfc_score / 13.0 * 100.0) if tfc_score is not None else 100.0)

        # Build SHAP feature list from canonical clinical impacts
        feature_vals = {
            "cag_repeat": float(cag_repeat) if cag_repeat is not None else 0.0,
            "motor_score": float(eff_motor),
            "memory_score": float(eff_memory),
            "age": float(age),
            "symptoms": float(len(symptom_list)) if symptom_list else 0.0,
        }
        shap_features = [
            SHAPFeature(name=name, value=feature_vals.get(name, 0.0), impact=impact)
            for name, impact in sorted(
                fused.feature_impacts.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            if name in feature_vals
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
            gradcam_url=gradcam_url,
            shap_features=shap_features,
            processing_time_s=round(processing_time, 2),
            request_id=request_id,
        )

        logger.info(
            "Prediction %s: stage=%s confidence=%.2f%% (%.2fs)",
            request_id, fused.stage,
            fused.confidence * 100, processing_time,
        )

        # ─── 5. Persist to MongoDB ───
        if not is_connected():
            await ensure_connected()

        if is_connected():
            try:
                # Map stage to human-readable prediction label
                stage_labels = {
                    "pre_manifest": "Pre-manifest HD",
                    "early": "Early Huntington's",
                    "advanced": "Advanced Huntington's",
                }
                # Map risk category to display format
                risk_labels = {
                    "low": "Low",
                    "medium": "Medium",
                    "high": "High",
                }
                pred_doc = PredictionDocument(
                    userId=user_id,
                    prediction=stage_labels.get(fused.stage, fused.stage),
                    confidence=min(100.0, max(0.0, round(fused.confidence * 100, 1))),
                    riskLevel=risk_labels.get(fused.risk_category, fused.risk_category),
                    modelVersion="v1.0",
                    uploadedFileId=mri_image.filename if mri_image.filename else None,
                    clinicalInputs={
                        "cag_repeat": cag_repeat,
                        "motor_score": round(eff_motor, 1) if eff_motor is not None else None,
                        "memory_score": round(eff_memory, 1) if eff_memory is not None else None,
                        "functional_score": round(eff_functional, 1) if eff_functional is not None else None,
                        "age": age,
                    },
                    stageProbabilities={
                        "pre_manifest": round(fused.pre_manifest_prob, 4) if fused.pre_manifest_prob is not None else 0.0,
                        "early": round(fused.early_prob, 4) if fused.early_prob is not None else 0.0,
                        "advanced": round(fused.advanced_prob, 4) if fused.advanced_prob is not None else 0.0,
                    },
                    progression12mo=round(fused.progression_12mo, 2) if fused.progression_12mo is not None else 0.0,
                    progression24mo=round(fused.progression_24mo, 2) if fused.progression_24mo is not None else 0.0,
                )
                await save_prediction(pred_doc)
                logger.info("Successfully persisted image prediction to MongoDB for user=%s", user_id)
            except Exception as db_err:
                logger.warning("Failed to save prediction to DB: %s", db_err, exc_info=True)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image prediction failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Prediction failed: {e}")


# ─── Migration: move predictions into user documents ───


@app.post(
    "/migrate-predictions",
    summary="Migrate Predictions into User Documents",
    description="Moves existing predictions from the standalone predictions collection into their respective user documents.",
    tags=["Admin"],
)
async def migrate_predictions():
    """One-time migration: embed standalone predictions into user docs."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    db = get_database()
    migrated = 0
    skipped = 0

    # Find all predictions that have a real userId (not anonymous)
    cursor = db.predictions.find({})
    async for pred in cursor:
        user_id_str = pred.get("userId", "anonymous")
        if user_id_str == "anonymous":
            skipped += 1
            continue

        try:
            # Build the embedded doc (remove _id and userId)
            import uuid
            embed_doc = {k: v for k, v in pred.items() if k not in ("_id", "userId")}
            embed_doc["predictionId"] = str(uuid.uuid4())[:12]

            # Push into user document
            result = await db.users.update_one(
                {"_id": ObjectId(user_id_str)},
                {"$push": {"predictions": embed_doc}},
            )

            if result.modified_count > 0:
                # Remove from standalone collection
                await db.predictions.delete_one({"_id": pred["_id"]})
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            logger.warning("Migration skip for pred %s: %s", pred["_id"], e)
            skipped += 1

    return {
        "message": "Migration complete",
        "migrated": migrated,
        "skipped": skipped,
    }


# ─── Test Results: embedded under user documents ───


class SaveTestRequest(BaseModel):
    """Request body for saving a test result."""

    testType: str = Field(..., description="Type of test: 'memory' or 'motor'")
    testName: str = Field(..., description="Specific test name, e.g. 'word_recall'")
    score: float = Field(..., description="Test score achieved")
    total: float = Field(..., description="Maximum possible score")
    percentage: float = Field(..., description="Score percentage")
    details: dict = Field(default_factory=dict, description="Additional test-specific data")


@app.post(
    "/users/{user_id}/tests",
    summary="Save Test Result",
    description="Save a cognitive or motor test result under a user's profile.",
    tags=["Tests"],
)
async def save_test_result_endpoint(user_id: str, request: SaveTestRequest):
    """Save a test result embedded in the user document and test_results collection."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    test_doc = {
        "testType": request.testType,
        "testName": request.testName,
        "score": request.score,
        "total": request.total,
        "percentage": request.percentage,
        "details": request.details,
        "takenAt": datetime.now(timezone.utc).isoformat(),
    }

    try:
        test_res = TestResultDocument(
            testName=request.testName,
            testCategory=request.testType,
            score=request.score,
            total=request.total,
            percentage=min(100.0, max(0.0, request.percentage)),
            details=request.details,
            patientId=user_id,
        )
        await save_test_result(test_res)
        return {"message": "Test result saved", "test": test_doc}
    except Exception as e:
        logger.error("Failed to save test result: %s", e)
        raise HTTPException(500, f"Failed to save test result: {e}")


@app.get(
    "/tests",
    summary="Get All Test Results",
    description="Retrieve all test results across users, optionally filtered by patient or category.",
    tags=["Tests"],
)
async def get_all_tests(
    patient_id: str | None = None,
    category: str | None = None,
    skip: int = 0,
    limit: int = 100,
):
    """Get test results from standalone collection."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    results = await get_test_results(
        patient_id=patient_id, category=category, skip=skip, limit=limit
    )
    return {"tests": results, "count": len(results)}


@app.get(
    "/users/{user_id}/tests",
    summary="Get User Test Results",
    description="Retrieve all test results for a specific user.",
    tags=["Tests"],
)
async def get_user_tests(user_id: str, test_type: str | None = None):
    """Get test results from user document or test_results collection."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    db = get_database()

    try:
        tests = []
        if ObjectId.is_valid(user_id):
            user = await db.users.find_one(
                {"_id": ObjectId(user_id)},
                {"tests": 1, "name": 1},
            )
            if user and "tests" in user:
                tests = user.get("tests", [])

        # Fallback to standalone test_results if none found in user document
        if not tests:
            raw_tests = await get_test_results(patient_id=user_id, category=test_type)
            tests = [
                {
                    "testType": t.get("testCategory", ""),
                    "testName": t.get("testName", ""),
                    "score": t.get("score", 0),
                    "total": t.get("total", 0),
                    "percentage": t.get("percentage", 0),
                    "details": t.get("details", {}),
                    "takenAt": str(t.get("createdAt", "")),
                }
                for t in raw_tests
            ]

        # Filter by test type if specified
        if test_type:
            tests = [t for t in tests if t.get("testType") == test_type]

        # Sort by most recent first
        tests.sort(key=lambda t: str(t.get("takenAt", "")), reverse=True)

        return {
            "userName": user.get("name", "Unknown") if 'user' in locals() and user else "User",
            "userId": user_id,
            "tests": tests,
            "totalTests": len(tests),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get test results: %s", e)
        raise HTTPException(500, f"Failed to get test results: {e}")


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
    cag_repeat: float | None = Form(
        default=None,
        ge=10.0,
        le=120.0,
        description="CAG repeat count (Optional: Normal: 10–26, HD: 36–120)",
    ),
    motor_score: float | None = Form(
        default=None,
        ge=0.0,
        le=100.0,
        description="Digital Motor Assessment score (0–100%)",
    ),
    memory_score: float | None = Form(
        default=None,
        ge=0.0,
        le=100.0,
        description="Digital Memory/Cognitive score (0–100%)",
    ),
    functional_score: float | None = Form(
        default=None,
        ge=0.0,
        le=100.0,
        description="Daily Functional Independence score (0–100%)",
    ),
    age: float = Form(
        ...,
        ge=18.0,
        le=90.0,
        description="Patient age in years (18–90)",
    ),
    # Legacy parameters
    uhdrs_motor: float | None = Form(default=None),
    uhdrs_cognitive: float | None = Form(default=None),
    tfc_score: float | None = Form(default=None),
    symptoms: str | None = Form(
        default=None,
        description="Comma-separated symptom IDs (e.g. 'chorea,dystonia,depression'). Optional.",
    ),
    user_id: str = Form(
        default="anonymous",
        description="Logged-in user ID for linking predictions to accounts",
    ),
) -> PredictionResponse:
    """Run HD prediction with full explainability pipeline.

    Accepts multipart/form-data with an optional MRI file and
    digital assessment measurements. Returns staging classification,
    progression forecasts, and XAI outputs.
    """
    # Check if 3D MRI pipeline is required
    if mri_file is not None and (_pipeline is None or not _pipeline.is_loaded):
        raise HTTPException(
            status_code=503,
            detail="3D MRI pipeline not loaded. Service is starting up or in degraded mode.",
        )

    # Build clinical input
    try:
        clinical = ClinicalInput(
            cag_repeat=cag_repeat,
            motor_score=motor_score if motor_score is not None else 85.0,
            memory_score=memory_score if memory_score is not None else 85.0,
            functional_score=functional_score if functional_score is not None else 100.0,
            age=age,
            uhdrs_motor=uhdrs_motor,
            uhdrs_cognitive=uhdrs_cognitive,
            tfc_score=tfc_score,
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
        if mri_path is not None and _pipeline is not None and _pipeline.is_loaded:
            result = _pipeline.predict(
                mri_path=mri_path,
                clinical=clinical,
                generate_heatmap=True,
                generate_shap=True,
            )
        else:
            # Standalone digital assessment scoring (when no 3D MRI volume is uploaded)
            from neurosense.api.clinical_scoring import compute_clinical_score
            import uuid

            # Parse comma-separated symptoms list
            symptom_list = (
                [s.strip() for s in symptoms.split(",") if s.strip()]
                if symptoms
                else None
            )

            clinical_res = compute_clinical_score(
                cag_repeat=cag_repeat,
                motor_score=clinical.motor_score,
                memory_score=clinical.memory_score,
                functional_score=clinical.functional_score,
                age=age,
                uhdrs_motor=uhdrs_motor,
                uhdrs_cognitive=uhdrs_cognitive,
                tfc_score=tfc_score,
                symptoms=symptom_list,
            )

            feature_vals = {
                "cag_repeat": float(cag_repeat) if cag_repeat is not None else 0.0,
                "motor_score": float(clinical.motor_score),
                "memory_score": float(clinical.memory_score),
                "age": float(age),
                "symptoms": float(len(symptom_list)) if symptom_list else 0.0,
            }
            shap_features = [
                SHAPFeature(name=name, value=feature_vals.get(name, 0.0), impact=impact)
                for name, impact in sorted(
                    clinical_res.feature_impacts.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                if name in feature_vals
            ]

            result = PredictionResponse(
                stage=clinical_res.stage,
                confidence=clinical_res.confidence,
                stage_probabilities=StageProbabilities(
                    pre_manifest=clinical_res.pre_manifest_prob,
                    early=clinical_res.early_prob,
                    advanced=clinical_res.advanced_prob,
                ),
                progression_12mo=clinical_res.progression_12mo,
                progression_24mo=clinical_res.progression_24mo,
                risk_category=clinical_res.risk_category,
                gradcam_url=None,
                shap_features=shap_features,
                processing_time_s=0.05,
                request_id=str(uuid.uuid4())[:8],
            )

        # Persist to MongoDB
        if not is_connected():
            await ensure_connected()

        if is_connected():
            try:
                stage_labels = {
                    "pre_manifest": "Pre-manifest HD",
                    "early": "Early Huntington's",
                    "advanced": "Advanced Huntington's",
                }
                risk_labels = {
                    "low": "Low",
                    "medium": "Medium",
                    "high": "High",
                }
                pred_doc = PredictionDocument(
                    userId=user_id,
                    prediction=stage_labels.get(result.stage, result.stage),
                    confidence=min(100.0, max(0.0, round(result.confidence * 100, 1))),
                    riskLevel=risk_labels.get(result.risk_category, result.risk_category),
                    modelVersion="v1.0",
                    uploadedFileId=mri_file.filename if mri_file else None,
                    clinicalInputs={
                        "cag_repeat": cag_repeat,
                        "motor_score": clinical.motor_score if clinical else None,
                        "memory_score": clinical.memory_score if clinical else None,
                        "functional_score": clinical.functional_score if clinical else None,
                        "age": age,
                    },
                    stageProbabilities={
                        "pre_manifest": round(result.stage_probabilities.pre_manifest, 4) if result.stage_probabilities else 0.0,
                        "early": round(result.stage_probabilities.early, 4) if result.stage_probabilities else 0.0,
                        "advanced": round(result.stage_probabilities.advanced, 4) if result.stage_probabilities else 0.0,
                    },
                    progression12mo=round(result.progression_12mo, 2) if result.progression_12mo is not None else 0.0,
                    progression24mo=round(result.progression_24mo, 2) if result.progression_24mo is not None else 0.0,
                )
                await save_prediction(pred_doc)
                logger.info("Successfully persisted prediction to MongoDB for user=%s", user_id)
            except Exception as db_err:
                logger.warning("Failed to save prediction to DB: %s", db_err, exc_info=True)

        return result

    except HTTPException:
        raise
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


# ═════════════════════════════════════════════════════════════════
#  User & Prediction History Endpoints
# ═════════════════════════════════════════════════════════════════


class CreateUserRequest(BaseModel):
    """Request body for creating a new user."""

    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    role: str = Field(default="patient", description="User role")
    age: float = Field(..., ge=0.0, le=120.0, description="User's age")
    gender: str = Field(..., description="User's gender")


class CreateUserResponse(BaseModel):
    """Response after creating a user."""

    _id: str
    message: str


@app.post(
    "/users",
    response_model=CreateUserResponse,
    summary="Create User",
    description="Register a new user in the database.",
    tags=["Users"],
)
async def create_user_endpoint(
    request: CreateUserRequest,
) -> CreateUserResponse:
    """Create a new user record."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    try:
        user_doc = UserDocument(
            name=request.name,
            email=request.email,
            password=request.password,
            role=request.role,
            age=request.age,
            gender=request.gender,
        )
        user_id = await create_user(user_doc)
        return CreateUserResponse(
            _id=user_id,
            message=f"User '{request.name}' created successfully",
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        logger.error("Failed to create user: %s", e)
        raise HTTPException(500, f"Failed to create user: {e}")


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


@app.post(
    "/login",
    summary="User Login",
    description="Authenticate a user with email and password.",
    tags=["Users"],
)
async def login_endpoint(request: LoginRequest):
    """Authenticate user and return profile."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    user = await get_user_by_email(request.email)
    if user is None:
        raise HTTPException(401, "Invalid email or password")

    # Simple password check (plaintext — matches how register stores it)
    if user.get("password") != request.password:
        raise HTTPException(401, "Invalid email or password")

    # Don't return password
    user.pop("password", None)
    return {"user": user, "message": "Login successful"}


@app.get(
    "/users",
    summary="List Users",
    description="List all registered users (passwords excluded).",
    tags=["Users"],
)
async def list_users_endpoint(
    skip: int = 0,
    limit: int = 100,
):
    """List all users, paginated."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    users = await list_users(skip=skip, limit=limit)
    return {"users": users, "count": len(users)}


@app.get(
    "/users/{user_id}",
    summary="Get User",
    description="Get a user record by ID.",
    tags=["Users"],
)
async def get_user_endpoint(user_id: str):
    """Get a single user by _id."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(404, f"User '{user_id}' not found")
    # Don't return password
    user.pop("password", None)
    return user


@app.get(
    "/users/{user_id}/predictions",
    summary="User Predictions",
    description="Get all prediction history for a user.",
    tags=["Users"],
)
async def get_user_predictions_endpoint(
    user_id: str,
    skip: int = 0,
    limit: int = 50,
):
    """Get all predictions for a specific user."""
    if not is_connected():
        await ensure_connected()
    if not is_connected():
        raise HTTPException(503, "Database not available")

    predictions = await get_predictions_by_user(
        user_id, skip=skip, limit=limit
    )
    return {"userId": user_id, "predictions": predictions, "count": len(predictions)}


@app.get(
    "/predictions",
    summary="All Predictions",
    description="Get all prediction records across all users.",
    tags=["Predictions"],
)
async def list_predictions_endpoint(
    skip: int = 0,
    limit: int = 50,
):
    """List all predictions, most recent first."""
    if not is_connected():
        raise HTTPException(503, "Database not available")

    predictions = await get_all_predictions(skip=skip, limit=limit)
    return {"predictions": predictions, "count": len(predictions)}


@app.get(
    "/predictions/{prediction_id}",
    summary="Get Prediction",
    description="Get a single prediction by its ID.",
    tags=["Predictions"],
)
async def get_prediction_endpoint(prediction_id: str):
    """Get a single prediction by _id."""
    if not is_connected():
        raise HTTPException(503, "Database not available")

    prediction = await get_prediction_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(404, f"Prediction '{prediction_id}' not found")
    return prediction


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
