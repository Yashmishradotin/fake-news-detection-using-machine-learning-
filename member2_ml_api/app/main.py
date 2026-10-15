"""
main.py — FastAPI application entry point with Multi-Model AI Support.

Exposes:
  - GET  /          → API Liveness ping
  - GET  /health    → Status of models & vectorizer
  - GET  /models    → Comparison & benchmark metrics for LR, SVM, Random Forest, XGBoost
  - POST /predict   → Classify news as REAL or FAKE using selected or best model
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import predictor
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    NewsRequest,
    PredictionResponse,
    RootResponse,
    ModelsInfoResponse,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan — load artifacts ONCE at startup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and vectorizer when server starts."""
    logger.info("Starting Fake News Detection ML API...")
    predictor.load_artifacts()

    if not predictor.is_model_loaded():
        logger.warning("No ML models found in models directory. Run ml.py to train.")
    else:
        logger.info(f"Loaded models: {predictor.get_available_models()}")

    if not predictor.is_vectorizer_loaded():
        logger.warning("vectorizer.pkl is missing. Run ml.py to train.")

    yield

    logger.info("Shutting down Fake News Detection ML API.")


# ──────────────────────────────────────────────
# App instance
# ──────────────────────────────────────────────

app = FastAPI(
    title="Fake News Detection ML API",
    description=(
        "**Production-grade Fake News Detection Engine**\n\n"
        "Features:\n"
        "- NLP Preprocessing with NLTK & TF-IDF Vectorization\n"
        "- 4 High-Performance ML Models: Logistic Regression, SVM, Random Forest, XGBoost\n"
        "- REST API with dynamic model selection and benchmark metrics\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
if _raw_origins.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Global exception handlers
# ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get(
    "/",
    response_model=RootResponse,
    summary="Root — API liveness check",
    tags=["System"],
)
async def root():
    """Simple liveness ping."""
    return RootResponse(message="Fake News Detection ML API is running")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check — model & vectorizer status",
    tags=["System"],
)
async def health():
    """Returns model loading status and list of active models."""
    return HealthResponse(
        status="healthy" if (predictor.is_model_loaded() and predictor.is_vectorizer_loaded()) else "degraded",
        model_loaded=predictor.is_model_loaded(),
        vectorizer_loaded=predictor.is_vectorizer_loaded(),
        available_models=predictor.get_available_models(),
    )


@app.get(
    "/models",
    response_model=ModelsInfoResponse,
    summary="Get all models benchmark metrics",
    tags=["Models"],
)
async def get_models_benchmark():
    """Returns the comparison metrics (Accuracy, F1-Score, etc.) for all trained models."""
    metrics = predictor.get_metrics()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics not found. Run python ml.py to generate benchmarks.",
        )
    return metrics


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict — classify news as REAL or FAKE",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful classification.", "model": PredictionResponse},
        422: {"description": "Validation error (empty/missing 'news')."},
        503: {"description": "Models not loaded."},
        500: {"description": "Unexpected prediction error."},
    },
)
async def predict(payload: NewsRequest):
    """
    Classify news article text as REAL or FAKE.
    Optional `model` parameter: 'best', 'lr', 'svm', 'rf', 'xgb'.
    """
    if not predictor.is_vectorizer_loaded() or not predictor.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ML artifacts not loaded.",
                "detail": "Models and vectorizer must be generated by running python ml.py.",
                "model_loaded": predictor.is_model_loaded(),
                "vectorizer_loaded": predictor.is_vectorizer_loaded(),
            },
        )

    try:
        label, confidence, model_name, model_key = predictor.predict_news(
            news_text=payload.news,
            model_choice=payload.model or "best"
        )
    except (RuntimeError, ValueError) as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prediction failed.", "detail": str(exc)},
        ) from exc

    return PredictionResponse(
        prediction=label,
        confidence=confidence,
        model_used=model_name,
        model_key=model_key,
    )
