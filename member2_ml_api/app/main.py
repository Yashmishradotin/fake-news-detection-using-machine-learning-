"""
main.py — FastAPI application entry point.

Member 2 responsibility:
  - Expose  GET  /
  - Expose  GET  /health
  - Expose  POST /predict
  - Configure CORS for Member 3's Java backend.
  - Load ML artifacts once at startup via lifespan event.
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
    """Load ML model and vectorizer when the server starts."""
    logger.info("🚀 Starting Fake News Detection ML API …")
    predictor.load_artifacts()

    if not predictor.is_model_loaded():
        logger.warning(
            "⚠️  trained_model.pkl is missing. "
            "POST /predict will return 503 until Member 1 supplies it."
        )
    if not predictor.is_vectorizer_loaded():
        logger.warning(
            "⚠️  vectorizer.pkl is missing. "
            "POST /predict will return 503 until Member 1 supplies it."
        )

    yield  # ← application runs here

    logger.info("🛑 Shutting down Fake News Detection ML API.")


# ──────────────────────────────────────────────
# App instance
# ──────────────────────────────────────────────

app = FastAPI(
    title="Fake News Detection ML API",
    description=(
        "**Member 2** of the Fake News Detection project.\n\n"
        "Accepts raw news text, transforms it with a TF-IDF vectorizer "
        "(Member 1's `vectorizer.pkl`), and classifies it as **REAL** or **FAKE** "
        "using a trained scikit-learn model (Member 1's `trained_model.pkl`).\n\n"
        "### Integration Guide for Member 3 (Java Backend)\n"
        "Send a `POST /predict` request with the body:\n"
        "```json\n{\"news\": \"Your news article text here...\"}\n```\n"
        "Receive:\n"
        "```json\n{\"prediction\": \"FAKE\", \"confidence\": 0.92}\n```\n"
        "`confidence` is `null` when the model does not expose probabilities."
    ),
    version="1.0.0",
    contact={"name": "Member 2 — ML API Engineer"},
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# CORS — allow Member 3's Java backend to call this API
# ──────────────────────────────────────────────

# Read from env so Member 3 can configure without code changes.
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://localhost:3000,http://localhost:4200",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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
    """Simple liveness ping. Returns a human-readable status message."""
    return RootResponse(message="Fake News Detection ML API is running")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check — model & vectorizer status",
    tags=["System"],
)
async def health():
    """
    Returns whether the ML model and vectorizer are loaded.
    Member 3 can poll this before sending prediction requests.
    """
    return HealthResponse(
        status="healthy" if (predictor.is_model_loaded() and predictor.is_vectorizer_loaded()) else "degraded",
        model_loaded=predictor.is_model_loaded(),
        vectorizer_loaded=predictor.is_vectorizer_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict — classify news as REAL or FAKE",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful classification.", "model": PredictionResponse},
        422: {"description": "Validation error (empty/missing/too-large 'news' field)."},
        503: {"description": "Model or vectorizer not loaded (Member 1 files missing)."},
        500: {"description": "Unexpected prediction error."},
    },
)
async def predict(payload: NewsRequest):
    """
    **POST /predict** — Main endpoint for Member 3 (Java backend).

    ### Request body
    ```json
    { "news": "Your news article text here..." }
    ```

    ### Response
    ```json
    { "prediction": "FAKE", "confidence": 0.92 }
    ```

    - `prediction` is always `"REAL"` or `"FAKE"`.
    - `confidence` is the probability of the predicted class (`null` if unavailable).

    ### Validation rules
    - `news` must be a non-empty string.
    - `news` must not be whitespace-only.
    - `news` must not exceed 50,000 characters.
    """

    # ── Check artifacts ──
    if not predictor.is_vectorizer_loaded() or not predictor.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ML artifacts not loaded.",
                "detail": (
                    "trained_model.pkl and/or vectorizer.pkl are missing. "
                    "Member 1 must place them in the models/ directory and restart the API."
                ),
                "model_loaded": predictor.is_model_loaded(),
                "vectorizer_loaded": predictor.is_vectorizer_loaded(),
            },
        )

    # ── Run prediction ──
    try:
        label, confidence = predictor.predict_news(payload.news)
    except (RuntimeError, ValueError) as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prediction failed.", "detail": str(exc)},
        ) from exc

    return PredictionResponse(prediction=label, confidence=confidence)
