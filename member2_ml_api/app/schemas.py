"""
schemas.py — Pydantic request/response models for the Fake News Detection API.

Member 2 responsibility: input validation, output typing, model selection schemas.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class NewsRequest(BaseModel):
    """
    Payload for POST /predict.
    Supports optional model selection: 'best', 'lr', 'svm', 'rf', 'xgb'.
    """

    news: str = Field(
        ...,
        min_length=1,
        max_length=50_000,          # ~50 KB — reject absurdly large inputs
        description="The news article text to classify.",
        examples=["Scientists discover a new species of frog in the Amazon rainforest."],
    )

    model: Optional[str] = Field(
        default="best",
        description="ML Model to use: 'best', 'lr', 'svm', 'rf', 'xgb'. Defaults to 'best'.",
        examples=["best"],
    )

    @field_validator("news")
    @classmethod
    def news_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("The 'news' field must not be empty or whitespace-only.")
        return stripped


# ──────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    JSON response returned by POST /predict.
    Includes prediction, confidence probability, and model info.
    """

    prediction: str = Field(
        ...,
        description="Classification result: 'REAL' or 'FAKE'.",
        examples=["FAKE"],
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class (0.0–1.0).",
        examples=[0.92],
    )
    model_used: Optional[str] = Field(
        default="Logistic Regression",
        description="Human readable name of the model used.",
        examples=["Logistic Regression"],
    )
    model_key: Optional[str] = Field(
        default="lr",
        description="Model identifier key (lr, svm, rf, xgb).",
        examples=["lr"],
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    vectorizer_loaded: bool
    available_models: list[str] = Field(default_factory=list)


class RootResponse(BaseModel):
    """Response body for GET /."""

    message: str = Field(
        ...,
        examples=["Fake News Detection ML API is running"],
    )


class ModelBenchmarkItem(BaseModel):
    """Benchmark metrics for a specific model."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    filename: str


class ModelsInfoResponse(BaseModel):
    """Response body for GET /models benchmark metrics."""
    best_model: str
    best_model_name: str
    total_samples: int
    features_count: int
    models: Dict[str, ModelBenchmarkItem]


class ErrorResponse(BaseModel):
    """Standard error envelope for 4xx / 5xx responses."""

    error: str
    detail: Optional[str] = None
