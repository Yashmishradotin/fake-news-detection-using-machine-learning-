"""
schemas.py — Pydantic request/response models for the Fake News Detection API.

Member 2 responsibility: input validation, output typing.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────

class NewsRequest(BaseModel):
    """
    Payload for POST /predict.
    Member 3 (Java backend) sends this JSON body.
    """

    news: str = Field(
        ...,
        min_length=1,
        max_length=50_000,          # ~50 KB — reject absurdly large inputs
        description="The news article text to classify.",
        examples=["Scientists discover a new species of frog in the Amazon rainforest."],
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
    'confidence' is omitted when the model does not support predict_proba.
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
        description="Probability of the predicted class (0.0–1.0). "
                    "Null when the model does not expose probabilities.",
        examples=[0.92],
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool
    vectorizer_loaded: bool


class RootResponse(BaseModel):
    """Response body for GET /."""

    message: str = Field(
        ...,
        examples=["Fake News Detection ML API is running"],
    )


class ErrorResponse(BaseModel):
    """Standard error envelope for 4xx / 5xx responses."""

    error: str
    detail: Optional[str] = None
