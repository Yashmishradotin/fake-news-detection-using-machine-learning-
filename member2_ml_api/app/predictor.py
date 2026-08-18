"""
predictor.py — Model loading and prediction logic.

Member 2 responsibility:
  - Load trained_model.pkl and vectorizer.pkl ONCE at startup.
  - Expose predict_news(text) → (label, confidence | None).

⚠️  Do NOT train a model here.
⚠️  Do NOT create fake .pkl files.
⚠️  These files MUST be provided by Member 1.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths (overridable via environment variables)
# ──────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent        # member2_ml_api/
_MODELS_DIR = Path(os.getenv("MODELS_DIR", _BASE_DIR / "models"))

MODEL_PATH      = _MODELS_DIR / "trained_model.pkl"
VECTORIZER_PATH = _MODELS_DIR / "vectorizer.pkl"

# ──────────────────────────────────────────────
# Module-level singletons (loaded once at startup)
# ──────────────────────────────────────────────

_model      = None
_vectorizer = None


def load_artifacts() -> None:
    """
    Load model and vectorizer into module-level singletons.
    Called once when FastAPI starts up — NOT on every request.
    Logs a clear warning when files are missing (Member 1 must supply them).
    """
    global _model, _vectorizer

    # ── Vectorizer ──
    if VECTORIZER_PATH.exists():
        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("✅ Vectorizer loaded from %s", VECTORIZER_PATH)
        except Exception as exc:
            logger.error("❌ Failed to load vectorizer: %s", exc)
            _vectorizer = None
    else:
        logger.warning(
            "⚠️  vectorizer.pkl NOT found at %s. "
            "Member 1 must place it in the models/ directory.",
            VECTORIZER_PATH,
        )

    # ── Model ──
    if MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            logger.info("✅ Model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.error("❌ Failed to load model: %s", exc)
            _model = None
    else:
        logger.warning(
            "⚠️  trained_model.pkl NOT found at %s. "
            "Member 1 must place it in the models/ directory.",
            MODEL_PATH,
        )


def is_model_loaded() -> bool:
    return _model is not None


def is_vectorizer_loaded() -> bool:
    return _vectorizer is not None


# ──────────────────────────────────────────────
# Core prediction function
# ──────────────────────────────────────────────

def predict_news(news_text: str) -> Tuple[str, Optional[float]]:
    """
    Classify a news article as REAL or FAKE.

    Parameters
    ----------
    news_text : str
        The cleaned (already validated) article text.

    Returns
    -------
    (label, confidence)
        label      — "REAL" or "FAKE"
        confidence — float in [0, 1] if model supports predict_proba,
                     None otherwise.

    Raises
    ------
    RuntimeError  — when model or vectorizer is not loaded.
    ValueError    — when news_text is empty after stripping.
    Exception     — bubbles up any unexpected sklearn/numpy errors.
    """
    # ── Guards ──
    if _vectorizer is None:
        raise RuntimeError(
            "Vectorizer is not loaded. "
            "Ensure Member 1's vectorizer.pkl is present in models/."
        )
    if _model is None:
        raise RuntimeError(
            "Model is not loaded. "
            "Ensure Member 1's trained_model.pkl is present in models/."
        )

    text = news_text.strip()
    if not text:
        raise ValueError("news_text must not be empty.")

    # ── Transform ──
    try:
        features = _vectorizer.transform([text])
    except Exception as exc:
        logger.error("Vectorizer transform failed: %s", exc)
        raise RuntimeError(f"Feature extraction failed: {exc}") from exc

    # ── Predict ──
    try:
        raw_prediction = _model.predict(features)[0]
    except Exception as exc:
        logger.error("Model predict failed: %s", exc)
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    # ── Decode label ──
    # Member 1's model may use 0/1 integers or string labels.
    label = _decode_label(raw_prediction)

    # ── Confidence (optional) ──
    confidence: Optional[float] = None
    if hasattr(_model, "predict_proba"):
        try:
            proba = _model.predict_proba(features)[0]   # shape: (n_classes,)
            classes = list(_model.classes_)

            # Map our decoded label back to the class index to get its probability
            fake_idx  = _find_class_index(classes, "fake",  1)
            real_idx  = _find_class_index(classes, "real",  0)

            if label == "FAKE" and fake_idx is not None:
                confidence = float(proba[fake_idx])
            elif label == "REAL" and real_idx is not None:
                confidence = float(proba[real_idx])
            else:
                # Fallback: probability of the predicted class by position
                predicted_idx = int(np.argmax(proba))
                confidence = float(proba[predicted_idx])

            confidence = round(confidence, 4)
        except Exception as exc:
            logger.warning("predict_proba failed, skipping confidence: %s", exc)
            confidence = None

    logger.info("Prediction → %s (confidence=%s)", label, confidence)
    return label, confidence


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _decode_label(raw) -> str:
    """
    Convert whatever the model outputs (0/1 int, 'FAKE'/'REAL' string,
    'fake'/'real' lowercase, etc.) into the canonical 'REAL' or 'FAKE'.
    """
    normalized = str(raw).strip().upper()

    if normalized in ("1", "FAKE"):
        return "FAKE"
    if normalized in ("0", "REAL"):
        return "REAL"

    # Unexpected label — log it and return as-is uppercased
    logger.warning("Unexpected raw label from model: %r → returning '%s'", raw, normalized)
    return normalized


def _find_class_index(classes: list, name: str, integer_fallback: int) -> Optional[int]:
    """Return index of the matching class label (case-insensitive)."""
    for i, cls in enumerate(classes):
        if str(cls).strip().upper() == name.upper():
            return i
        if str(cls).strip() == str(integer_fallback):
            return i
    return None
