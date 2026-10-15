"""
predictor.py — Multi-Model loading and prediction logic with NLTK preprocessing.

Supports:
  - Default _model and _vectorizer for backward compatibility and unit tests.
  - Multi-model dictionary: 'lr', 'svm', 'rf', 'xgb'.
  - NLTK text preprocessing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np

from app.preprocessing import clean_text

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent        # member2_ml_api/
_MODELS_DIR = Path(os.getenv("MODELS_DIR", _BASE_DIR / "models"))

VECTORIZER_PATH = _MODELS_DIR / "vectorizer.pkl"
METRICS_PATH    = _MODELS_DIR / "model_metrics.json"

MODEL_PATHS = {
    "lr": _MODELS_DIR / "lr_model.pkl",
    "svm": _MODELS_DIR / "svm_model.pkl",
    "rf": _MODELS_DIR / "rf_model.pkl",
    "xgb": _MODELS_DIR / "xgb_model.pkl",
}

# ──────────────────────────────────────────────
# Module-level singletons (used by tests & app)
# ──────────────────────────────────────────────

_model      = None
_vectorizer = None
_loaded_models: Dict[str, Any] = {}
_metrics: Dict[str, Any] = {}
_best_model_key: str = "lr"

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "svm": "Support Vector Machine",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


def load_artifacts() -> None:
    """Load vectorizer, all 4 ML models, and metrics into singletons at startup."""
    global _model, _vectorizer, _loaded_models, _metrics, _best_model_key

    # ── 1. Vectorizer ──
    if VECTORIZER_PATH.exists():
        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Vectorizer loaded from %s", VECTORIZER_PATH)
        except Exception as exc:
            logger.error("Failed to load vectorizer: %s", exc)
            _vectorizer = None
    else:
        logger.warning("vectorizer.pkl NOT found at %s", VECTORIZER_PATH)

    # ── 2. Metrics & Best Model ──
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
                _best_model_key = _metrics.get("best_model", "lr")
            logger.info("Metrics loaded. Best model is %s", _best_model_key)
        except Exception as exc:
            logger.warning("Failed to load metrics json: %s", exc)

    # ── 3. Models ──
    _loaded_models.clear()
    for key, path in MODEL_PATHS.items():
        if path.exists():
            try:
                _loaded_models[key] = joblib.load(path)
                logger.info("Loaded model [%s] from %s", key, path.name)
            except Exception as exc:
                logger.error("Failed to load model [%s]: %s", key, exc)

    # Set default / primary _model
    fallback_path = _MODELS_DIR / "trained_model.pkl"
    if fallback_path.exists():
        try:
            _model = joblib.load(fallback_path)
        except Exception:
            _model = _loaded_models.get(_best_model_key)
    else:
        _model = _loaded_models.get(_best_model_key)


def is_model_loaded() -> bool:
    return _model is not None


def is_vectorizer_loaded() -> bool:
    return _vectorizer is not None


def get_available_models() -> list[str]:
    return list(_loaded_models.keys()) if _loaded_models else (["lr"] if _model is not None else [])


def get_metrics() -> Dict[str, Any]:
    return _metrics


# ──────────────────────────────────────────────
# Core prediction function
# ──────────────────────────────────────────────

def predict_news(
    news_text: str,
    model_choice: str = "best"
) -> Tuple[str, Optional[float], str, str]:
    """
    Classify news article as REAL or FAKE using chosen model.

    Returns:
    (label, confidence, model_name, model_key)
    """
    if _vectorizer is None:
        raise RuntimeError("Vectorizer is not loaded. Train models or verify vectorizer.pkl.")
    if _model is None and not _loaded_models:
        raise RuntimeError("Model is not loaded.")

    chosen_key = model_choice.lower().strip() if model_choice else "best"

    # Select active model
    if chosen_key in _loaded_models and chosen_key != "best":
        model = _loaded_models[chosen_key]
        model_name = MODEL_DISPLAY_NAMES.get(chosen_key, chosen_key.upper())
    elif _model is not None:
        model = _model
        model_name = MODEL_DISPLAY_NAMES.get(_best_model_key, "Logistic Regression")
        chosen_key = _best_model_key
    elif _best_model_key in _loaded_models:
        model = _loaded_models[_best_model_key]
        model_name = MODEL_DISPLAY_NAMES.get(_best_model_key, "Logistic Regression")
        chosen_key = _best_model_key
    else:
        chosen_key = next(iter(_loaded_models))
        model = _loaded_models[chosen_key]
        model_name = MODEL_DISPLAY_NAMES.get(chosen_key, chosen_key.upper())

    # 1. NLP Preprocessing with NLTK
    cleaned = clean_text(news_text)
    if not cleaned:
        cleaned = news_text.strip()

    # 2. Vectorize with TF-IDF
    features = _vectorizer.transform([cleaned])

    # 3. Predict
    raw_pred = model.predict(features)[0]
    label = _decode_label(raw_pred)

    # 4. Confidence via predict_proba
    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)[0]
            classes = list(getattr(model, "classes_", [0, 1]))

            fake_idx = _find_class_index(classes, "fake", 1)
            real_idx = _find_class_index(classes, "real", 0)

            if label == "FAKE" and fake_idx is not None:
                confidence = float(proba[fake_idx])
            elif label == "REAL" and real_idx is not None:
                confidence = float(proba[real_idx])
            else:
                pred_pos = int(np.argmax(proba))
                confidence = float(proba[pred_pos])

            confidence = round(confidence, 4)
        except Exception as exc:
            logger.warning("predict_proba failed for %s: %s", chosen_key, exc)
            confidence = None

    logger.info("Predict [%s] -> %s (conf=%s)", model_name, label, confidence)
    return label, confidence, model_name, chosen_key


def _decode_label(raw) -> str:
    """Normalize output label to REAL or FAKE."""
    normalized = str(raw).strip().upper()
    if normalized in ("1", "FAKE"):
        return "FAKE"
    if normalized in ("0", "REAL"):
        return "REAL"
    return normalized


def _find_class_index(classes: list, name: str, int_fallback: int) -> Optional[int]:
    """Find index of class name or numerical equivalent."""
    for i, cls in enumerate(classes):
        if str(cls).strip().upper() == name.upper():
            return i
        if str(cls).strip() == str(int_fallback):
            return i
    return None
