"""
test_api.py — Pytest test suite for the Fake News Detection ML API.

Covers:
  GET  /
  GET  /health
  POST /predict  (with & without model loaded)
  Validation errors (empty, whitespace, missing field, type mismatch, oversize)

Run with:
    pytest tests/ -v
"""

import types
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient that exercises the real app (no model files required)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _make_mock_model(predict_return=1, proba=None, classes=None):
    """Return a minimal sklearn-like mock model."""
    model = MagicMock()
    model.predict.return_value = np.array([predict_return])

    if proba is not None:
        model.predict_proba.return_value = np.array([proba])
        model.classes_ = classes or [0, 1]
    else:
        del model.predict_proba   # attribute must not exist

    return model


def _make_mock_vectorizer(output_shape=(1, 10)):
    vect = MagicMock()
    vect.transform.return_value = np.zeros(output_shape)
    return vect


# ──────────────────────────────────────────────
# GET /
# ──────────────────────────────────────────────

class TestRoot:
    def test_status_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_response_body(self, client):
        data = client.get("/").json()
        assert data["message"] == "Fake News Detection ML API is running"


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "vectorizer_loaded" in data

    def test_degraded_when_no_artifacts(self, client):
        """Without pkl files the status should be 'degraded'."""
        with (
            patch("app.predictor._model", None),
            patch("app.predictor._vectorizer", None),
        ):
            data = client.get("/health").json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["vectorizer_loaded"] is False

    def test_healthy_when_artifacts_loaded(self, client):
        mock_model = _make_mock_model()
        mock_vect  = _make_mock_vectorizer()
        with (
            patch("app.predictor._model", mock_model),
            patch("app.predictor._vectorizer", mock_vect),
        ):
            data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is True


# ──────────────────────────────────────────────
# POST /predict — no artifacts (503)
# ──────────────────────────────────────────────

class TestPredictNoArtifacts:
    def test_503_when_model_missing(self, client):
        with (
            patch("app.predictor._model", None),
            patch("app.predictor._vectorizer", _make_mock_vectorizer()),
        ):
            resp = client.post("/predict", json={"news": "Some news text"})
        assert resp.status_code == 503

    def test_503_when_vectorizer_missing(self, client):
        with (
            patch("app.predictor._model", _make_mock_model()),
            patch("app.predictor._vectorizer", None),
        ):
            resp = client.post("/predict", json={"news": "Some news text"})
        assert resp.status_code == 503

    def test_503_when_both_missing(self, client):
        with (
            patch("app.predictor._model", None),
            patch("app.predictor._vectorizer", None),
        ):
            resp = client.post("/predict", json={"news": "Some news text"})
        assert resp.status_code == 503


# ──────────────────────────────────────────────
# POST /predict — valid input (with mocked model)
# ──────────────────────────────────────────────

class TestPredictValidInput:
    """All tests mock model + vectorizer so no .pkl files are needed."""

    def _post(self, client, news_text, predict_return=1, proba=None, classes=None):
        mock_model = _make_mock_model(predict_return, proba, classes)
        mock_vect  = _make_mock_vectorizer()
        with (
            patch("app.predictor._model", mock_model),
            patch("app.predictor._vectorizer", mock_vect),
        ):
            return client.post("/predict", json={"news": news_text})

    def test_fake_prediction_integer_label(self, client):
        resp = self._post(client, "Breaking news: politicians lie!", predict_return=1)
        assert resp.status_code == 200
        assert resp.json()["prediction"] == "FAKE"

    def test_real_prediction_integer_label(self, client):
        resp = self._post(client, "Scientists confirm water boils at 100°C.", predict_return=0)
        assert resp.status_code == 200
        assert resp.json()["prediction"] == "REAL"

    def test_fake_prediction_string_label(self, client):
        resp = self._post(client, "Aliens land in New York.", predict_return="FAKE")
        assert resp.status_code == 200
        assert resp.json()["prediction"] == "FAKE"

    def test_real_prediction_string_label(self, client):
        resp = self._post(client, "Sun rises in the east.", predict_return="REAL")
        assert resp.status_code == 200
        assert resp.json()["prediction"] == "REAL"

    def test_confidence_present_when_proba_available(self, client):
        resp = self._post(
            client,
            "Shocking conspiracy exposed!",
            predict_return=1,
            proba=[0.08, 0.92],
            classes=[0, 1],
        )
        data = resp.json()
        assert resp.status_code == 200
        assert data["confidence"] is not None
        assert 0.0 <= data["confidence"] <= 1.0

    def test_confidence_null_when_no_proba(self, client):
        resp = self._post(client, "A normal article.", predict_return=0, proba=None)
        data = resp.json()
        assert resp.status_code == 200
        assert data["confidence"] is None

    def test_response_shape(self, client):
        resp = self._post(client, "Some article text.", predict_return=1)
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data


# ──────────────────────────────────────────────
# POST /predict — invalid input (422)
# ──────────────────────────────────────────────

class TestPredictInvalidInput:
    def test_empty_string(self, client):
        resp = client.post("/predict", json={"news": ""})
        assert resp.status_code == 422

    def test_whitespace_only(self, client):
        resp = client.post("/predict", json={"news": "   "})
        assert resp.status_code == 422

    def test_missing_news_field(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_wrong_type_integer(self, client):
        resp = client.post("/predict", json={"news": 12345})
        # FastAPI coerces int → str for string fields; if it does, the text "12345"
        # is valid. Either 200 or 422 is acceptable — we just assert not 500.
        assert resp.status_code != 500

    def test_wrong_type_null(self, client):
        resp = client.post("/predict", json={"news": None})
        assert resp.status_code == 422

    def test_oversized_input(self, client):
        """50 001 characters must be rejected."""
        resp = client.post("/predict", json={"news": "a" * 50_001})
        assert resp.status_code == 422

    def test_extra_fields_ignored(self, client):
        """Pydantic v2 ignores extra fields by default — should not cause errors."""
        mock_model = _make_mock_model(predict_return=1)
        mock_vect  = _make_mock_vectorizer()
        with (
            patch("app.predictor._model", mock_model),
            patch("app.predictor._vectorizer", mock_vect),
        ):
            resp = client.post("/predict", json={"news": "Valid text", "extra_key": "ignored"})
        assert resp.status_code == 200
