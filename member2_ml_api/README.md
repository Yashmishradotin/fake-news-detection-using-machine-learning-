# 🛡️ Fake News Detection — Member 2: Python ML API

> **Role**: Turn Member 1's trained scikit-learn model into a production-ready REST API.

---

## 📁 Project Structure

```
member2_ml_api/
├── models/
│   ├── trained_model.pkl   ← ⚠️ PROVIDED BY MEMBER 1
│   └── vectorizer.pkl      ← ⚠️ PROVIDED BY MEMBER 1
├── app/
│   ├── __init__.py
│   ├── main.py             ← FastAPI app, routes, CORS
│   ├── predictor.py        ← Model loading + predict_news()
│   └── schemas.py          ← Pydantic request/response models
├── tests/
│   └── test_api.py         ← pytest suite (no .pkl files needed)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1 — Create & activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add Member 1's model files

Copy `trained_model.pkl` and `vectorizer.pkl` into `models/`:

```
member2_ml_api/
└── models/
    ├── trained_model.pkl   ← from Member 1
    └── vectorizer.pkl      ← from Member 1
```

> **Without these files the API starts but `/predict` returns HTTP 503.**

### 4 — Configure environment (optional)

```bash
cp .env.example .env
# Edit ALLOWED_ORIGINS to match Member 3's Java backend URL
```

### 5 — Run the API

```bash
# From inside member2_ml_api/
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Endpoints

| Method | Path       | Description                         |
|--------|------------|-------------------------------------|
| GET    | `/`        | Liveness ping                       |
| GET    | `/health`  | Model & vectorizer status           |
| POST   | `/predict` | Classify news as REAL or FAKE       |
| GET    | `/docs`    | Swagger UI (auto-generated)         |
| GET    | `/redoc`   | ReDoc documentation                 |

---

## 🔗 Integration Guide for Member 3 (Java Backend)

### Endpoint
```
POST http://<api-host>:8000/predict
Content-Type: application/json
```

### Request body
```json
{
  "news": "Your news article text here..."
}
```

### Successful response — `200 OK`
```json
{
  "prediction": "FAKE",
  "confidence": 0.92
}
```

- `prediction` — always `"REAL"` or `"FAKE"` (never anything else).
- `confidence` — probability of the predicted class (`0.0` – `1.0`).  
  **`null`** when the model does not support `predict_proba`.

### Error responses

| HTTP | When |
|------|------|
| `422 Unprocessable Entity` | Missing / empty / too-large `news` field |
| `503 Service Unavailable`  | `.pkl` files not loaded yet |
| `500 Internal Server Error`| Unexpected prediction failure |

### Java (Spring Boot) example
```java
RestTemplate rest = new RestTemplate();
String url = "http://localhost:8000/predict";

Map<String, String> body = Map.of("news", articleText);
ResponseEntity<Map> response = rest.postForEntity(url, body, Map.class);

String prediction  = (String) response.getBody().get("prediction");
Double confidence  = (Double) response.getBody().get("confidence");
```

---

## ✅ Validation Rules

| Rule | Limit |
|------|-------|
| `news` required | Must be present in JSON body |
| Minimum length | 1 character (non-whitespace) |
| Maximum length | 50 000 characters |
| Data type | String |

---

## 🧪 Running Tests

> Tests use `unittest.mock` — **no `.pkl` files required**.

```bash
# From inside member2_ml_api/
pytest tests/ -v
```

Expected output (all 22 tests passing):
```
tests/test_api.py::TestRoot::test_status_200              PASSED
tests/test_api.py::TestRoot::test_response_body           PASSED
tests/test_api.py::TestHealth::...                        PASSED
tests/test_api.py::TestPredictNoArtifacts::...            PASSED
tests/test_api.py::TestPredictValidInput::...             PASSED
tests/test_api.py::TestPredictInvalidInput::...           PASSED
```

---

## 🔒 CORS

Configured via `ALLOWED_ORIGINS` environment variable (comma-separated).  
Default: `http://localhost:8080, http://localhost:3000, http://localhost:4200`

Member 3: add your Java backend URL to `.env`:
```env
ALLOWED_ORIGINS=http://localhost:8080,http://your-java-backend-host:port
```

---

## 📖 Interactive Docs

Visit **http://localhost:8000/docs** for Swagger UI once the server is running.

---

## ⚠️ Important Notes

- **Do NOT** commit `trained_model.pkl` or `vectorizer.pkl` to git.
- **Do NOT** train a new model here — use Member 1's files.
- Labels `0` / `"real"` / `"REAL"` → `"REAL"`;  `1` / `"fake"` / `"FAKE"` → `"FAKE"`.
- If Member 1's model uses different label conventions, update `_decode_label()` in `predictor.py`.
