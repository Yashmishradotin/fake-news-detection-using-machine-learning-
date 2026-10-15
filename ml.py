"""
ml.py — Multi-Model Machine Learning Pipeline for Fake News Detection.

Technologies & Innovations used:
- ML / NLP: Pandas, NumPy, NLTK, Scikit-learn
- Models: Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), XGBoost (XGB)
- Feature Extraction: NLTK text preprocessing + TF-IDF Vectorization
- Artifact Serialization: Joblib (.pkl) & JSON metrics
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Configure UTF-8 stdout if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Add member2_ml_api to sys.path to import preprocessing
sys.path.append(os.path.abspath("member2_ml_api"))
from app.preprocessing import clean_text

MODELS_DIR = os.path.join("member2_ml_api", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  MULTI-MODEL TRAINING PIPELINE (LR, SVM, RF, XGB)")
print("=" * 60)

# =========================================================
# STEP 1: Load Dataset using Pandas
# =========================================================
csv_file_path = "dataset.csv"
if not os.path.exists(csv_file_path):
    print(f"Error: {csv_file_path} not found.")
    sys.exit(1)

print(f"\n[1/5] Loading dataset using Pandas from '{csv_file_path}'...")
df = pd.read_csv(csv_file_path)
print(f"      Loaded {len(df)} news samples.")
print(f"      Class distribution:\n{df['label'].value_counts().to_string()}")

# Normalize labels
df['label'] = df['label'].astype(str).str.strip().str.upper()

# =========================================================
# STEP 2: NLTK NLP Preprocessing
# =========================================================
print("\n[2/5] Applying NLTK NLP Preprocessing (cleaning, stopwords, lemmatization)...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Ensure no empty strings
df['cleaned_text'] = df['cleaned_text'].replace("", "news article")

# =========================================================
# STEP 3: TF-IDF Feature Extraction
# =========================================================
print("\n[3/5] Extracting TF-IDF Features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(df['cleaned_text'])

y = df['label'].values
# Numeric mapping for XGBoost: REAL -> 0, FAKE -> 1
y_numeric = np.where(y == "FAKE", 1, 0)

# Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test, y_train_num, y_test_num = train_test_split(
    X_vectorized, y, y_numeric, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"      Feature matrix shape: {X_vectorized.shape}")

# =========================================================
# STEP 4: Train & Evaluate 4 Machine Learning Models
# =========================================================
print("\n[4/5] Training & Evaluating 4 Models (LR, SVM, Random Forest, XGBoost)...")

models = {
    "lr": {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "is_numeric": False,
        "filename": "lr_model.pkl"
    },
    "svm": {
        "name": "Support Vector Machine",
        "model": SVC(kernel="linear", probability=True, random_state=42),
        "is_numeric": False,
        "filename": "svm_model.pkl"
    },
    "rf": {
        "name": "Random Forest",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "is_numeric": False,
        "filename": "rf_model.pkl"
    },
    "xgb": {
        "name": "XGBoost",
        "model": XGBClassifier(eval_metric="logloss", random_state=42),
        "is_numeric": True,
        "filename": "xgb_model.pkl"
    }
}

metrics_results = {}
best_model_key = None
best_f1 = -1.0

print("\n" + "-" * 75)
print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 75)

for key, item in models.items():
    mdl = item["model"]
    is_num = item["is_numeric"]

    # Train
    if is_num:
        mdl.fit(X_train, y_train_num)
        preds = mdl.predict(X_test)
        preds_label = np.where(preds == 1, "FAKE", "REAL")
    else:
        mdl.fit(X_train, y_train)
        preds_label = mdl.predict(X_test)

    # Compute metrics
    acc = float(accuracy_score(y_test, preds_label))
    prec = float(precision_score(y_test, preds_label, pos_label="FAKE", zero_division=0))
    rec = float(recall_score(y_test, preds_label, pos_label="FAKE", zero_division=0))
    f1 = float(f1_score(y_test, preds_label, pos_label="FAKE", zero_division=0))

    metrics_results[key] = {
        "name": item["name"],
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "filename": item["filename"]
    }

    print(f"{item['name']:<25} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}%")

    if f1 > best_f1:
        best_f1 = f1
        best_model_key = key

print("-" * 75)
print(f"[BEST MODEL] {models[best_model_key]['name']} (F1: {best_f1*100:.2f}%)")

# =========================================================
# STEP 5: Save All Model Artifacts & Metrics
# =========================================================
print("\n[5/5] Saving model artifacts to member2_ml_api/models/...")

# Save Vectorizer
vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)
print(f"      [OK] Saved Vectorizer: {vectorizer_path}")

# Save Each Model
for key, item in models.items():
    file_path = os.path.join(MODELS_DIR, item["filename"])
    joblib.dump(item["model"], file_path)
    print(f"      [OK] Saved {item['name']}: {file_path}")

# Save the best model as trained_model.pkl for compatibility
best_model_obj = models[best_model_key]["model"]
trained_model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
joblib.dump(best_model_obj, trained_model_path)
print(f"      [OK] Saved Default/Best Model ({models[best_model_key]['name']}) -> trained_model.pkl")

# Save Metrics JSON
metrics_payload = {
    "best_model": best_model_key,
    "best_model_name": models[best_model_key]["name"],
    "total_samples": len(df),
    "features_count": X_vectorized.shape[1],
    "models": metrics_results
}
metrics_json_path = os.path.join(MODELS_DIR, "model_metrics.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=4)
print(f"      [OK] Saved Metrics JSON: {metrics_json_path}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE! All 4 models and metrics are ready.")
print("=" * 60)
