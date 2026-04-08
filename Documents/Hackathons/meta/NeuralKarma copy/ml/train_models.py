"""
NeuralKarma — ML Model Training Pipeline
Trains real classifiers on real data from HuggingFace:
  1. Prosociality classifier (ProsocialDialog safety labels)
  2. Five ETHICS-axis classifiers (commonsense, deontology, justice, virtue, utilitarianism)

All models use TF-IDF vectorization + Logistic Regression with cross-validation.
Models are saved as .joblib files for production inference.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR = PROJECT_ROOT / "data" / "cache"


def ensure_model_dir():
    """Create model directory if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_utilitarianism(df):
    """
    Convert utilitarianism pairwise comparison format to binary classification.
    'baseline' is the more ethical option (label=1), 'less_pleasant' is less ethical (label=0).
    We interleave both into a single text+label DataFrame.
    """
    positive = pd.DataFrame({
        "text": df["baseline"].astype(str),
        "label": 1,
    })
    negative = pd.DataFrame({
        "text": df["less_pleasant"].astype(str),
        "label": 0,
    })
    combined = pd.concat([positive, negative], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def get_text_column(df, subset_name):
    """
    Intelligently find the text column in an ETHICS subset DataFrame.
    Different ETHICS subsets have different column structures.
    """
    if "input" in df.columns:
        return df["input"].astype(str)
    elif "scenario" in df.columns:
        # Deontology / Justice / Virtue
        if "excuse" in df.columns:
            return (df["scenario"].astype(str) + " " + df["excuse"].astype(str)).str.strip()
        return df["scenario"].astype(str)
    elif "text" in df.columns:
        return df["text"].astype(str)
    elif "baseline" in df.columns:
        # Utilitarianism — already preprocessed
        return df["text"].astype(str) if "text" in df.columns else df["baseline"].astype(str)
    else:
        # Fallback: concatenate all string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            return df[str_cols[0]].astype(str)
        raise ValueError(f"Cannot find text column in {subset_name}: {df.columns.tolist()}")


def get_label_column(df, subset_name):
    """Find the label column in a dataset."""
    if "label" in df.columns:
        return df["label"].astype(int)
    elif "is_short" in df.columns:
        return df["is_short"].astype(int)
    else:
        label_cols = [c for c in df.columns if "label" in c.lower()]
        if label_cols:
            return df[label_cols[0]].astype(int)
        raise ValueError(f"Cannot find label column in {subset_name}: {df.columns.tolist()}")


def balance_dataset(texts, labels, max_per_class=5000, random_state=42):
    """
    Balance a dataset by upsampling minority and downsampling majority classes.
    Caps at max_per_class to keep training fast.
    """
    df = pd.DataFrame({"text": texts, "label": labels})
    classes = df["label"].unique()
    balanced_parts = []

    for c in classes:
        class_df = df[df["label"] == c]
        if len(class_df) > max_per_class:
            class_df = class_df.sample(n=max_per_class, random_state=random_state)
        elif len(class_df) < max_per_class // 2 and len(class_df) > 10:
            class_df = resample(class_df, n_samples=min(max_per_class, len(class_df) * 3),
                                random_state=random_state, replace=True)
        balanced_parts.append(class_df)

    result = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=random_state)
    return result["text"].values, result["label"].values


def train_prosociality_model(prosocial_df):
    """
    Train prosociality classifier on ProsocialDialog safety labels.
    Maps safety labels to binary: safe/casual → prosocial (1), caution/intervention → not prosocial (0)
    """
    model_path = MODEL_DIR / "prosociality_model.joblib"
    vectorizer_path = MODEL_DIR / "prosociality_vectorizer.joblib"

    if model_path.exists() and vectorizer_path.exists():
        print("  [OK] Prosociality model already trained, loading from cache...")
        return joblib.load(model_path), joblib.load(vectorizer_path)

    print("  ⚙ Training Prosociality Classifier...")
    start = time.time()

    # Binary mapping: safe/casual → 1 (prosocial), caution/intervention → 0 (not)
    label_map = {
        "__ok__": 1,
        "__casual__": 1,
        "__probably_needs_caution__": 0,
        "__needs_caution__": 0,
        "__needs_intervention__": 0,
    }

    df = prosocial_df.copy()
    df["binary_label"] = df["safety_label"].map(label_map)
    df = df.dropna(subset=["binary_label", "context"])
    df["binary_label"] = df["binary_label"].astype(int)

    # Use context as input text
    texts = df["context"].astype(str).values
    labels = df["binary_label"].values

    # Balance and cap
    texts, labels = balance_dataset(texts, labels, max_per_class=8000)

    print(f"    Training on {len(texts):,} samples (balanced)...")

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)

    # Logistic Regression with cross-validation
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"    CV Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")

    # Final fit on all data
    model.fit(X, labels)

    # Save
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    elapsed = time.time() - start
    print(f"  [OK] Prosociality model trained in {elapsed:.1f}s")

    return model, vectorizer


def train_ethics_model(subset_name, subset_df):
    """
    Train a classifier for one ETHICS benchmark subset.
    """
    model_path = MODEL_DIR / f"ethics_{subset_name}_model.joblib"
    vectorizer_path = MODEL_DIR / f"ethics_{subset_name}_vectorizer.joblib"

    if model_path.exists() and vectorizer_path.exists():
        print(f"  [OK] ETHICS/{subset_name} model already trained, loading from cache...")
        return joblib.load(model_path), joblib.load(vectorizer_path)

    print(f"  ⚙ Training ETHICS/{subset_name} Classifier...")
    start = time.time()

    try:
        texts = get_text_column(subset_df, subset_name)
        labels = get_label_column(subset_df, subset_name)
    except ValueError as e:
        print(f"    [WARNING] Skipping {subset_name}: {e}")
        return None, None

    # Clean
    mask = texts.str.strip().str.len() > 0
    texts = texts[mask].values
    labels = labels[mask].values

    if len(texts) < 10:
        print(f"    [WARNING] Too few samples ({len(texts)}), skipping {subset_name}")
        return None, None

    # Balance
    texts, labels = balance_dataset(texts, labels, max_per_class=5000)
    print(f"    Training on {len(texts):,} samples...")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2 if len(texts) > 100 else 1,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )

    # Cross-validation
    n_splits = min(5, min(np.bincount(labels)))
    n_splits = max(2, n_splits)
    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"    CV Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
    except Exception:
        print(f"    (CV skipped due to small dataset)")

    model.fit(X, labels)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    elapsed = time.time() - start
    print(f"  [OK] ETHICS/{subset_name} model trained in {elapsed:.1f}s")

    return model, vectorizer


def train_all_models():
    """
    Train all NeuralKarma models. Main entry point for ML pipeline.
    Returns dict of all trained models and vectorizers.
    """
    ensure_model_dir()

    print("=" * 60)
    print("NeuralKarma — ML Model Training Pipeline")
    print("=" * 60)

    # Load cached datasets
    from data.download_datasets import download_all
    prosocial_df, ethics_data, norms_df = download_all()

    models = {}

    print("\n[1/6] Prosociality Model")
    model, vec = train_prosociality_model(prosocial_df)
    models["prosociality"] = {"model": model, "vectorizer": vec}

    axis_names = ["commonsense", "deontology", "justice", "virtue", "utilitarianism"]
    for i, subset_name in enumerate(axis_names):
        print(f"\n[{i+2}/6] ETHICS/{subset_name} Model")
        if subset_name in ethics_data:
            subset_df = ethics_data[subset_name]
            # Utilitarianism needs special preprocessing (pairwise → classification)
            if subset_name == "utilitarianism" and "baseline" in subset_df.columns and "label" not in subset_df.columns:
                subset_df = preprocess_utilitarianism(subset_df)
            model, vec = train_ethics_model(subset_name, subset_df)
            models[subset_name] = {"model": model, "vectorizer": vec}
        else:
            print(f"  [WARNING] No data for {subset_name}")

    print("\n" + "=" * 60)
    print("Training Summary:")
    for name, m in models.items():
        status = "[OK] Ready" if m.get("model") is not None else "[FAILED] Failed"
        print(f"  • {name}: {status}")
    print(f"  • Models Dir: {MODEL_DIR.resolve()}")
    print("=" * 60)

    return models


if __name__ == "__main__":
    train_all_models()
