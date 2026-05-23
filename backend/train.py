"""
ClauseIQ — Model Training Pipeline
Trains a TF-IDF + Logistic Regression classifier on CUAD legal contract clauses.

Usage:
    python train.py
    python train.py --data_path data/cuad_processed.csv --test_size 0.2
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logging setup — gives us timestamped, professional output
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clauseiq.train")

# ---------------------------------------------------------------------------
# Phase 1: Data Loading & Validation
# ---------------------------------------------------------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    """Load and validate the processed CUAD dataset."""
    log.info(f"Loading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run this first: python -c 'from app.services.data_loader import load_and_save; load_and_save()'"
        )

    df = pd.read_csv(data_path)

    # Validation checks
    required_cols = {"clause_text", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with empty text
    initial_len = len(df)
    df = df.dropna(subset=["clause_text", "category"])
    df = df[df["clause_text"].str.strip().str.len() > 0]
    dropped = initial_len - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with empty/null text")

    log.info(f"Loaded {len(df):,} clauses across {df['category'].nunique()} categories")
    return df

# ---------------------------------------------------------------------------
# Phase 2: Model Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
) -> dict:
    """Train a TF-IDF + Logistic Regression pipeline.

    Args:
        df: DataFrame with 'clause_text' and 'category' columns.
        test_size: Fraction of data reserved for testing.
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range: N-gram range for TF-IDF (1,2) means unigrams + bigrams.
        C: Inverse regularization strength for Logistic Regression.
        random_state: Seed for reproducibility.

    Returns:
        Dict with trained pipeline, metrics, and split data.
    """
    X = df["clause_text"].values
    y = df["category"].values

    # --- Stratified train/test split ---
    # stratify=y ensures each category has proportional representation
    # in both train and test sets (critical for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    log.info(f"Train set: {len(X_train):,} samples")
    log.info(f"Test set:  {len(X_test):,} samples")

    # --- Build sklearn Pipeline ---
    # Pipeline chains TF-IDF vectorization and classification into one object.
    # This means the same preprocessing is applied during training AND prediction.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # Apply log normalization to term frequencies
            strip_accents="unicode",    # Handle accented characters in legal text
            min_df=2,                   # Ignore terms appearing in fewer than 2 documents
            max_df=0.95,                # Ignore terms appearing in >95% of documents
        )),
        ("clf", LogisticRegression(
            solver="lbfgs",             # Efficient solver for multinomial
            max_iter=1000,              # Legal text needs more iterations to converge
            class_weight="balanced",    # Automatically upweight rare categories
            C=C,                        # Regularization strength
            random_state=random_state,
        )),
    ])

    # --- Train ---
    log.info("Training TF-IDF + Logistic Regression pipeline...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    log.info(f"Training completed in {train_time:.1f}s")

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    # --- Cross-validation (5-fold) for robust estimate ---
    log.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_macro")

    return {
        "pipeline": pipeline,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "cv_scores": cv_scores,
        "train_time": train_time,
    }

# ---------------------------------------------------------------------------
# Phase 3: Evaluation & Persistence
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    """Print a detailed, professional evaluation report."""
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    cv_scores = results["cv_scores"]

    # Get sorted category labels for consistent ordering
    labels = sorted(set(y_test) | set(y_pred))

    print("\n" + "=" * 70)
    print("  ClauseIQ — Model Evaluation Report")
    print("=" * 70)

    print(f"\n  Train accuracy : {results['train_acc']:.4f}")
    print(f"  Test accuracy  : {results['test_acc']:.4f}")
    print(f"  Training time  : {results['train_time']:.1f}s")
    print(f"\n  Cross-validation F1 (macro):")
    print(f"    Mean : {cv_scores.mean():.4f}")
    print(f"    Std  : {cv_scores.std():.4f}")
    print(f"    Folds: {', '.join(f'{s:.4f}' for s in cv_scores)}")

    print(f"\n{'─' * 70}")
    print("  Per-Category Classification Report")
    print(f"{'─' * 70}")
    report = classification_report(y_test, y_pred, labels=labels, digits=3)
    print(report)

    print(f"{'─' * 70}")
    print("  Confusion Matrix")
    print(f"{'─' * 70}")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Print with row/column labels
    col_w = max(len(l) for l in labels)
    header = " " * (col_w + 2) + "  ".join(f"{l[:6]:>6}" for l in labels)
    print(f"  {header}")
    for i, label in enumerate(labels):
        row = "  ".join(f"{v:>6}" for v in cm[i])
        print(f"  {label:<{col_w}}  {row}")

    # Feature importance — top terms per category
    print(f"\n{'─' * 70}")
    print("  Top Predictive Terms per Category")
    print(f"{'─' * 70}")

    pipeline = results["pipeline"]
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    for i, category in enumerate(clf.classes_):
        coef = clf.coef_[i]
        top_indices = np.argsort(coef)[-8:][::-1]
        top_terms = [feature_names[j] for j in top_indices]
        print(f"  {category:20s} → {', '.join(top_terms)}")

    print("=" * 70 + "\n")


def save_model(results: dict, output_dir: Path) -> None:
    """Save the trained pipeline and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full pipeline (TF-IDF + classifier in one file)
    model_path = output_dir / "clause_classifier.joblib"
    joblib.dump(results["pipeline"], model_path)
    log.info(f"Model saved to {model_path}")

    # Save metadata for reproducibility
    meta = {
        "test_accuracy": float(results["test_acc"]),
        "train_accuracy": float(results["train_acc"]),
        "cv_f1_mean": float(results["cv_scores"].mean()),
        "cv_f1_std": float(results["cv_scores"].std()),
        "categories": sorted(results["pipeline"].classes_.tolist()),
        "n_train": len(results["X_train"]),
        "n_test": len(results["X_test"]),
    }
    import json
    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ClauseIQ classifier")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to cuad_processed.csv")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory to save trained model")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_features", type=int, default=10_000)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    # Resolve default paths relative to this file
    repo_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_path) if args.data_path else repo_root / "data" / "cuad_processed.csv"
    model_dir = Path(args.model_dir) if args.model_dir else Path(__file__).resolve().parent / "ml_models"

    # Phase 1: Load
    df = load_data(data_path)

    # Phase 2: Train
    results = train_model(df, test_size=args.test_size, max_features=args.max_features, C=args.C)

    # Phase 3: Evaluate & Save
    print_results(results)
    save_model(results, model_dir)

    log.info("Done. Model is ready for serving.")


if __name__ == "__main__":
    main()