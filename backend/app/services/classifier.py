"""Clause-type classifier for CUAD legal contract analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ClauseCategory(str, Enum):
    """The seven clause categories derived from the CUAD taxonomy."""

    TERMINATION = "Termination"
    LIABILITY = "Liability"
    IP_RIGHTS = "IP Rights"
    GOVERNANCE = "Governance"
    PAYMENT = "Payment"
    DURATION = "Duration"
    OTHER = "Other"


@dataclass
class ClausePrediction:
    text: str
    predicted_category: ClauseCategory
    confidence: float
    probabilities: dict[str, float]


class ClauseClassifier:
    """Multi-class classifier that assigns each contract clause to one of the
    seven CUAD-derived categories.

    Wraps a trained scikit-learn pipeline (TF-IDF + LogisticRegression).
    Falls back to a keyword heuristic if no model is loaded.

    Usage::

        clf = ClauseClassifier()
        clf.load()  # loads default model path
        prediction = clf.predict("Either party may terminate with 30 days notice.")
    """

    _DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "ml_models" / "clause_classifier.joblib"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, model_path: str | Path | None = None) -> None:
        """Load a trained joblib pipeline.

        Args:
            model_path: Path to the .joblib file. Uses default if None.
        """
        import joblib

        path = Path(model_path) if model_path else self._DEFAULT_MODEL_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Train it first: cd backend && python train.py"
            )

        self._model = joblib.load(path)
        self._is_loaded = True

    def predict(self, clause_text: str) -> ClausePrediction:
        """Classify a single clause string.

        Args:
            clause_text: Raw contract clause text.

        Returns:
            ClausePrediction with predicted category and per-class probabilities.
        """
        if not self._is_loaded:
            try:
                self.load()
            except FileNotFoundError:
                return self._stub_prediction(clause_text)

        return self._predict_with_model(clause_text)

    def predict_batch(self, clauses: list[str]) -> list[ClausePrediction]:
        """Classify a list of clause strings in one call."""
        if not self._is_loaded:
            try:
                self.load()
            except FileNotFoundError:
                return [self._stub_prediction(c) for c in clauses]

        proba_matrix = self._model.predict_proba(clauses)
        classes = self._model.classes_.tolist()

        results = []
        for i, text in enumerate(clauses):
            proba = proba_matrix[i]
            idx = int(np.argmax(proba))
            results.append(ClausePrediction(
                text=text,
                predicted_category=ClauseCategory(classes[idx]),
                confidence=float(proba[idx]),
                probabilities={c: float(p) for c, p in zip(classes, proba)},
            ))
        return results

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return raw probability matrix — used by LIME explainer.

        Args:
            texts: List of clause strings.

        Returns:
            numpy array of shape (n_samples, n_classes).
        """
        if not self._is_loaded:
            self.load()
        return self._model.predict_proba(texts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_with_model(self, text: str) -> ClausePrediction:
        proba_array: np.ndarray = self._model.predict_proba([text])[0]
        classes: list[str] = self._model.classes_.tolist()
        idx = int(np.argmax(proba_array))
        return ClausePrediction(
            text=text,
            predicted_category=ClauseCategory(classes[idx]),
            confidence=float(proba_array[idx]),
            probabilities={c: float(p) for c, p in zip(classes, proba_array)},
        )

    @staticmethod
    def _stub_prediction(text: str) -> ClausePrediction:
        """Keyword-heuristic stub used before a real model is loaded."""
        lower = text.lower()
        if any(w in lower for w in ("terminat", "expir", "cancel")):
            cat = ClauseCategory.TERMINATION
        elif any(w in lower for w in ("liabil", "indemni", "damages")):
            cat = ClauseCategory.LIABILITY
        elif any(w in lower for w in ("intellectual", "patent", "copyright", "ip", "license")):
            cat = ClauseCategory.IP_RIGHTS
        elif any(w in lower for w in ("govern", "jurisdict", "dispute", "insurance")):
            cat = ClauseCategory.GOVERNANCE
        elif any(w in lower for w in ("pay", "revenue", "audit", "price", "fee")):
            cat = ClauseCategory.PAYMENT
        elif any(w in lower for w in ("date", "renew", "notice period", "term of")):
            cat = ClauseCategory.DURATION
        else:
            cat = ClauseCategory.OTHER

        uniform = 1.0 / len(ClauseCategory)
        return ClausePrediction(
            text=text,
            predicted_category=cat,
            confidence=0.6,
            probabilities={c.value: uniform for c in ClauseCategory},
        )