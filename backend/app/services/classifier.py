"""Clause-type classifier for CUAD legal contract analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ClauseCategory(str, Enum):
    """The eight clause categories derived from the CUAD taxonomy."""

    TERMINATION = "Termination"
    LIABILITY = "Liability"
    IP_RIGHTS = "IP Rights"
    CONFIDENTIALITY = "Confidentiality"
    PAYMENT = "Payment"
    GOVERNANCE = "Governance"
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
    eight CUAD-derived categories.

    In production this wraps a scikit-learn pipeline (TF-IDF + LogisticRegression
    or a fine-tuned transformer). The placeholder below returns deterministic
    stubs so the API is usable before model training.

    Usage::

        clf = ClauseClassifier()
        clf.load("ml_models/clause_clf_v1.joblib")
        prediction = clf.predict("Either party may terminate with 30 days notice.")
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._is_loaded: bool = False

    def load(self, model_path: str) -> None:
        """Load a serialised joblib pipeline from *model_path*.

        Args:
            model_path: Filesystem path to a ``joblib``-serialised sklearn pipeline.
        """
        import joblib  # noqa: PLC0415

        self._model = joblib.load(model_path)
        self._is_loaded = True

    def predict(self, clause_text: str) -> ClausePrediction:
        """Classify a single clause string.

        Args:
            clause_text: Raw or preprocessed clause text.

        Returns:
            ClausePrediction with predicted category and per-class probabilities.
        """
        if self._is_loaded and self._model is not None:
            return self._predict_with_model(clause_text)
        return self._stub_prediction(clause_text)

    def predict_batch(self, clauses: list[str]) -> list[ClausePrediction]:
        """Classify a list of clause strings in one call.

        Args:
            clauses: List of clause text strings.

        Returns:
            List of ClausePrediction objects in the same order as *clauses*.
        """
        return [self.predict(c) for c in clauses]

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
        elif any(w in lower for w in ("intellectual", "patent", "copyright", "ip")):
            cat = ClauseCategory.IP_RIGHTS
        elif any(w in lower for w in ("confidential", "proprietary", "secret")):
            cat = ClauseCategory.CONFIDENTIALITY
        elif any(w in lower for w in ("payment", "fee", "invoice", "price")):
            cat = ClauseCategory.PAYMENT
        elif any(w in lower for w in ("govern", "jurisdiction", "arbitrat")):
            cat = ClauseCategory.GOVERNANCE
        elif any(w in lower for w in ("term", "period", "duration", "year", "month")):
            cat = ClauseCategory.DURATION
        else:
            cat = ClauseCategory.OTHER

        uniform = 1.0 / len(ClauseCategory)
        probs = {c.value: uniform for c in ClauseCategory}
        probs[cat.value] = 0.7
        return ClausePrediction(
            text=text,
            predicted_category=cat,
            confidence=0.7,
            probabilities=probs,
        )
