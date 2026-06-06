"""LIME-based explainability for clause classification predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ExplanationEntry:
    """A single feature contribution from a LIME explanation."""
    feature: str
    weight: float
    positive: bool


@dataclass
class ClauseExplanation:
    """Full explanation for one clause prediction."""
    clause_text: str
    predicted_label: str
    confidence: float
    top_features: list[ExplanationEntry]


class ClauseExplainer:
    """Generates LIME explanations for individual clause classifications.

    Usage::

        from app.services.classifier import ClauseClassifier
        clf = ClauseClassifier()
        clf.load()

        explainer = ClauseExplainer()
        explanation = explainer.explain(
            clause_text="The parties agree to keep all information confidential.",
            predict_fn=clf.predict_proba,
        )
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        num_features: int = 10,
        num_samples: int = 300,
    ) -> None:
        from app.services.classifier import ClauseCategory

        self.class_names = class_names or [c.value for c in ClauseCategory]
        self.num_features = num_features
        self.num_samples = num_samples
        self._lime_explainer: Any | None = None

    def _get_lime_explainer(self) -> Any:
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._lime_explainer = LimeTextExplainer(class_names=self.class_names)
        return self._lime_explainer

    def explain(
        self,
        clause_text: str,
        predict_fn: Callable[[list[str]], np.ndarray],
        label_index: int | None = None,
    ) -> ClauseExplanation:
        """Explain a single clause prediction using LIME.

        Args:
            clause_text: The contract clause to explain.
            predict_fn: Callable taking list[str], returning (n_samples, n_classes) array.
            label_index: Index of class to explain. If None, uses predicted class.

        Returns:
            ClauseExplanation with ranked feature contributions.
        """
        explainer = self._get_lime_explainer()

        proba = predict_fn([clause_text])
        predicted_idx = int(np.argmax(proba[0]))
        target_idx = label_index if label_index is not None else predicted_idx

        lime_exp = explainer.explain_instance(
            clause_text,
            predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[target_idx],
        )

        raw_pairs = lime_exp.as_list(label=target_idx)
        top_features = [
            ExplanationEntry(
                feature=feat,
                weight=round(weight, 4),
                positive=weight > 0,
            )
            for feat, weight in raw_pairs
        ]

        return ClauseExplanation(
            clause_text=clause_text,
            predicted_label=self.class_names[predicted_idx],
            confidence=round(float(proba[0][predicted_idx]), 4),
            top_features=top_features,
        )