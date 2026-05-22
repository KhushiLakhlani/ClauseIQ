"""LIME-based explainability for clause classification predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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
    raw_lime_explanation: Any | None = None


class ClauseExplainer:
    """Generates LIME explanations for individual clause classifications.

    Wraps ``lime.lime_text.LimeTextExplainer`` and converts its output into
    a structured, JSON-serialisable format for the API layer.

    Usage::

        explainer = ClauseExplainer(class_names=[c.value for c in ClauseCategory])
        explanation = explainer.explain(
            clause_text="The parties agree to keep all information confidential.",
            predict_fn=classifier.predict_proba,
        )
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> None:
        """
        Args:
            class_names: Ordered list of class label strings (must match
                the probability vector returned by *predict_fn*).
            num_features: Maximum number of features to include per explanation.
            num_samples: Number of perturbed samples LIME generates internally.
        """
        from app.services.classifier import ClauseCategory  # noqa: PLC0415

        self.class_names: list[str] = class_names or [c.value for c in ClauseCategory]
        self.num_features = num_features
        self.num_samples = num_samples
        self._lime_explainer: Any | None = None

    def _get_lime_explainer(self) -> Any:
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer  # noqa: PLC0415

            self._lime_explainer = LimeTextExplainer(class_names=self.class_names)
        return self._lime_explainer

    def explain(
        self,
        clause_text: str,
        predict_fn: Callable[[list[str]], Any],
        label_index: int | None = None,
    ) -> ClauseExplanation:
        """Explain a single clause prediction using LIME.

        Args:
            clause_text: The contract clause to explain.
            predict_fn: A callable that takes a list of strings and returns
                a 2-D probability array with shape ``(n_samples, n_classes)``.
                Typically ``sklearn_pipeline.predict_proba``.
            label_index: Index of the class to explain. If *None*, the
                predicted (argmax) class is used automatically.

        Returns:
            ClauseExplanation with ranked feature contributions.
        """
        import numpy as np  # noqa: PLC0415

        explainer = self._get_lime_explainer()

        proba_matrix = predict_fn([clause_text])
        proba_row: list[float] = proba_matrix[0].tolist()
        predicted_idx = int(np.argmax(proba_row))
        target_idx = label_index if label_index is not None else predicted_idx

        lime_exp = explainer.explain_instance(
            clause_text,
            predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=[target_idx],
        )

        raw_pairs: list[tuple[str, float]] = lime_exp.as_list(label=target_idx)
        top_features = [
            ExplanationEntry(
                feature=feat,
                weight=weight,
                positive=weight > 0,
            )
            for feat, weight in raw_pairs
        ]

        return ClauseExplanation(
            clause_text=clause_text,
            predicted_label=self.class_names[predicted_idx],
            confidence=proba_row[predicted_idx],
            top_features=top_features,
            raw_lime_explanation=lime_exp,
        )

    def explain_batch(
        self,
        clauses: list[str],
        predict_fn: Callable[[list[str]], Any],
    ) -> list[ClauseExplanation]:
        """Explain multiple clauses sequentially.

        Args:
            clauses: List of clause text strings.
            predict_fn: Probability callable (see :meth:`explain`).

        Returns:
            List of ClauseExplanation objects in the same order as *clauses*.
        """
        return [self.explain(c, predict_fn) for c in clauses]
