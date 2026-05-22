"""Topic modelling for legal contract clause collections.

Supports two backends:
- KMeans on TF-IDF vectors (fast, interpretable cluster centroids)
- NMF (Non-negative Matrix Factorisation) for soft topic assignments
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopicResult:
    """Result for a single document / clause after topic assignment."""

    text: str
    topic_id: int
    topic_label: str
    top_terms: list[str]


@dataclass
class TopicModelSummary:
    """Aggregate output after fitting the topic model on a corpus."""

    n_topics: int
    topic_labels: list[str]
    topic_top_terms: list[list[str]]
    assignments: list[TopicResult]
    coherence_score: float | None = None


class TopicModeler:
    """Unsupervised topic modelling over a corpus of contract clauses.

    Two modes are supported, selected via *method*:

    - ``"kmeans"``: TF-IDF vectorisation followed by KMeans clustering.
      Good for hard cluster assignment and centroid inspection.
    - ``"nmf"``: TF-IDF followed by Non-negative Matrix Factorisation.
      Good for soft topic weights and interpretable term-topic matrices.

    Usage::

        modeler = TopicModeler(n_topics=8, method="nmf")
        summary = modeler.fit_transform(clause_texts)
        print(summary.topic_top_terms)
    """

    def __init__(
        self,
        n_topics: int = 8,
        method: str = "kmeans",
        max_features: int = 5_000,
        n_top_terms: int = 10,
    ) -> None:
        """
        Args:
            n_topics: Number of topics / clusters to discover.
            method: ``"kmeans"`` or ``"nmf"``.
            max_features: Vocabulary size cap for TF-IDF.
            n_top_terms: How many top terms to surface per topic.
        """
        if method not in ("kmeans", "nmf"):
            raise ValueError(f"method must be 'kmeans' or 'nmf', got '{method}'")
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.n_top_terms = n_top_terms
        self._vectorizer: object | None = None
        self._model: object | None = None

    def fit_transform(self, texts: list[str]) -> TopicModelSummary:
        """Fit the topic model on *texts* and return topic assignments.

        Args:
            texts: List of clause or sentence strings to cluster.

        Returns:
            TopicModelSummary with per-clause assignments and topic metadata.
        """
        from sklearn.decomposition import NMF  # noqa: PLC0415
        from sklearn.cluster import KMeans  # noqa: PLC0415
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        feature_names: list[str] = self._vectorizer.get_feature_names_out().tolist()

        if self.method == "nmf":
            return self._run_nmf(texts, tfidf_matrix, feature_names)
        return self._run_kmeans(texts, tfidf_matrix, feature_names)

    def transform(self, texts: list[str]) -> list[TopicResult]:
        """Assign topics to *texts* using an already-fitted model.

        Args:
            texts: New clause texts to assign to existing topics.

        Returns:
            List of TopicResult objects.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if self._model is None or self._vectorizer is None:
            raise RuntimeError("Call fit_transform() before transform().")
        import numpy as np  # noqa: PLC0415

        tfidf_matrix = self._vectorizer.transform(texts)  # type: ignore[union-attr]
        feature_names: list[str] = self._vectorizer.get_feature_names_out().tolist()

        if self.method == "nmf":
            W = self._model.transform(tfidf_matrix)  # type: ignore[union-attr]
            assignments = np.argmax(W, axis=1).tolist()
            topic_top_terms = self._top_terms_nmf(feature_names)
        else:
            assignments = self._model.predict(tfidf_matrix).tolist()  # type: ignore[union-attr]
            topic_top_terms = self._top_terms_kmeans(feature_names)

        return [
            TopicResult(
                text=text,
                topic_id=int(tid),
                topic_label=f"Topic {tid}",
                top_terms=topic_top_terms[int(tid)],
            )
            for text, tid in zip(texts, assignments)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_nmf(
        self, texts: list[str], matrix: object, feature_names: list[str]
    ) -> TopicModelSummary:
        from sklearn.decomposition import NMF  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        self._model = NMF(n_components=self.n_topics, random_state=42, max_iter=400)
        W: np.ndarray = self._model.fit_transform(matrix)  # type: ignore[union-attr]
        top_terms = self._top_terms_nmf(feature_names)
        assignments_idx = np.argmax(W, axis=1).tolist()
        assignments = [
            TopicResult(
                text=t,
                topic_id=int(tid),
                topic_label=f"Topic {tid}",
                top_terms=top_terms[int(tid)],
            )
            for t, tid in zip(texts, assignments_idx)
        ]
        return TopicModelSummary(
            n_topics=self.n_topics,
            topic_labels=[f"Topic {i}" for i in range(self.n_topics)],
            topic_top_terms=top_terms,
            assignments=assignments,
        )

    def _run_kmeans(
        self, texts: list[str], matrix: object, feature_names: list[str]
    ) -> TopicModelSummary:
        from sklearn.cluster import KMeans  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        self._model = KMeans(n_clusters=self.n_topics, random_state=42, n_init="auto")
        self._model.fit(matrix)  # type: ignore[union-attr]
        labels: list[int] = self._model.labels_.tolist()
        top_terms = self._top_terms_kmeans(feature_names)
        assignments = [
            TopicResult(
                text=t,
                topic_id=int(tid),
                topic_label=f"Topic {tid}",
                top_terms=top_terms[int(tid)],
            )
            for t, tid in zip(texts, labels)
        ]
        return TopicModelSummary(
            n_topics=self.n_topics,
            topic_labels=[f"Topic {i}" for i in range(self.n_topics)],
            topic_top_terms=top_terms,
            assignments=assignments,
        )

    def _top_terms_nmf(self, feature_names: list[str]) -> list[list[str]]:
        import numpy as np  # noqa: PLC0415

        H: np.ndarray = self._model.components_  # type: ignore[union-attr]
        return [
            [feature_names[i] for i in np.argsort(row)[-self.n_top_terms:][::-1]]
            for row in H
        ]

    def _top_terms_kmeans(self, feature_names: list[str]) -> list[list[str]]:
        import numpy as np  # noqa: PLC0415

        centroids: np.ndarray = self._model.cluster_centers_  # type: ignore[union-attr]
        return [
            [feature_names[i] for i in np.argsort(row)[-self.n_top_terms:][::-1]]
            for row in centroids
        ]
