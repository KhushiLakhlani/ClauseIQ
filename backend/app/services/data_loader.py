"""CUAD dataset loader: downloads, parses, and categorises legal contract clauses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Clause-type → high-level category mapping
# ---------------------------------------------------------------------------

# All 41 CUAD question titles (lowercased) mapped to one of 8 categories.
# Keys are substrings that appear in the CUAD question name; the first match wins.
_CATEGORY_RULES: list[tuple[str, str]] = [
    # Termination
    ("termination for convenience", "Termination"),
    ("termination for cause", "Termination"),
    ("change of control", "Termination"),
    # Liability
    ("cap on liability", "Liability"),
    ("limitation of liability", "Liability"),
    ("indemnification", "Liability"),
    ("warranty duration", "Liability"),
    # IP Rights
    ("ip ownership", "IP Rights"),
    ("intellectual property", "IP Rights"),
    ("license grant", "IP Rights"),
    ("non-compete", "IP Rights"),
    ("non compete", "IP Rights"),
    ("source code escrow", "IP Rights"),
    ("exclusivity", "IP Rights"),
    ("most favored nation", "IP Rights"),
    # Confidentiality
    ("confidential", "Confidentiality"),
    ("non-disclosure", "Confidentiality"),
    ("non disclosure", "Confidentiality"),
    # Payment
    ("revenue/profit sharing", "Payment"),
    ("revenue profit sharing", "Payment"),
    ("minimum commitment", "Payment"),
    ("price restriction", "Payment"),
    ("audit rights", "Payment"),
    ("anti-assignment", "Payment"),
    # Governance
    ("governing law", "Governance"),
    ("jurisdiction", "Governance"),
    ("dispute resolution", "Governance"),
    ("third party beneficiary", "Governance"),
    ("insurance", "Governance"),
    ("covenant not to sue", "Governance"),
    ("uncapped liability", "Governance"),
    # Duration
    ("agreement date", "Duration"),
    ("effective date", "Duration"),
    ("expiration date", "Duration"),
    ("renewal term", "Duration"),
    ("notice period", "Duration"),
    ("post-termination services", "Duration"),
    ("rofr/rofo/rofn", "Duration"),
]


def _map_clause_type(raw_type: str) -> str:
    """Return the high-level category for a CUAD clause-type label.

    Args:
        raw_type: The raw CUAD question/label name (any casing).

    Returns:
        One of the 8 category strings, or ``"Other"`` if no rule matches.
    """
    lower = raw_type.lower()
    for fragment, category in _CATEGORY_RULES:
        if fragment in lower:
            return category
    return "Other"


# ---------------------------------------------------------------------------
# Core loading logic
# ---------------------------------------------------------------------------

def load_cuad_dataset() -> pd.DataFrame:
    """Download the CUAD dataset from Hugging Face and return a tidy DataFrame.

    Each row represents a single positive (annotated) clause span.

    Returns:
        DataFrame with columns:
        ``contract_id``, ``clause_text``, ``clause_type``, ``source_file``,
        ``category`` (high-level grouping of *clause_type*).

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        RuntimeError: If the dataset download or parsing fails.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to download CUAD. "
            "Install it with: pip install datasets"
        ) from exc

    try:
        raw = load_dataset("cuad", split="train", trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the CUAD dataset from Hugging Face: {exc}"
        ) from exc

    records: list[dict] = []

    for example in raw:
        contract_id: str = example.get("id", "")
        source_file: str = example.get("title", "")
        context: str = example.get("context", "")

        qa_pairs: dict = example.get("answers", {})
        # CUAD stores answers as a dict-of-lists: {"text": [...], "answer_start": [...]}
        texts: list[str] = qa_pairs.get("text", []) if isinstance(qa_pairs, dict) else []

        clause_type: str = example.get("question_name", "")
        if not clause_type:
            # Fallback: derive from the question field if available
            clause_type = example.get("question", "")

        for span in texts:
            span = span.strip()
            if not span:
                continue
            records.append(
                {
                    "contract_id": contract_id,
                    "clause_text": span,
                    "clause_type": clause_type,
                    "source_file": source_file,
                    "category": _map_clause_type(clause_type),
                }
            )

    if not records:
        raise RuntimeError(
            "Parsed zero clause records from the CUAD dataset. "
            "The dataset schema may have changed — inspect the raw fields."
        )

    df = pd.DataFrame(records, columns=["contract_id", "clause_text", "clause_type", "source_file", "category"])
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_processed_data(
    df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    filename: str = "cuad_processed.csv",
) -> Path:
    """Persist *df* to a CSV file under *output_dir*.

    Args:
        df: The processed DataFrame returned by :func:`load_cuad_dataset`.
        output_dir: Directory to write the CSV into. Defaults to
            ``<repo_root>/data/``.
        filename: Name of the output file (default: ``cuad_processed.csv``).

    Returns:
        Absolute :class:`~pathlib.Path` of the written file.
    """
    if output_dir is None:
        # Resolve relative to this file: backend/app/services/ → ../../.. → repo root
        output_dir = Path(__file__).resolve().parents[4] / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the processed CUAD DataFrame.

    Args:
        df: DataFrame produced by :func:`load_cuad_dataset`.
    """
    total = len(df)
    avg_len = df["clause_text"].str.len().mean()

    print(f"\n{'='*50}")
    print(f"  CUAD Dataset — Processed Summary")
    print(f"{'='*50}")
    print(f"  Total clauses   : {total:,}")
    print(f"  Avg clause length: {avg_len:,.0f} characters")
    print(f"\n  Distribution by category:")

    dist = (
        df.groupby("category")
        .size()
        .rename("count")
        .sort_values(ascending=False)
    )
    pct = (dist / total * 100).round(1)

    col_w = max(len(c) for c in dist.index) + 2
    for cat in dist.index:
        bar = "#" * int(pct[cat] / 2)
        print(f"    {cat:<{col_w}} {dist[cat]:>6,}  ({pct[cat]:>5.1f}%)  {bar}")

    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def load_and_save(output_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Download CUAD, print stats, and persist to CSV in one call.

    Args:
        output_dir: Passed through to :func:`save_processed_data`.

    Returns:
        The processed DataFrame.
    """
    df = load_cuad_dataset()
    print_stats(df)
    save_processed_data(df, output_dir=output_dir)
    return df


if __name__ == "__main__":
    load_and_save()
