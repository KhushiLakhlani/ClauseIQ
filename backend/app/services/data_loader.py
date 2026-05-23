"""CUAD dataset loader: parses JSON, maps clause types, and exports processed data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Clause-type → high-level category mapping
# ---------------------------------------------------------------------------

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
    # IP Rights (additional license types)
    ("non-transferable license", "IP Rights"),
    ("irrevocable or perpetual", "IP Rights"),
    ("affiliate license", "IP Rights"),
    ("unlimited/all-you-can-eat", "IP Rights"),
    # Liability (additional)
    ("liquidated damages", "Liability"),
    # Governance (restrictive covenants)
    ("no-solicit", "Governance"),
    ("non-disparagement", "Governance"),
    ("competitive restriction", "IP Rights"),
    # Payment (additional)
    ("volume restriction", "Payment"),
]


def _map_clause_type(raw_type: str) -> str:
    """Return the high-level category for a CUAD clause-type label."""
    lower = raw_type.lower()
    for fragment, category in _CATEGORY_RULES:
        if fragment in lower:
            return category
    return "Other"


def _extract_clause_type(question: str) -> str:
    """Pull the clause-type label from a CUAD question string.

    CUAD questions follow the pattern:
      'Highlight the parts ... related to "Document Name" that should ...'
    This extracts the quoted portion.
    """
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1)
    return question.strip()


# ---------------------------------------------------------------------------
# Core loading logic
# ---------------------------------------------------------------------------

def load_cuad_dataset(json_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Parse the CUAD JSON file and return a tidy DataFrame.

    Each row represents a single positive (annotated) clause span.

    Args:
        json_path: Path to CUADv1.json. Defaults to data/CUADv1.json
            relative to the repo root.

    Returns:
        DataFrame with columns: contract_id, clause_text, clause_type,
        source_file, category.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        RuntimeError: If parsing produces zero records.
    """
    if json_path is None:
        json_path = Path(__file__).resolve().parents[3] / "data" / "CUADv1.json"
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"CUAD JSON not found at {json_path}. "
            "Download it first:\n"
            "  python -c \"import urllib.request; "
            "urllib.request.urlretrieve("
            "'https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip', "
            "'data/cuad_data.zip')\"\n"
            "  Then unzip data/cuad_data.zip into data/"
        )

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    records: list[dict] = []

    for contract in raw.get("data", []):
        source_file: str = contract.get("title", "")

        for para in contract.get("paragraphs", []):
            for qa in para.get("qas", []):
                if qa.get("is_impossible", True):
                    continue

                question: str = qa.get("question", "")
                clause_type = _extract_clause_type(question)

                for answer in qa.get("answers", []):
                    span = answer.get("text", "").strip()
                    if not span:
                        continue
                    records.append({
                        "contract_id": qa.get("id", ""),
                        "clause_text": span,
                        "clause_type": clause_type,
                        "source_file": source_file,
                        "category": _map_clause_type(clause_type),
                    })

    if not records:
        raise RuntimeError(
            "Parsed zero clause records from the CUAD dataset. "
            "The JSON schema may have changed — inspect the raw fields."
        )

    df = pd.DataFrame(records, columns=[
        "contract_id", "clause_text", "clause_type", "source_file", "category"
    ])
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_processed_data(
    df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    filename: str = "cuad_processed.csv",
) -> Path:
    """Persist df to a CSV file under output_dir."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data"
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
    """Print a human-readable summary of the processed CUAD DataFrame."""
    total = len(df)
    avg_len = df["clause_text"].str.len().mean()

    print(f"\n{'='*60}")
    print(f"  ClauseIQ — CUAD Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total clauses    : {total:,}")
    print(f"  Unique contracts : {df['source_file'].nunique():,}")
    print(f"  Clause types     : {df['clause_type'].nunique()}")
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
        bar = "█" * int(pct[cat] / 2)
        print(f"    {cat:<{col_w}} {dist[cat]:>6,}  ({pct[cat]:>5.1f}%)  {bar}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def load_and_save(
    json_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Parse CUAD JSON, print stats, and persist to CSV in one call."""
    df = load_cuad_dataset(json_path=json_path)
    print_stats(df)
    save_processed_data(df, output_dir=output_dir)
    return df


if __name__ == "__main__":
    load_and_save()