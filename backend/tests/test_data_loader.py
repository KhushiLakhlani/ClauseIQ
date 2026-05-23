"""Unit tests for data_loader — no network calls; HuggingFace download is mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.data_loader import (
    _map_clause_type,
    load_cuad_dataset,
    print_stats,
    save_processed_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_CATEGORIES = {
    "Termination",
    "Liability",
    "IP Rights",
    "Confidentiality",
    "Payment",
    "Governance",
    "Duration",
    "Other",
}


def _make_example(
    contract_id: str,
    question_name: str,
    spans: list[str],
    source_file: str = "contract.pdf",
    context: str = "Full contract text goes here.",
) -> dict:
    """Build a minimal CUAD-shaped example dict."""
    return {
        "id": contract_id,
        "title": source_file,
        "context": context,
        "question_name": question_name,
        "question": question_name,
        "answers": {"text": spans, "answer_start": [0] * len(spans)},
    }


def _mock_dataset(examples: list[dict]):
    """Return a plain list that quacks enough like a HuggingFace Dataset."""
    return examples


# ---------------------------------------------------------------------------
# _map_clause_type
# ---------------------------------------------------------------------------

class TestMapClauseType:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Termination For Convenience", "Termination"),
            ("Termination For Cause", "Termination"),
            ("Change of Control", "Termination"),
            ("Cap on Liability", "Liability"),
            ("Limitation of Liability", "Liability"),
            ("Indemnification", "Liability"),
            ("IP Ownership", "IP Rights"),
            ("License Grant", "IP Rights"),
            ("Non-Compete", "IP Rights"),
            ("Confidentiality", "Confidentiality"),
            ("Non-Disclosure", "Confidentiality"),
            ("Price Restriction", "Payment"),
            ("Audit Rights", "Payment"),
            ("Minimum Commitment", "Payment"),
            ("Governing Law", "Governance"),
            ("Jurisdiction", "Governance"),
            ("Dispute Resolution", "Governance"),
            ("Renewal Term", "Duration"),
            ("Notice Period", "Duration"),
            ("Expiration Date", "Duration"),
            ("Something Totally Unknown Clause XYZ", "Other"),
        ],
    )
    def test_known_clause_types(self, raw: str, expected: str) -> None:
        assert _map_clause_type(raw) == expected

    def test_case_insensitive(self) -> None:
        assert _map_clause_type("GOVERNING LAW") == "Governance"
        assert _map_clause_type("governing law") == "Governance"

    def test_all_returned_values_are_valid_categories(self) -> None:
        samples = [
            "Termination For Convenience", "Cap on Liability", "IP Ownership",
            "Confidentiality", "Audit Rights", "Governing Law", "Renewal Term",
            "Some Unknown Thing",
        ]
        for s in samples:
            assert _map_clause_type(s) in _ALL_CATEGORIES

    def test_unknown_returns_other(self) -> None:
        assert _map_clause_type("") == "Other"
        assert _map_clause_type("Foobar clause") == "Other"


# ---------------------------------------------------------------------------
# load_cuad_dataset — with mocked HuggingFace call
# ---------------------------------------------------------------------------

class TestLoadCuadDataset:
    @patch("datasets.load_dataset")
    def test_returns_dataframe_with_correct_columns(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Governing Law", ["This agreement is governed by NY law."]),
            _make_example("c2", "Termination For Convenience", ["Either party may terminate."]),
        ])
        df = load_cuad_dataset()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"contract_id", "clause_text", "clause_type", "source_file", "category"}

    @patch("datasets.load_dataset")
    def test_each_span_becomes_one_row(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Indemnification", ["Span A.", "Span B.", "Span C."]),
        ])
        df = load_cuad_dataset()
        assert len(df) == 3
        assert list(df["clause_text"]) == ["Span A.", "Span B.", "Span C."]

    @patch("datasets.load_dataset")
    def test_category_column_populated(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Governing Law", ["NY law applies."]),
            _make_example("c2", "Termination For Convenience", ["30-day notice."]),
        ])
        df = load_cuad_dataset()
        assert df.loc[df["clause_type"] == "Governing Law", "category"].iloc[0] == "Governance"
        assert df.loc[df["clause_type"] == "Termination For Convenience", "category"].iloc[0] == "Termination"

    @patch("datasets.load_dataset")
    def test_empty_spans_are_skipped(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Indemnification", ["", "  ", "Valid span."]),
        ])
        df = load_cuad_dataset()
        assert len(df) == 1
        assert df.iloc[0]["clause_text"] == "Valid span."

    @patch("datasets.load_dataset")
    def test_all_empty_spans_raises_runtime_error(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Indemnification", ["", "  "]),
        ])
        with pytest.raises(RuntimeError, match="zero clause records"):
            load_cuad_dataset()

    @patch("datasets.load_dataset")
    def test_no_examples_raises_runtime_error(self, mock_ld: MagicMock) -> None:
        mock_ld.return_value = _mock_dataset([])
        with pytest.raises(RuntimeError, match="zero clause records"):
            load_cuad_dataset()

    @patch("datasets.load_dataset", side_effect=Exception("network error"))
    def test_download_failure_raises_runtime_error(self, mock_ld: MagicMock) -> None:
        with pytest.raises(RuntimeError, match="Failed to download"):
            load_cuad_dataset()

    def test_missing_datasets_package_raises_import_error(self) -> None:
        import builtins
        real_import = builtins.__import__

        def _block_datasets(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_datasets):
            # Re-import to force the ImportError path
            import importlib
            import app.services.data_loader as mod
            importlib.reload(mod)
            with pytest.raises(ImportError, match="datasets"):
                mod.load_cuad_dataset()
            importlib.reload(mod)  # restore for other tests


# ---------------------------------------------------------------------------
# save_processed_data
# ---------------------------------------------------------------------------

class TestSaveProcessedData:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "contract_id": ["c1", "c2"],
                "clause_text": ["Tenant shall pay rent monthly.", "IP belongs to licensor."],
                "clause_type": ["Payment Terms", "IP Ownership"],
                "source_file": ["a.pdf", "b.pdf"],
                "category": ["Payment", "IP Rights"],
            }
        )

    def test_creates_csv_file(self, tmp_path: Path) -> None:
        df = self._sample_df()
        out = save_processed_data(df, output_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".csv"

    def test_csv_round_trips_correctly(self, tmp_path: Path) -> None:
        df = self._sample_df()
        out = save_processed_data(df, output_dir=tmp_path)
        reloaded = pd.read_csv(out)
        assert len(reloaded) == len(df)
        assert list(reloaded.columns) == list(df.columns)
        assert list(reloaded["clause_type"]) == list(df["clause_type"])

    def test_custom_filename(self, tmp_path: Path) -> None:
        df = self._sample_df()
        out = save_processed_data(df, output_dir=tmp_path, filename="custom.csv")
        assert out.name == "custom.csv"

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        save_processed_data(self._sample_df(), output_dir=nested)
        assert nested.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        out = save_processed_data(self._sample_df(), output_dir=tmp_path)
        assert isinstance(out, Path)


# ---------------------------------------------------------------------------
# print_stats (smoke test — just check it doesn't raise)
# ---------------------------------------------------------------------------

class TestPrintStats:
    @patch("datasets.load_dataset")
    def test_runs_without_error(self, mock_ld: MagicMock, capsys) -> None:
        mock_ld.return_value = _mock_dataset([
            _make_example("c1", "Governing Law", ["NY law."]),
            _make_example("c2", "Termination For Convenience", ["30 days."]),
            _make_example("c3", "Unknown XYZ", ["Some text."]),
        ])
        df = load_cuad_dataset()
        print_stats(df)
        captured = capsys.readouterr()
        assert "Total clauses" in captured.out
        assert "Distribution by category" in captured.out

    def test_all_categories_appear_when_present(self, capsys) -> None:
        rows = []
        for cat in _ALL_CATEGORIES:
            rows.append({"contract_id": "c1", "clause_text": "text", "clause_type": cat, "source_file": "f", "category": cat})
        df = pd.DataFrame(rows)
        print_stats(df)
        captured = capsys.readouterr()
        for cat in _ALL_CATEGORIES:
            assert cat in captured.out
