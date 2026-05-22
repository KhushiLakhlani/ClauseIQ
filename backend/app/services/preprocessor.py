"""Legal text preprocessing pipeline for CUAD contract analysis."""

import re
from dataclasses import dataclass, field


@dataclass
class PreprocessedDocument:
    raw_text: str
    cleaned_text: str
    sentences: list[str]
    tokens: list[str]
    char_count: int
    word_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class TextPreprocessor:
    """Cleans and segments raw legal contract text for downstream NLP tasks.

    Handles common legal document artifacts: exhibit headers, page numbers,
    ALL-CAPS section titles, and boilerplate signature blocks.
    """

    # Section headers commonly found in CUAD contracts
    _SECTION_HEADER_RE = re.compile(
        r"^(ARTICLE|SECTION|EXHIBIT|SCHEDULE|ANNEX)\s+[\dIVXA-Z]+[.\s]",
        re.IGNORECASE | re.MULTILINE,
    )
    _PAGE_NUMBER_RE = re.compile(r"\n\s*-?\s*\d+\s*-?\s*\n")
    _EXCESS_WHITESPACE_RE = re.compile(r"\s{3,}")

    def preprocess(self, text: str, doc_id: str = "") -> PreprocessedDocument:
        """Clean *text* and return a structured document ready for inference.

        Args:
            text: Raw contract text extracted from PDF or plain file.
            doc_id: Optional identifier carried through to metadata.

        Returns:
            PreprocessedDocument with cleaned text, sentence list, and stats.
        """
        cleaned = self._clean(text)
        sentences = self._sentence_split(cleaned)
        tokens = self._tokenize(cleaned)
        return PreprocessedDocument(
            raw_text=text,
            cleaned_text=cleaned,
            sentences=sentences,
            tokens=tokens,
            char_count=len(cleaned),
            word_count=len(tokens),
            metadata={"doc_id": doc_id} if doc_id else {},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        text = self._PAGE_NUMBER_RE.sub("\n", text)
        text = self._EXCESS_WHITESPACE_RE.sub(" ", text)
        text = text.strip()
        return text

    def _sentence_split(self, text: str) -> list[str]:
        """Naive sentence splitter — replace with spaCy sentencizer in prod."""
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in raw if s.strip()]

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase whitespace tokenizer — replace with spaCy tokenizer in prod."""
        return re.findall(r"\b\w+\b", text.lower())
