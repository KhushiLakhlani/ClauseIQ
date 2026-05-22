"""Clause classification endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.classifier import ClauseCategory, ClauseClassifier, ClausePrediction

router = APIRouter()
_classifier = ClauseClassifier()


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    text: str
    predicted_category: ClauseCategory
    confidence: float
    probabilities: dict[str, float]


class BatchClassifyRequest(BaseModel):
    clauses: list[str]


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


@router.post("/classify", response_model=ClassifyResponse)
async def classify_clause(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify a single contract clause into one of the eight CUAD categories."""
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    pred: ClausePrediction = _classifier.predict(payload.text)
    return ClassifyResponse(
        text=pred.text,
        predicted_category=pred.predicted_category,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
    )


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_clauses_batch(payload: BatchClassifyRequest) -> BatchClassifyResponse:
    """Classify a list of contract clauses in a single request."""
    if not payload.clauses:
        raise HTTPException(status_code=422, detail="clauses list must not be empty")
    preds = _classifier.predict_batch(payload.clauses)
    return BatchClassifyResponse(
        results=[
            ClassifyResponse(
                text=p.text,
                predicted_category=p.predicted_category,
                confidence=p.confidence,
                probabilities=p.probabilities,
            )
            for p in preds
        ]
    )
