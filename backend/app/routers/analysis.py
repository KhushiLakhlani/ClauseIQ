"""Clause classification and explainability endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.classifier import ClauseCategory, ClauseClassifier, ClausePrediction
from app.services.explainer import ClauseExplainer

router = APIRouter()
_classifier = ClauseClassifier()
# Load model eagerly so we can pass correct class order to explainer
try:
    _classifier.load()
    _class_names = _classifier._model.classes_.tolist()
except FileNotFoundError:
    _class_names = None

_explainer = ClauseExplainer(class_names=_class_names)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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

class ExplainRequest(BaseModel):
    text: str
    num_features: int = 10

class FeatureContribution(BaseModel):
    feature: str
    weight: float
    positive: bool

class ExplainResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    top_features: list[FeatureContribution]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/classify", response_model=ClassifyResponse)
async def classify_clause(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify a single contract clause into one of the seven CUAD categories."""
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


@router.post("/explain", response_model=ExplainResponse)
async def explain_clause(payload: ExplainRequest) -> ExplainResponse:
    """Explain why the model classified a clause the way it did using LIME."""
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    explanation = _explainer.explain(
        clause_text=payload.text,
        predict_fn=_classifier.predict_proba,
    )

    return ExplainResponse(
        text=explanation.clause_text,
        predicted_category=explanation.predicted_label,
        confidence=explanation.confidence,
        top_features=[
            FeatureContribution(
                feature=f.feature,
                weight=f.weight,
                positive=f.positive,
            )
            for f in explanation.top_features
        ],
    )