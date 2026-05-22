"""Topic modelling / clustering endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.clustering import TopicModeler, TopicModelSummary

router = APIRouter()


class ClusterRequest(BaseModel):
    clauses: list[str]
    n_topics: int = 8
    method: str = "nmf"


class TopicTerms(BaseModel):
    topic_id: int
    topic_label: str
    top_terms: list[str]


class ClusterAssignment(BaseModel):
    text: str
    topic_id: int
    topic_label: str
    top_terms: list[str]


class ClusterResponse(BaseModel):
    n_topics: int
    topics: list[TopicTerms]
    assignments: list[ClusterAssignment]


@router.post("/cluster", response_model=ClusterResponse)
async def cluster_clauses(payload: ClusterRequest) -> ClusterResponse:
    """Run topic modelling over a list of contract clauses."""
    if len(payload.clauses) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 clauses to cluster")
    if payload.method not in ("kmeans", "nmf"):
        raise HTTPException(status_code=422, detail="method must be 'kmeans' or 'nmf'")

    modeler = TopicModeler(
        n_topics=min(payload.n_topics, len(payload.clauses)),
        method=payload.method,
    )
    summary: TopicModelSummary = modeler.fit_transform(payload.clauses)

    topics = [
        TopicTerms(topic_id=i, topic_label=summary.topic_labels[i], top_terms=terms)
        for i, terms in enumerate(summary.topic_top_terms)
    ]
    assignments = [
        ClusterAssignment(
            text=a.text,
            topic_id=a.topic_id,
            topic_label=a.topic_label,
            top_terms=a.top_terms,
        )
        for a in summary.assignments
    ]
    return ClusterResponse(
        n_topics=summary.n_topics,
        topics=topics,
        assignments=assignments,
    )
