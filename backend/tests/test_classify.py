import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_classify_termination_clause(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/classify",
        json={"text": "Either party may terminate this agreement upon 30 days written notice."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"] == "Termination"
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classify_empty_text_returns_422(client: AsyncClient) -> None:
    response = await client.post("/api/v1/classify", json={"text": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_batch(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/classify/batch",
        json={
            "clauses": [
                "Either party may terminate this agreement with 30 days notice.",
                "All intellectual property created hereunder belongs to the Company.",
            ]
        },
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
