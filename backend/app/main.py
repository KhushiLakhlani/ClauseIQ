from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, analysis, clustering, documents

app = FastAPI(
    title="InsightLens API",
    version="0.1.0",
    description="Legal Contract NLP Analytics — powered by the CUAD dataset",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["classification"])
app.include_router(clustering.router, prefix="/api/v1", tags=["clustering"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
