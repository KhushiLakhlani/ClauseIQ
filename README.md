# ClauseIQ

> Legal Contract NLP Analytics Platform — powered by the CUAD dataset.

## Project Structure

```
clauseiq/
├── backend/      # FastAPI — ML inference, REST API
└── frontend/     # React + Vite + Tailwind CSS
```

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
