---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/classify` | Classify a single clause |
| POST | `/api/v1/classify/batch` | Classify multiple clauses |
| POST | `/api/v1/explain` | LIME explanation for a clause |
| POST | `/api/v1/cluster` | Topic modeling on clause set |
| GET | `/api/v1/health` | Health check |

**Example request:**
```bash
curl -X POST https://clauseiq-sq4w.onrender.com/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Either party may terminate this agreement upon 30 days written notice."}'
```

---

## Run Locally

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train.py              # Train the model (takes ~10s)
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                  # Opens on http://localhost:5173
```

---

## Key Design Decisions

- **TF-IDF + Logistic Regression over transformers** — 942KB model vs 400MB+ BERT, with 90.5% accuracy and full LIME interpretability
- **7 categories from 41 CUAD types** — Grouped to avoid class imbalance; smallest class (Termination) has 499 samples
- **Balanced class weights** — Compensates for 6:1 ratio between largest and smallest categories
- **Macro F1 as primary metric** — Treats all categories equally regardless of size
- **Substring matching for category mapping** — More robust than exact match against dataset label drift

---

## Dataset

Built on the [CUAD dataset](https://www.atticusprojectai.org/cuad) — 510 commercial legal contracts with 13,000+ expert-annotated clause spans across 41 clause types, created by The Atticus Project.

---

## Author

**Khushi Lakhlani** — MS Information Systems, Northeastern University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/khushilakhlani)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/KhushiLakhlani)
