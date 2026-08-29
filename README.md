# Lead Scoring B2B — MLOps

**Which leads have the highest conversion probability and should be prioritized by the sales team?**

End-to-end ML pipeline that scores B2B leads by conversion probability, served via a FastAPI endpoint and containerized with Docker.

## Key Results

| Metric | Value |
|--------|-------|
| Model | LightGBM Classifier |
| AUC-ROC | **0.83** |
| Deploy | FastAPI + Docker |
| Output | Conversion probability + priority tier |

## Stack

`Python` · `LightGBM` · `Scikit-Learn` · `FastAPI` · `Docker` · `Pandas` · `Seaborn`

## How It Works

1. **EDA** — Analyze conversion patterns, lead sources, and segment behavior
2. **Feature Engineering** — Transform raw CRM data into model-ready features
3. **Training** — LightGBM classifier with threshold optimization
4. **Scoring** — Each lead gets a 0–100 score mapped to priority tiers (high/medium/low)
5. **API** — FastAPI endpoint for real-time inference
6. **Container** — Dockerized for deployment

## API Example

**Request:**
```json
{
  "company_size": "medium",
  "segment": "technology",
  "lead_source": "paid_media",
  "number_of_interactions": 5,
  "days_since_first_contact": 12,
  "has_requested_demo": 1
}
```

**Response:**
```json
{
  "conversion_probability": 0.78,
  "lead_score": 78,
  "priority": "high"
}
```

## Project Structure

```
├── analise_b2b.ipynb            # EDA + model training
├── api_scoring.py               # FastAPI inference endpoint
├── pipeline_retreino.py         # Retraining pipeline
├── gerador_crm.py               # Synthetic CRM data generator
├── dashboard_comercial.html     # Commercial dashboard
├── docs/
│   └── MANUAL_OPERACIONAL.md    # Full technical manual (Portuguese)
├── DockerFile
└── requirements.txt
```

## Model Artifacts

The trained model is **not versioned**. `modelo_lead_scoring.pkl`, `features_modelo.pkl` and `threshold_otimo.pkl` are build outputs, regenerated end-to-end from source — the CRM data is synthetic and seeded (`np.random.seed(42)`), so the pipeline is fully reproducible from a clean clone.

## How to Run

```bash
git clone https://github.com/guilhermehrsilva/lead-scoring-b2b-mlops.git
cd lead-scoring-b2b-mlops
pip install -r requirements.txt
```

Then run the three steps in order — the API will not start until the artifacts exist:

```bash
python gerador_crm.py        # 1. generate synthetic CRM data
python pipeline_retreino.py  # 2. train and emit the three .pkl artifacts
uvicorn api_scoring:app --reload   # 3. serve
```

### With Docker

```bash
docker build -t lead-scoring-b2b .
docker run -p 8000:8000 lead-scoring-b2b
```

API docs available at `http://localhost:8000/docs`

## License

MIT — see [LICENSE](LICENSE).
