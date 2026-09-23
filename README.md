# Polyglot Support

An explainable multilingual customer-support assistant. It detects the customer's language, routes common support intents, grounds answers in local support documentation, and shows the words and sources behind each decision.

## What works

- English, German, French, Spanish, and Bengali response templates
- Password reset, order tracking/cancellation, subscription cancellation, and 2FA recovery intents
- Confidence-aware fallback and human-handoff messaging
- Local TXT, Markdown, and PDF knowledge-base search
- Inspectable intent evidence and cited knowledge files
- FastAPI API with validation, health check, OpenAPI docs, and CORS
- Responsive Angular support console
- Optional BERT, multilingual embeddings, Chroma, LIME/IG, and Airflow experiments

The default route is deliberately lightweight and deterministic. It starts without downloading an ML model. The original experimental ML files remain available under `backend/intent_classifier`, `backend/app/services`, `backend/notebooks`, and `airflow` for future evaluation—not as unverified production dependencies.

## Run locally

### API

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

### Web app

```powershell
cd multilingual-support-fronend
npm install
npm start -- --port 4300
```

Open `http://127.0.0.1:4300`.

### Docker

```powershell
docker compose up --build
```

The web app is exposed on port 4300 and the API on port 8000.

## API example

```http
POST /chat
Content-Type: application/json

{
  "message": "Wo ist meine Bestellung?",
  "explain_method": "lime"
}
```

The response includes the detected language, normalized intent, confidence, localized answer, resolution route, knowledge sources, and token-level evidence. `explain_method` enables the lightweight explanation output; both accepted values currently use the baseline classifier's exact feature contributions.

## Tests

```powershell
cd backend
pytest -q

cd ..\multilingual-support-fronend
npm run build
```

## Architecture

```text
Customer message
  -> language detection
  -> transparent intent scoring
       -> confident: localized support answer + related knowledge sources
       -> uncertain: lexical knowledge lookup or human-handoff message
  -> explanation and source metadata
  -> Angular support console
```


