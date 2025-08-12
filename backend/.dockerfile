FROM python:3.11-slim
WORKDIR /app

# System deps (pdftools etc. only if you need them)
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app

# Speed up HF downloads (cached in the container filesystem)
ENV TRANSFORMERS_CACHE=/app/.hf_cache

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
