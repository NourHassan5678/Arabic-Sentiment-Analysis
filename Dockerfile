FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY models/production_model ./models/production_model

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "arabic_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]