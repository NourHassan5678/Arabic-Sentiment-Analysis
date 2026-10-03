FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU version of PyTorch to keep image size small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy only project config and code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install serving dependencies
RUN pip install --no-cache-dir ".[serving]"

EXPOSE 8000

CMD ["uvicorn", "src.arabic_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]