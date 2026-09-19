FROM python:3.11-slim
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration and install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package globally in the container
RUN pip install --no-cache-dir -e .

# Copy the trained model
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "arabic_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]