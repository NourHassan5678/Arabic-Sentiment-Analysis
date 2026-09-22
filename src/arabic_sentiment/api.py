import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from arabic_sentiment.baseline_model import TransformerSentimentModel

# Instantiate model wrapper globally
model = TransformerSentimentModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (model loading) and shutdown events cleanly."""
    model_path = os.getenv("MODEL_PATH", None)
    try:
        model.load(local_path=model_path)
    except Exception as e:
        print(f"Warning: Could not load production model on startup: {e}")
        print("Ensure 'python src/arabic_sentiment/train_transformer.py' has run or MLflow is accessible.")
    yield


app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    # min_length=1 ensures empty strings ("") automatically trigger a 422 Unprocessable Entity error
    text: str = Field(..., min_length=1, description="Arabic e-commerce review text")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    if not model.model or not model.tokenizer:
        raise HTTPException(status_code=503, detail="Model server uninitialized or model not loaded.")

    try:
        return model.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arabic_sentiment.api:app", host="0.0.0.0", port=8000, reload=True)