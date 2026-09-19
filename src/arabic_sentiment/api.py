from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from contextlib import asynccontextmanager

from arabic_sentiment.baseline_model import SentimentModel

# Global model instance
model = SentimentModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_path = os.getenv("MODEL_PATH", "models/baseline_model.joblib")
    try:
        model.load(model_path)
    except FileNotFoundError:
        print(f"Warning: Model not found at {model_path}. Please train it first.")
    yield
    # Cleanup on shutdown (if any)

app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    # Field min_length=1 ensures empty strings ("") automatically return a 422 error
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
    if not model.pipeline:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        result = model.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("arabic_sentiment.api:app", host="0.0.0.0", port=8000, reload=True)