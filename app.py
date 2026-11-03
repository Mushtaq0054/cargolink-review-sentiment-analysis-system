"""
CargoLink Review Sentiment Analysis System
Task 4: FastAPI Service for Real-Time Sentiment Prediction
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src modules (text cleaning pipeline) are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import clean_text  # Required for unpickling custom preprocessor

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "tuned_model.joblib")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")

# Global reference for loaded model pipeline
model_pipeline = None
model_name = ""


def load_model():
    """Load model artifact into memory."""
    global model_pipeline, model_name

    if os.path.exists(TUNED_MODEL_PATH):
        print(f"[INFO] Loading Tuned Model pipeline from: {TUNED_MODEL_PATH}")
        model_pipeline = joblib.load(TUNED_MODEL_PATH)
        model_name = "Tuned Linear SVM (Bigrams, C=0.5)"
    elif os.path.exists(BASELINE_MODEL_PATH):
        print(f"[WARN] Tuned model not found. Falling back to Baseline from: {BASELINE_MODEL_PATH}")
        model_pipeline = joblib.load(BASELINE_MODEL_PATH)
        model_name = "Baseline Logistic Regression (1-grams)"
    else:
        raise FileNotFoundError(
            f"No model artifacts found in {MODELS_DIR}. Run training pipeline first."
        )

    print(f"[SUCCESS] {model_name} successfully loaded into memory!")
    return model_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event: Load model artifact upon server startup."""
    load_model()
    yield
    print("[INFO] Shutting down CargoLink Sentiment API service.")


# Initialize FastAPI app
app = FastAPI(
    title="CargoLink Review Sentiment Analysis API",
    description=(
        "Production microservice for CargoLink Logistics to analyze customer review sentiments "
        "on shipments, package handling, transit times, and courier performance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend / Streamlit cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=3,
        description="The raw customer review text to be analyzed for sentiment.",
        example="Package arrived within 24 hours in flawless condition. Super fast delivery!"
    )


class PredictionResponse(BaseModel):
    review_text: str
    sentiment: str = Field(..., description="Predicted sentiment classification: 'Positive' or 'Negative'")
    label: int = Field(..., description="Binary sentiment label: 1 for Positive, 0 for Negative")
    confidence: float = Field(..., description="Confidence score of the prediction (between 0.0 and 1.0)")
    probability_positive: float = Field(..., description="Calibrated probability of Positive sentiment")
    probability_negative: float = Field(..., description="Calibrated probability of Negative sentiment")
    model_version: str


class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_items=1,
        description="List of raw customer review strings.",
        example=[
            "Super fast shipping and safe delivery!",
            "Delayed shipment by 5 days, box crushed and courier was rude."
        ]
    )


class BatchPredictionResponse(BaseModel):
    total_reviews: int
    positive_count: int
    negative_count: int
    results: List[PredictionResponse]


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Service landing page and health verification."""
    return {
        "status": "online",
        "service": "CargoLink Review Sentiment Analysis API",
        "version": "1.0.0",
        "model_loaded": model_name,
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check endpoint for cloud uptime monitors."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded into memory."
        )
    return {
        "status": "healthy",
        "model": model_name,
        "ready_for_inference": True
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict sentiment for a single review"
)
def predict_sentiment(request: ReviewRequest):
    """
    Accepts customer review text as JSON and returns sentiment prediction,
    calibrated confidence, and class probabilities.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable."
        )

    text = request.review_text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty or solely whitespace."
        )

    try:
        # Prediction
        pred = int(model_pipeline.predict([text])[0])
        probs = model_pipeline.predict_proba([text])[0]

        prob_neg = float(probs[0])
        prob_pos = float(probs[1])

        sentiment_str = "Positive" if pred == 1 else "Negative"
        confidence_score = float(max(prob_pos, prob_neg))

        return PredictionResponse(
            review_text=text,
            sentiment=sentiment_str,
            label=pred,
            confidence=round(confidence_score, 4),
            probability_positive=round(prob_pos, 4),
            probability_negative=round(prob_neg, 4),
            model_version=model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict sentiment for multiple reviews in batch"
)
def predict_batch(request: BatchReviewRequest):
    """Accepts multiple customer reviews and returns array of sentiment predictions."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable."
        )

    reviews = [r.strip() for r in request.reviews if r.strip()]
    if not reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least one valid review string."
        )

    try:
        preds = model_pipeline.predict(reviews)
        probs = model_pipeline.predict_proba(reviews)

        results = []
        pos_cnt = 0
        neg_cnt = 0

        for r_text, pred, prob in zip(reviews, preds, probs):
            pred_int = int(pred)
            prob_neg = float(prob[0])
            prob_pos = float(prob[1])
            confidence = float(max(prob_pos, prob_neg))
            sent_str = "Positive" if pred_int == 1 else "Negative"

            if pred_int == 1:
                pos_cnt += 1
            else:
                neg_cnt += 1

            results.append(
                PredictionResponse(
                    review_text=r_text,
                    sentiment=sent_str,
                    label=pred_int,
                    confidence=round(confidence, 4),
                    probability_positive=round(prob_pos, 4),
                    probability_negative=round(prob_neg, 4),
                    model_version=model_name
                )
            )

        return BatchPredictionResponse(
            total_reviews=len(results),
            positive_count=pos_cnt,
            negative_count=neg_cnt,
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
