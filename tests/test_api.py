"""
Unit Tests for CargoLink FastAPI Service Endpoints
"""

import os
import sys
import pytest
from pydantic import ValidationError

# Ensure workspace root is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import (
    app,
    load_model,
    root,
    health_check,
    predict_sentiment,
    predict_batch,
    ReviewRequest,
    BatchReviewRequest,
    PredictionResponse,
    BatchPredictionResponse,
)


@pytest.fixture(scope="module", autouse=True)
def setup_api_model():
    """Ensure model is loaded before running API endpoint tests."""
    load_model()


class TestFastAPIEndpoints:
    """Test suite for FastAPI endpoints and Pydantic schemas."""

    def test_root_endpoint(self):
        res = root()
        assert isinstance(res, dict)
        assert res["status"] == "online"
        assert res["service"] == "CargoLink Review Sentiment Analysis API"
        assert "endpoints" in res
        assert "predict" in res["endpoints"]

    def test_health_endpoint(self):
        res = health_check()
        assert isinstance(res, dict)
        assert res["status"] == "healthy"
        assert res["ready_for_inference"] is True
        assert "model" in res

    def test_predict_positive_sentiment(self):
        req = ReviewRequest(review_text="Package arrived within 24 hours, super fast delivery!")
        res = predict_sentiment(req)

        assert isinstance(res, PredictionResponse)
        assert res.sentiment == "Positive"
        assert res.label == 1
        assert res.confidence > 0.8
        assert res.probability_positive > 0.8
        assert res.probability_negative < 0.2

    def test_predict_negative_sentiment(self):
        req = ReviewRequest(review_text="Delayed shipment by 5 days, box was crushed and rider rude.")
        res = predict_sentiment(req)

        assert isinstance(res, PredictionResponse)
        assert res.sentiment == "Negative"
        assert res.label == 0
        assert res.confidence > 0.5
        assert res.probability_negative > 0.5

    def test_predict_empty_text_validation(self):
        """Pydantic min_length validation should prevent empty reviews."""
        with pytest.raises(ValidationError):
            ReviewRequest(review_text="")

    def test_predict_batch_endpoint(self):
        req = BatchReviewRequest(reviews=[
            "Flawless packaging and very fast dispatch.",
            "Worst courier ever, lost parcel in transit."
        ])
        res = predict_batch(req)

        assert isinstance(res, BatchPredictionResponse)
        assert res.total_reviews == 2
        assert res.positive_count == 1
        assert res.negative_count == 1
        assert len(res.results) == 2
        assert res.results[0].sentiment == "Positive"
        assert res.results[1].sentiment == "Negative"

