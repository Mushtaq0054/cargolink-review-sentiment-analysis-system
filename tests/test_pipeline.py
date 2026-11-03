"""
Unit Tests for CargoLink Text Cleaning and Model Pipeline
"""

import os
import sys
import pytest
import joblib

# Ensure src is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import clean_text

MODELS_DIR = os.path.join(BASE_DIR, "models")
TUNED_MODEL_FILE = os.path.join(MODELS_DIR, "tuned_model.joblib")
BASELINE_MODEL_FILE = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")


class TestTextCleaner:
    """Test suite for the NLP text preprocessing pipeline."""

    def test_lowercase_conversion(self):
        raw = "SUPER FAST DELIVERY"
        cleaned = clean_text(raw)
        assert cleaned == cleaned.lower()
        assert "super" in cleaned
        assert "delivery" in cleaned

    def test_punctuation_and_number_removal(self):
        raw = "Package arrived in 3 days! Box was damaged & crushed..."
        cleaned = clean_text(raw)
        assert "!" not in cleaned
        assert "&" not in cleaned
        assert "..." not in cleaned
        assert "3" not in cleaned
        assert "package arrived" in cleaned or "package" in cleaned

    def test_stopwords_removal(self):
        raw = "This is a very good delivery of the package"
        cleaned = clean_text(raw)
        tokens = cleaned.split()
        assert "is" not in tokens
        assert "the" not in tokens
        assert "of" not in tokens

    def test_negation_words_preservation(self):
        """Crucial test: Verify negation words like 'not' and 'no' are retained."""
        raw = "The parcel was not delayed and there were no issues"
        cleaned = clean_text(raw)
        tokens = cleaned.split()
        assert "not" in tokens
        assert "no" in tokens
        assert "delayed" in tokens

    def test_empty_and_non_string_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text(12345) == ""


class TestModelArtifacts:
    """Test suite for serialized model pipeline artifacts."""

    def test_model_files_exist(self):
        assert os.path.exists(TUNED_MODEL_FILE), "Tuned model artifact missing!"
        assert os.path.exists(BASELINE_MODEL_FILE), "Baseline model artifact missing!"

    def test_tuned_model_inference_positive(self):
        model = joblib.load(TUNED_MODEL_FILE)
        review = ["Package arrived within 24 hours in flawless packaging. Super fast delivery!"]
        pred = model.predict(review)[0]
        prob = model.predict_proba(review)[0]

        assert pred == 1, "Expected positive sentiment (1)"
        assert prob[1] > 0.5, "Expected positive probability > 0.5"
        assert abs(sum(prob) - 1.0) < 1e-4, "Probabilities must sum to 1"

    def test_tuned_model_inference_negative(self):
        model = joblib.load(TUNED_MODEL_FILE)
        review = ["Delayed shipment by 5 days, box was completely crushed and rider was rude."]
        pred = model.predict(review)[0]
        prob = model.predict_proba(review)[0]

        assert pred == 0, "Expected negative sentiment (0)"
        assert prob[0] > 0.5, "Expected negative probability > 0.5"

    def test_batch_prediction_shape(self):
        model = joblib.load(TUNED_MODEL_FILE)
        reviews = [
            "Excellent packaging and speedy courier delivery.",
            "Horrible experience, item lost in transit.",
            "Fast dispatch, arrived on time."
        ]
        preds = model.predict(reviews)
        probs = model.predict_proba(reviews)

        assert len(preds) == 3
        assert probs.shape == (3, 2)

