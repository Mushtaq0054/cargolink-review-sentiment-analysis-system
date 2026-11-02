"""
CargoLink Review Sentiment Analysis System
Task 2: Text Cleaning Pipeline & Baseline Logistic Regression Model
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional

import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUTPUT_FILE = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")

# Ensure NLTK stopwords are accessible
try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))

# Preserve key negation terms essential for sentiment semantics (e.g. "not delayed", "no issues")
NEGATION_WORDS = {"not", "no", "nor", "never", "neither", "barely", "hardly", "scarcely"}
CUSTOM_STOP_WORDS = _STOP_WORDS - NEGATION_WORDS


def clean_text(text: str) -> str:
    """
    Clean and preprocess raw review text:
    1. Converts text to lowercase.
    2. Strips URL patterns and non-alphabetical punctuation.
    3. Tokenizes and filters out standard stopwords (retaining negation indicators).
    4. Drops single-character noise.

    Args:
        text (str): Raw input customer review.

    Returns:
        str: Cleaned, space-separated normalized token string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URL links if present
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove non-alphabetical characters / punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize and remove stopwords
    tokens = [
        word for word in text.split()
        if word not in CUSTOM_STOP_WORDS and len(word) > 1
    ]

    return " ".join(tokens)


def build_baseline_pipeline(
    max_features: int = 5000,
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a reproducible scikit-learn Pipeline combining
    TF-IDF Vectorization and Logistic Regression.

    Args:
        max_features (int): Vocabulary limit for TF-IDF.
        C (float): Inverse regularization strength for Logistic Regression.
        random_state (int): Seed for reproducibility.

    Returns:
        Pipeline: Untrained scikit-learn Pipeline object.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=max_features,
                    sublinear_tf=True,
                    ngram_range=(1, 1),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate(
    train_path: str = TRAIN_FILE,
    test_path: str = TEST_FILE,
    save_model_path: Optional[str] = MODEL_OUTPUT_FILE,
) -> Tuple[Pipeline, dict]:
    """
    Loads train and test sets, trains the baseline pipeline, evaluates metrics,
    and optionally persists the trained pipeline artifact.

    Args:
        train_path (str): Path to training CSV (columns: review_text, label).
        test_path (str): Path to test CSV (columns: review_text, label).
        save_model_path (str, optional): Target path to persist joblib model.

    Returns:
        Tuple[Pipeline, dict]: Trained pipeline and dictionary of evaluation metrics.
    """
    print(f"[INFO] Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path).dropna(subset=["review_text", "label"])

    print(f"[INFO] Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path).dropna(subset=["review_text", "label"])

    X_train = train_df["review_text"]
    y_train = train_df["label"].astype(int)

    X_test = test_df["review_text"]
    y_test = test_df["label"].astype(int)

    print(f"  Training samples: {len(X_train):,} (Pos: {(y_train == 1).sum():,}, Neg: {(y_train == 0).sum():,})")
    print(f"  Testing samples:  {len(X_test):,} (Pos: {(y_test == 1).sum():,}, Neg: {(y_test == 0).sum():,})")

    # Build and fit pipeline
    print("\n[INFO] Building and training TF-IDF + LogisticRegression baseline pipeline...")
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    print("[INFO] Evaluating baseline model on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"])

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    print("\n" + "=" * 50)
    print("[TASK 2] BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Baseline Accuracy: {acc * 100:.2f}%  (Mentor Target: > 70.00%)")
    print(f"Precision:         {prec * 100:.2f}%")
    print(f"Recall:            {rec * 100:.2f}%")
    print(f"F1 Score:          {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  TN (True Negatives):  {cm[0, 0]} | FP (False Positives): {cm[0, 1]}")
    print(f"  FN (False Negatives): {cm[1, 0]} | TP (True Positives):  {cm[1, 1]}")
    print("\nClassification Report:\n", report)
    print("=" * 50)

    # Save model artifact
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(pipeline, save_model_path)
        print(f"\n[SUCCESS] Baseline model successfully saved to: {save_model_path}")

    return pipeline, metrics


def test_sample_predictions(pipeline: Pipeline):
    """Demonstrate inference on sample real-world logistics reviews."""
    sample_reviews = [
        "Package arrived within 24 hours in flawless packaging. Super fast delivery!",
        "Delayed shipment by 5 days, box was completely crushed and rider was rude.",
        "Smooth parcel dispatch, tracking was accurate and courier was professional.",
        "Terrible experience. Package lost in transit and customer service did not help.",
    ]

    print("\n--- SAMPLE REAL-WORLD INFERENCE DEMO ---")
    predictions = pipeline.predict(sample_reviews)
    probabilities = pipeline.predict_proba(sample_reviews)

    for review, pred, prob in zip(sample_reviews, predictions, probabilities):
        label_str = "POSITIVE (+)" if pred == 1 else "NEGATIVE (-)"
        confidence = prob[pred] * 100
        print(f"Review: \"{review}\"")
        print(f"  -> Predicted: {label_str} (Confidence: {confidence:.2f}%)\n")


def main():
    pipeline, _ = train_and_evaluate()
    test_sample_predictions(pipeline)


if __name__ == "__main__":
    main()
