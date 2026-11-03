"""
CargoLink Review Sentiment Analysis System
Task 3: Feature Enhancement (N-grams) and Hyperparameter Tuning (SVM & XGBoost)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import clean_text, TRAIN_FILE, TEST_FILE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TUNED_MODEL_FILE = os.path.join(MODELS_DIR, "tuned_model.joblib")


def evaluate_model_performance(name: str, model: Any, X_test, y_test) -> Dict[str, Any]:
    """Helper to evaluate and format model metrics on the test set."""
    preds = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X_test)
    else:
        probs = preds

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="binary")
    rec = recall_score(y_test, preds, average="binary")
    f1 = f1_score(y_test, preds, average="binary")
    roc = roc_auc_score(y_test, probs)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": roc,
        "Predictions": preds,
        "Probabilities": probs,
    }


def run_experiments_and_tuning() -> Tuple[Pipeline, pd.DataFrame]:
    """
    Executes Task 3 experiments:
    1. Loads train and test datasets.
    2. Builds n-gram (1, 2) TF-IDF vectorizer.
    3. Trains Baseline LR (Unigram), Tuned LR (Bigram), Linear SVM (Bigram), and XGBoost (Bigram).
    4. Performs GridSearchCV to tune SVM hyperparameters.
    5. Saves best performing model.
    """
    print("[INFO] Loading datasets...")
    train_df = pd.read_csv(TRAIN_FILE).dropna(subset=["review_text", "label"])
    test_df = pd.read_csv(TEST_FILE).dropna(subset=["review_text", "label"])

    X_train, y_train = train_df["review_text"], train_df["label"].astype(int)
    X_test, y_test = test_df["review_text"], test_df["label"].astype(int)

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples:  {len(X_test):,}")

    # ==========================================
    # 1. Feature Engineering: Unigrams vs Bigrams
    # ==========================================
    print("\n[INFO] Initializing TF-IDF Vectorizer with N-grams (1, 2)...")
    tfidf_bigram = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    results = []

    # ------------------------------------------
    # Model 1: Baseline Logistic Regression (Unigram)
    # ------------------------------------------
    print("\n[1/4] Evaluating Baseline Logistic Regression (Unigrams)...")
    pipe_baseline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 1), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipe_baseline.fit(X_train, y_train)
    res_baseline = evaluate_model_performance("Baseline Logistic Regression (1-gram)", pipe_baseline, X_test, y_test)
    results.append(res_baseline)

    # ------------------------------------------
    # Model 2: Logistic Regression with Bigrams
    # ------------------------------------------
    print("[2/4] Training Logistic Regression with Bigrams (1-2 grams)...")
    pipe_lr_bigram = Pipeline([
        ("tfidf", tfidf_bigram),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
    ])
    pipe_lr_bigram.fit(X_train, y_train)
    res_lr_bigram = evaluate_model_performance("Logistic Regression (Bigrams, C=5.0)", pipe_lr_bigram, X_test, y_test)
    results.append(res_lr_bigram)

    # ------------------------------------------
    # Model 3: XGBoost Classifier (Bigrams)
    # ------------------------------------------
    print("[3/4] Training XGBoost Classifier with Bigrams...")
    pipe_xgb = Pipeline([
        ("tfidf", tfidf_bigram),
        ("clf", XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss",
        )),
    ])
    pipe_xgb.fit(X_train, y_train)
    res_xgb = evaluate_model_performance("XGBoost (Bigrams, depth=6, lr=0.1)", pipe_xgb, X_test, y_test)
    results.append(res_xgb)

    # ------------------------------------------
    # Model 4: GridSearchCV for Support Vector Machine (SVM)
    # ------------------------------------------
    print("\n[4/4] Setting up GridSearchCV for Support Vector Machine (LinearSVC)...")
    param_grid = {
        "clf__estimator__C": [0.1, 0.5, 1.0, 2.0],
    }

    base_svm_pipeline = Pipeline([
        ("tfidf", tfidf_bigram),
        ("clf", CalibratedClassifierCV(estimator=LinearSVC(max_iter=2000, random_state=42))),
    ])

    grid_search = GridSearchCV(
        estimator=base_svm_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )

    print("  Running Grid Search across hyperparameter space...")
    grid_search.fit(X_train, y_train)

    best_svm_pipe = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"  Best Hyperparameters: {best_params}")
    print(f"  Best 3-Fold CV Accuracy: {best_cv_score * 100:.2f}%")

    res_svm = evaluate_model_performance(f"Tuned Linear SVM (Bigrams, Best C={best_params['clf__estimator__C']})", best_svm_pipe, X_test, y_test)
    results.append(res_svm)

    # ==========================================
    # Comparison Summary Table
    # ==========================================
    summary_df = pd.DataFrame([
        {
            "Model Name": r["Model"],
            "Accuracy": f"{r['Accuracy'] * 100:.2f}%",
            "Precision": f"{r['Precision'] * 100:.2f}%",
            "Recall": f"{r['Recall'] * 100:.2f}%",
            "F1-Score": f"{r['F1_Score'] * 100:.2f}%",
            "ROC-AUC": f"{r['ROC_AUC']:.4f}",
        }
        for r in results
    ])

    print("\n" + "=" * 80)
    print("[TASK 3] COMPREHENSIVE MODEL COMPARISON ON TEST SET")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Final Mentor Check: Improved accuracy > baseline
    baseline_acc = res_baseline["Accuracy"]
    best_acc = res_svm["Accuracy"]
    improvement = (best_acc - baseline_acc) * 100

    print(f"\n[VERIFICATION]")
    print(f"  Baseline Accuracy: {baseline_acc * 100:.2f}%")
    print(f"  Tuned SVM Accuracy: {best_acc * 100:.2f}%")
    print(f"  Accuracy Improvement: +{improvement:.2f}%")
    print(f"  Status: {'PASSED (Improved > Baseline)' if best_acc > baseline_acc else 'FAILED'}")

    # Save Best Model Artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_svm_pipe, TUNED_MODEL_FILE)
    print(f"\n[SUCCESS] Best tuned model saved to: {TUNED_MODEL_FILE}")

    # Print sample predictions on real-world logistics edge cases
    test_edge_cases(best_svm_pipe)

    return best_svm_pipe, summary_df


def test_edge_cases(model: Pipeline):
    """Demonstrate inference on challenging logistics reviews where bigrams excel."""
    reviews = [
        "Package arrived within 24 hours in flawless packaging. Super fast delivery!",
        "Delayed shipment by 5 days, box was completely crushed and rider was rude.",
        "The courier was not late, dispatch was smooth and package was intact.",
        "Worst delivery ever! Terrible packaging and rider never showed up.",
        "Tracking status was updated in real time and delivery was fast.",
    ]

    print("\n--- TUNED MODEL EDGE-CASE PREDICTIONS (BIGRAMS IMPACT) ---")
    preds = model.predict(reviews)
    probs = model.predict_proba(reviews)

    for review, pred, prob in zip(reviews, preds, probs):
        sentiment = "POSITIVE (+)" if pred == 1 else "NEGATIVE (-)"
        confidence = prob[pred] * 100
        print(f"Review: \"{review}\"")
        print(f"  -> Predicted: {sentiment} (Confidence: {confidence:.2f}%)\n")


def main():
    run_experiments_and_tuning()


if __name__ == "__main__":
    main()
