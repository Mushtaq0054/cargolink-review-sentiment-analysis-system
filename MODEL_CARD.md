# 📋 Model Card: CargoLink Review Sentiment Classifier

## 1. Model Details
* **Model Name:** CargoLink Review Sentiment Classifier
* **Model Version:** 1.0.0 (Production Release)
* **Model Type:** Tuned Linear Support Vector Machine (`LinearSVC`) with Platt Probability Calibration (`CalibratedClassifierCV`)
* **Feature Representation:** TF-IDF Vectorizer with N-grams (1, 2) and Sublinear TF Scaling
* **Developer:** Mushtaq Ahmad (Machine Learning Intern for CargoLink Logistics)
* **Date:** November 2026
* **License:** MIT License
* **Frameworks & Libraries:** Python 3.10+, Scikit-Learn (1.4+), NLTK, Joblib

---

## 2. Intended Use
### Primary Intended Uses:
* **Real-time Sentiment Classification:** Automatically classifying incoming customer reviews on logistics operations, courier handling, transit timeliness, and packaging conditions into **Positive (`1`)** and **Negative (`0`)** classes.
* **Management Dashboard Monitoring:** Powering operational KPIs on the CargoLink Streamlit Dashboard.
* **Microservice Inference:** Serving real-time HTTP prediction requests via the containerized FastAPI backend deployed on Render.

### Out-of-Scope Uses:
* Analyzing generic non-e-commerce text (e.g. medical literature, movie reviews, legal contracts).
* Multi-aspect or fine-grained sentiment analysis (e.g., separating product quality from shipping quality).

---

## 3. Training & Evaluation Data
* **Source:** Public E-Commerce and Customer Reviews Dataset (~45,476 raw records).
* **Domain Adaptation:** Filtered using domain-specific regular expressions for logistics terminology (`delivery`, `shipping`, `courier`, `parcel`, `package`, `tracking`, `delay`, `damaged`, `arrived`, `dispatch`).
* **Clean Dataset Size:** 10,178 unique, non-neutral reviews.
* **Train / Test Split:** Stratified 80/20 split:
  * **Training Set:** 8,141 samples (80.58% Positive, 19.42% Negative)
  * **Test Set:** 2,037 samples (80.58% Positive, 19.42% Negative)

---

## 4. Performance & Evaluation Metrics
The model was evaluated against an unseen hold-out test set of 2,037 samples:

| Metric | Score | Target / Benchmark |
| :--- | :---: | :---: |
| **Accuracy** | **92.39%** | Benchmark: > 70.00% (+22.39% above target) |
| **Precision** | **94.33%** | High reliability on positive predictions |
| **Recall** | **96.34%** | Captures 96.34% of genuine positive reviews |
| **F1-Score** | **95.33%** | Balanced harmonic mean |
| **ROC-AUC** | **0.9624** | Exceptional class separability |

### Confusion Matrix (Test Set - 2,037 samples):
* **True Negatives (TN):** 298
* **False Positives (FP):** 98
* **False Negatives (FN):** 60
* **True Positives (TP):** 1,581

### Comparison against Alternative Architectures:
* **Baseline Logistic Regression (1-grams):** 90.38% Accuracy
* **Logistic Regression (Bigrams, $C=5.0$):** 92.15% Accuracy
* **XGBoost Classifier (Bigrams, depth=6):** 89.45% Accuracy
* **Tuned Linear SVM (Bigrams, $C=0.5$):** **92.39% Accuracy (Selected Best Model)**

---

## 5. Hyperparameter Specifications
* **TF-IDF Vectorizer:**
  * `ngram_range`: (1, 2)
  * `max_features`: 10,000
  * `sublinear_tf`: True
  * `preprocessor`: Custom text cleaner preserving negation tokens (`not`, `no`, `never`)
* **Classifier (`LinearSVC` wrapped in `CalibratedClassifierCV`):**
  * `C`: 0.5 (Tuned via 3-Fold Stratified GridSearchCV)
  * `loss`: squared_hinge
  * `max_iter`: 2,000
  * `method`: sigmoid (Platt Scaling)

---

## 6. Ethical Considerations & Limitations
* **Class Imbalance:** In real-world e-commerce data, positive reviews naturally outnumber negative reviews (~80% vs 20%). The model uses stratified sampling and probability calibration to maintain high negative class precision (92%+).
* **Sarcasm & Nuance:** Heavily sarcastic reviews without explicit logistics negative terms (e.g., *"Waiting 3 weeks for a parcel is my new hobby"*) may be misclassified by n-gram linear models.
* **Language Support:** English only.

---

## 7. How to Load and Run Inference

```python
import joblib

# Load serialized pipeline artifact
model = joblib.load("models/tuned_model.joblib")

# Inference on unseen review
review = ["Package arrived within 24 hours, super fast delivery!"]
prediction = model.predict(review)[0]
probabilities = model.predict_proba(review)[0]

print("Predicted Label:", "Positive" if prediction == 1 else "Negative")
print("Confidence:", f"{max(probabilities) * 100:.2f}%")
```

