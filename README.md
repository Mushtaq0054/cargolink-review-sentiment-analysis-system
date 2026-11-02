# 📦 CargoLink Review Sentiment Analysis System

CargoLink Logistics is an end-to-end Machine Learning sentiment analysis system designed to classify customer reviews into positive and negative sentiments, expose predictions via a FastAPI microservice, and monitor service delivery trends on a real-time Streamlit dashboard.

---

## 🚀 Task 1: Collect and Prepare CargoLink Review Dataset

### 1. Dataset Source
* **Source:** Open-source Public Amazon Customer Reviews Dataset
* **Source URL:** [Amazon Reviews CSV (GitHub)](https://raw.githubusercontent.com/shreya-maher/DSEM_Datasets/main/amazon_reviews.csv)
* **Raw Size:** ~24.2 MB (45,476 raw customer records)

---

### 2. Data Filtering & Preprocessing Methodology
To tailor the general e-commerce feedback specifically to **CargoLink Logistics** operations:
1. **Sentiment Labeling:**
   * Customer ratings of 4 and 5 were mapped to **Positive (`1`)**.
   * Customer ratings of 1 and 2 were mapped to **Negative (`0`)**.
   * Neutral ratings (Score = 3) were omitted to ensure high-confidence binary classification.
2. **Logistics Domain Filtering:**
   * Filtered reviews using regex matching against logistics-critical terminology:
     `delivery`, `shipping`, `shipment`, `courier`, `package`, `packaging`, `parcel`, `tracking`, `transit`, `delay`, `delayed`, `arrived`, `arrival`, `carrier`, `box`, `damaged`, `order`, `received`, `dispatch`.
   * Out of 42,075 non-neutral reviews, **10,852** logistics-relevant reviews were extracted.
3. **Deduplication & Quality Check:**
   * Removed exact duplicate reviews.
   * Filtered out noisy reviews with fewer than 15 characters, yielding **10,178** high-quality logistics reviews.

---

### 3. Train / Test Split (80 / 20)
The filtered logistics dataset was split into training and test partitions using stratified sampling (`seed=42`) to preserve class proportions across splits:

| Split | Percentage | Total Samples | Positive (`1`) | Negative (`0`) |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 80% | **8,141** | 6,560 (80.58%) | 1,581 (19.42%) |
| **Test Set** | 20% | **2,037** | 1,641 (80.58%) | 396 (19.42%) |
| **Total** | 100% | **10,178** | 8,201 (80.58%) | 1,977 (19.42%) |

---

## ⚡ Task 2: Build Text Cleaning Pipeline & Baseline Logistic Regression

### 1. NLP Text Preprocessing Pipeline
Textual reviews cannot be processed directly by Machine Learning algorithms without domain cleaning. A modular cleaning pipeline was built in [`src/pipeline.py`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/src/pipeline.py) implementing:
1. **Lowercase Normalization:** Ensures uniform case sensitivity across all terms.
2. **URL & Noise Stripping:** Strips web links, HTML artifacts, and stray numbers.
3. **Punctuation Removal:** Cleans non-alphabetical characters using regex patterns (`[^a-zA-Z\s]`).
4. **Intelligent Stopwords Removal:** Filters English stopwords using NLTK while crucially preserving negation indicators (`not`, `no`, `never`, `nor`, `neither`, `barely`, `hardly`) so critical sentiment modifiers (e.g. *"not delivered"*, *"no delay"*) are preserved.
5. **Token Filtering:** Drops single-character noise and normalizes whitespace.

---

### 2. Scikit-Learn Pipeline Architecture
The baseline model combines feature extraction and classification into a unified `sklearn.pipeline.Pipeline`:
* **Stage 1 (Vectorization):** `TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 1))`
  * Converts cleaned text into term-frequency inverse-document-frequency numerical feature vectors.
  * Sublinear TF scaling ($1 + \log(\text{tf})$) dampens the impact of overly repeated words.
* **Stage 2 (Classification):** `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)`
  * Fast, interpretable linear classification baseline with standard L2 penalty regularization.

---

### 3. Model Evaluation Results (Test Set)
The trained baseline model was evaluated on the unseen test set (2,037 samples):

| Metric | Baseline Score | Mentor Target | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **90.38%** | **> 70.00%** | **PASSED (+20.38% above target)** |
| **Precision** | **90.12%** | - | High confidence on positive predictions |
| **Recall** | **98.90%** | - | Exceptional capture of positive sentiments |
| **F1 Score** | **94.31%** | - | Robust balance |

#### Confusion Matrix:
| True / Pred | Predicted Negative (`0`) | Predicted Positive (`1`) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Negative (`0`)** | **218 (TN)** | 178 (FP) | 396 |
| **Actual Positive (`1`)** | 18 (FN) | **1623 (TP)** | 1641 |

#### Classification Report:
```
              precision    recall  f1-score   support

Negative (0)       0.92      0.55      0.69       396
Positive (1)       0.90      0.99      0.94      1641

    accuracy                           0.90      2037
   macro avg       0.91      0.77      0.82      2037
weighted avg       0.91      0.90      0.89      2037
```

---

### 4. Saved Model Artifacts & Notebook
* **Pipeline Script:** [`src/pipeline.py`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/src/pipeline.py)
* **Pre-executed Notebook:** [`notebooks/01_baseline_logistic_regression.ipynb`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/notebooks/01_baseline_logistic_regression.ipynb)
* **Trained Model:** [`models/baseline_logistic_regression.joblib`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/models/baseline_logistic_regression.joblib)

---

### 5. How to Run & Reproduce

#### Run Baseline Pipeline Training & Evaluation:
```bash
python src/pipeline.py
```

#### Inspect the Jupyter Notebook:
Open [`notebooks/01_baseline_logistic_regression.ipynb`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/notebooks/01_baseline_logistic_regression.ipynb) in VS Code or Jupyter Lab to view data distributions, confusion matrix heatmap, ROC curve, and top sentiment features.

---

### 6. Project Directory Structure
```
cargolink-review-sentiment-analysis-system/
├── cargo/                                 # Dedicated virtual environment
├── data/
│   ├── raw/
│   │   └── amazon_reviews_raw.csv         # Downloaded raw dataset
│   ├── cargolink_reviews_cleaned.csv      # Filtered logistics dataset (10,178 rows)
│   ├── train.csv                          # 80% Train set (8,141 rows)
│   └── test.csv                           # 20% Test set (2,037 rows)
├── models/
│   └── baseline_logistic_regression.joblib # Saved baseline model pipeline
├── notebooks/
│   └── 01_baseline_logistic_regression.ipynb # Task 2 complete analysis & evaluation
├── src/
│   ├── __init__.py
│   ├── prepare_dataset.py                 # Task 1: Data collection & filtering
│   └── pipeline.py                        # Task 2: Text cleaning & Baseline pipeline
├── requirements.txt                       # Project dependencies
├── PROJECT_OVERVIEW.md                    # Project roadmap & guide
└── README.md                              # Project documentation & task reports
```
