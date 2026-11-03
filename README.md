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

---

## 🔬 Task 3: Enhance Features and Tune Model for Better Accuracy

### 1. Feature Engineering with N-grams (Bigrams)
To overcome the unigram limitation where multi-word contextual signals are lost, the TF-IDF vectorizer was extended to extract **Bigrams** (`ngram_range=(1, 2)`) with a `10,000` max vocabulary:
* Effectively captures critical phrases such as *"fast delivery"*, *"delayed transit"*, and *"not late"*.

---

### 2. Algorithm Experimentation & Comparison
We evaluated four model configurations on the 2,037 test samples:
1. **Baseline Logistic Regression (1-gram):** Our established benchmark (90.38%).
2. **Logistic Regression with Bigrams ($C=5.0$):** Improved to 92.15% via n-gram enrichment.
3. **XGBoost Classifier (Bigrams, depth=6, lr=0.1):** Achieved 89.45% (tree models struggle with 10,000 sparse dimensions relative to linear classifiers).
4. **Tuned Linear SVM (LinearSVC with CalibratedClassifierCV):** Achieved **92.39% Accuracy** and **95.33% F1-Score**.

#### Comprehensive Model Comparison Table:
| Model Name | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | Unigrams (1, 1) | 90.38% | 90.12% | **98.90%** | 94.31% | 0.9589 | Benchmark |
| **Logistic Regression (Bigrams)** | Bigrams (1, 2) | 92.15% | 92.88% | 97.75% | 95.25% | 0.9617 | Improved |
| **XGBoost Classifier** | Bigrams (1, 2) | 89.45% | 90.15% | 97.56% | 93.71% | 0.9310 | Sparse penalty |
| **Tuned Linear SVM (Best C=0.5)** | **Bigrams (1, 2)** | **92.39%** | **94.33%** | 96.34% | **95.33%** | **0.9624** | **WINNER (+2.01%)** |

---

### 3. Hyperparameter Tuning via GridSearchCV
* **Grid Space:** $C \in [0.1, 0.5, 1.0, 2.0]$
* **Cross-Validation:** 3-Fold Stratified Cross-Validation on training data
* **Best Parameters:** `{'clf__estimator__C': 0.5}`
* **Best 3-Fold CV Accuracy:** **92.30%**
* **Verification:** Improved Accuracy (`92.39%`) > Baseline Accuracy (`90.38%`) $\rightarrow$ **PASSED (+2.01%)**

---

### 4. Saved Artifacts & Deliverables
* **Tuning & Evaluation Script:** [`src/tune_model.py`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/src/tune_model.py)
* **Pre-executed Notebook:** [`notebooks/02_feature_engineering_and_model_tuning.ipynb`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/notebooks/02_feature_engineering_and_model_tuning.ipynb)
* **Best Tuned Model:** [`models/tuned_model.joblib`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/models/tuned_model.joblib)
* **Comprehensive Study Guide:** [`docs/TASK_3_EXPLANATION.md`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/docs/TASK_3_EXPLANATION.md)

---

### 5. How to Run & Reproduce

#### Run Baseline Pipeline:
```bash
python src/pipeline.py
```

#### Run Model Tuning & Experiments:
```bash
python src/tune_model.py
```

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
├── docs/
│   ├── TASK_1_EXPLANATION.md              # Task 1 guide, Interview Q&As & MCQs
│   ├── TASK_2_EXPLANATION.md              # Task 2 guide, Interview Q&As & MCQs
│   └── TASK_3_EXPLANATION.md              # Task 3 guide, Interview Q&As & MCQs
├── models/
│   ├── baseline_logistic_regression.joblib # Baseline model artifact
│   └── tuned_model.joblib                 # Tuned Linear SVM best model artifact
├── notebooks/
│   ├── 01_baseline_logistic_regression.ipynb # Task 2 baseline notebook
│   └── 02_feature_engineering_and_model_tuning.ipynb # Task 3 tuning & comparison notebook
├── src/
│   ├── __init__.py
│   ├── prepare_dataset.py                 # Task 1: Data collection & filtering
│   ├── pipeline.py                        # Task 2: Baseline pipeline
│   └── tune_model.py                      # Task 3: N-grams & Model Tuning
├── requirements.txt                       # Project dependencies
├── PROJECT_OVERVIEW.md                    # Project roadmap & guide
└── README.md                              # Comprehensive project documentation
```
