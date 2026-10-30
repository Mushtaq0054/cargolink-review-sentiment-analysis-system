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

### 4. Dataset Files & Schema

Output files saved in `data/`:
* [`data/train.csv`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/data/train.csv) (8,141 rows)
* [`data/test.csv`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/data/test.csv) (2,037 rows)
* [`data/cargolink_reviews_cleaned.csv`](file:///c:/Users/Mushtaq%20Ahmad%20Madni/Desktop/internship%20projects/cargolink-review-sentiment-analysis-system/data/cargolink_reviews_cleaned.csv) (10,178 rows, with `split` column)

#### Schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `review_text` | String | Cleaned customer review text describing delivery, packaging, or courier service. |
| `label` | Integer | Binary sentiment label: `0` for Negative, `1` for Positive. |

---

### 5. How to Reproduce Dataset Preparation

Make sure the virtual environment `cargo` is activated, then run:

```bash
python src/prepare_dataset.py
```

This script will automatically:
1. Download the raw CSV directly via `requests` into `data/raw/`.
2. Apply logistics domain filters and binary labeling via `pandas`.
3. Perform the stratified 80/20 train/test split.
4. Export the final standardized CSV files.

---

### 6. Project Directory Structure
```
cargolink-review-sentiment-analysis-system/
├── cargo/                                 # Dedicated virtual environment
├── data/
│   ├── raw/
│   │   └── amazon_reviews_raw.csv         # Downloaded raw dataset (24 MB)
│   ├── cargolink_reviews_cleaned.csv      # Combined filtered logistics dataset
│   ├── train.csv                          # 80% Train set (8,141 rows)
│   └── test.csv                           # 20% Test set (2,037 rows)
├── src/
│   ├── __init__.py
│   └── prepare_dataset.py                 # Automated data download & prep pipeline
├── requirements.txt                       # Project dependencies
├── PROJECT_OVERVIEW.md                    # Complete project guide and roadmap
└── README.md                              # Task 1 documentation & submission report
```

