"""
CargoLink Review Sentiment Analysis System
Task 1: Dataset Collection, Logistics Filtering, and 80/20 Train/Test Split
"""

import os
import re
import requests
import pandas as pd
import numpy as np

# URLs and file paths
DATA_URL = "https://raw.githubusercontent.com/shreya-maher/DSEM_Datasets/main/amazon_reviews.csv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_FILE = os.path.join(RAW_DIR, "amazon_reviews_raw.csv")

CLEANED_FILE = os.path.join(DATA_DIR, "cargolink_reviews_cleaned.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")


def ensure_directories():
    """Ensure data directories exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def download_raw_data(url: str, dest_path: str):
    """Download raw dataset from public URL using requests with chunking."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        print(f"[INFO] Raw file already exists at {dest_path} ({os.path.getsize(dest_path):,} bytes). Skipping download.")
        return

    print(f"[INFO] Downloading raw dataset from: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                if total_bytes % (1024 * 1024 * 5) < 1024 * 64:
                    print(f"  Downloaded: {total_bytes / (1024 * 1024):.1f} MB...")

    print(f"[SUCCESS] Download completed: {os.path.getsize(dest_path):,} bytes saved to {dest_path}")


def process_and_filter_dataset():
    """Load raw dataset, filter for logistics terms, label sentiments, and split 80/20."""
    print("\n[INFO] Loading raw dataset into pandas...")
    df_raw = pd.read_csv(RAW_FILE)
    print(f"  Raw dataset shape: {df_raw.shape}")
    print(f"  Raw columns: {list(df_raw.columns)}")

    # Ensure required columns exist
    text_col = "Text" if "Text" in df_raw.columns else None
    score_col = "Score" if "Score" in df_raw.columns else None

    if not text_col or not score_col:
        raise ValueError(f"Expected 'Text' and 'Score' columns. Found: {list(df_raw.columns)}")

    # Drop null reviews
    df = df_raw.dropna(subset=[text_col, score_col]).copy()

    # Map scores to binary sentiment labels:
    # Score > 3 -> 1 (Positive)
    # Score < 3 -> 0 (Negative)
    # Score == 3 (Neutral) excluded for clear binary classification
    df = df[df[score_col] != 3].copy()
    df["label"] = (df[score_col] > 3).astype(int)

    # Combine Summary and Text if Summary is available for richer context
    if "Summary" in df.columns:
        df["review_text"] = df["Summary"].fillna("").astype(str) + ". " + df[text_col].astype(str)
        # Clean up leading dots if Summary was empty
        df["review_text"] = df["review_text"].str.strip().str.lstrip(".").str.strip()
    else:
        df["review_text"] = df[text_col].astype(str)

    # 3. Filter rows containing logistics terms
    # Comprehensive logistics / shipping / packaging keywords pattern
    logistics_pattern = (
        r"\b(?:delivery|deliveries|delivered|delivering|shipping|shipment|shipped|"
        r"courier|package|packaging|packaged|parcel|tracking|transit|delay|delayed|"
        r"delays|arrived|arrival|carrier|box|damaged|order|received|fast delivery|quick delivery|"
        r"on time|late|dispatch|dispatched)\b"
    )

    print("\n[INFO] Filtering rows containing logistics terms...")
    is_logistics = df["review_text"].str.contains(logistics_pattern, case=False, regex=True, na=False)
    df_filtered = df[is_logistics].copy()

    print(f"  Total non-neutral reviews: {len(df):,}")
    print(f"  Logistics-relevant reviews found: {len(df_filtered):,}")

    # Keep only the standardized deliverables columns: review_text and label
    df_clean = df_filtered[["review_text", "label"]].copy()

    # Remove duplicates and very short invalid reviews (< 15 characters)
    df_clean = df_clean.drop_duplicates(subset=["review_text"])
    df_clean = df_clean[df_clean["review_text"].str.len() >= 15].reset_index(drop=True)

    print(f"  Clean unique logistics reviews: {len(df_clean):,}")
    print("  Label distribution:")
    label_counts = df_clean["label"].value_counts()
    for lbl, cnt in label_counts.items():
        name = "Positive (1)" if lbl == 1 else "Negative (0)"
        print(f"    {name}: {cnt:,} ({cnt / len(df_clean) * 100:.2f}%)")

    # 4. Split into 80/20 train/test
    print("\n[INFO] Performing 80/20 stratified Train/Test split...")
    rng = np.random.default_rng(seed=42)
    pos_indices = df_clean[df_clean["label"] == 1].index.to_numpy().copy()
    neg_indices = df_clean[df_clean["label"] == 0].index.to_numpy().copy()

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    pos_split = int(0.8 * len(pos_indices))
    neg_split = int(0.8 * len(neg_indices))

    train_indices = np.concatenate([pos_indices[:pos_split], neg_indices[:neg_split]])
    test_indices = np.concatenate([pos_indices[pos_split:], neg_indices[neg_split:]])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_df = df_clean.loc[train_indices].copy().reset_index(drop=True)
    test_df = df_clean.loc[test_indices].copy().reset_index(drop=True)

    # Mark split in combined dataframe
    df_clean["split"] = "train"
    df_clean.loc[test_indices, "split"] = "test"

    # 5. Save to CSV
    print("\n[INFO] Saving datasets to CSV...")
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    df_clean.to_csv(CLEANED_FILE, index=False)

    print(f"  Saved Train Dataset: {TRAIN_FILE} ({len(train_df):,} rows)")
    print(f"  Saved Test Dataset:  {TEST_FILE} ({len(test_df):,} rows)")
    print(f"  Saved Combined File: {CLEANED_FILE} ({len(df_clean):,} rows)")

    print("\n=== DATASET PREPARATION SUMMARY ===")
    print(f"Total Logistics Reviews: {len(df_clean):,}")
    print(f"Train Set (80%): {len(train_df):,} rows (Positive: {(train_df['label']==1).sum():,}, Negative: {(train_df['label']==0).sum():,})")
    print(f"Test Set (20%):  {len(test_df):,} rows (Positive: {(test_df['label']==1).sum():,}, Negative: {(test_df['label']==0).sum():,})")
    print("Columns in Train & Test: 'review_text', 'label'")
    print("=" * 36)


def main():
    ensure_directories()
    download_raw_data(DATA_URL, RAW_FILE)
    process_and_filter_dataset()


if __name__ == "__main__":
    main()
