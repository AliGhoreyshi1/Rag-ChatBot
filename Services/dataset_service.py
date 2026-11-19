"""بارگذاری داده‌های پروژه با بررسی‌های مشترک."""
import pandas as pd

from Services.data_service import (
    COMMENTS_FULL_CSV_PATH,
    DATASET_COMMENTS_FULL_DOWNLOAD_URL,
    DATASET_DOCTORS_RAW_CANONICAL_DOWNLOAD_URL,
    DATASET_NORMALIZED_COMMENTS_DOWNLOAD_URL,
    DOCTORS_RAW_CANONICAL_CSV_PATH,
    NORMALIZED_COMMENTS_CSV_PATH,
    ensure_dataset_available,
    read_csv,
)

COMMENTS_REQUIRED = ["doctor_id", "text", "rate", "label"]
NORMALIZED_REQUIRED = COMMENTS_REQUIRED + ["text_norm"]
DOCTORS_REQUIRED = ["doctor_id", "name", "specialty", "profile_url", "rating_score_5"]

RATE_BY_LABEL = {1: 5.0, 0: 2.5, -1: 0.0}


def _check_comments(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    if verbose:
        empty = int((df["text"].fillna("").str.strip() == "").sum())
        mismatch = int((df["label"].map(RATE_BY_LABEL) != df["rate"]).sum())
        print(f"Empty comments: {empty}")
        print(f"Rate/label mismatches: {mismatch}")
    return df


def load_comments(verbose: bool = True) -> pd.DataFrame:
    path = ensure_dataset_available(COMMENTS_FULL_CSV_PATH, DATASET_COMMENTS_FULL_DOWNLOAD_URL)
    return _check_comments(read_csv(path, COMMENTS_REQUIRED), verbose)


def load_normalized_comments(verbose: bool = True) -> pd.DataFrame:
    path = ensure_dataset_available(NORMALIZED_COMMENTS_CSV_PATH, DATASET_NORMALIZED_COMMENTS_DOWNLOAD_URL)
    return _check_comments(read_csv(path, NORMALIZED_REQUIRED), verbose)


def load_doctors(verbose: bool = True) -> pd.DataFrame:
    path = ensure_dataset_available(DOCTORS_RAW_CANONICAL_CSV_PATH, DATASET_DOCTORS_RAW_CANONICAL_DOWNLOAD_URL)
    df = read_csv(path, DOCTORS_REQUIRED)
    if verbose:
        print(f"Duplicate doctor records: {int(df['doctor_id'].duplicated(keep=False).sum())}")
    return df