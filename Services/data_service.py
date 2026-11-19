"""مسیرها، لینک‌های دانلود و خواندن CSV. هیچ منطق دامنه‌ای اینجا نیست."""
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

COMMENTS_FULL_CSV_PATH = os.path.join(DATA_DIR, "comments_full.csv")
DOCTORS_RAW_CANONICAL_CSV_PATH = os.path.join(DATA_DIR, "doctors_raw_canonical.csv")
NORMALIZED_COMMENTS_CSV_PATH = os.path.join(DATA_DIR, "normalized_comment_v2.csv")
DOCTORS_FOR_RAG_CSV_PATH = os.path.join(OUTPUT_DIR, "doctors_for_rag.csv")
TFIDF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tfidf")

DATASET_COMMENTS_FULL_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1sgUhjUKzC3pNoQX3-_vgXFgI1l0371Hy/view?usp=sharing"
)
DATASET_DOCTORS_RAW_CANONICAL_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1pNWjDf2h9BhFTlD-9_b1FaPNFx3hdj6L/view?usp=sharing"
)
DATASET_NORMALIZED_COMMENTS_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1XlghcNbUzVNXQRswymgJ69b-Cf2vApqC/view?usp=sharing"
)


def ensure_dataset_available(csv_path: str, download_url: str | None = None) -> str:
    """اگر فایل محلی موجود بود همان را برمی‌گرداند، وگرنه از Google Drive دانلود می‌کند."""
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path

    if not download_url:
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put the CSV there or set its download URL."
        )
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is not installed. Run: pip install gdown") from exc

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    try:
        result = gdown.download(url=download_url, output=csv_path, quiet=False)
    except Exception as exc:
        # فایل ناقص نباید باقی بماند، وگرنه دفعه‌ی بعد به‌عنوان داده‌ی سالم پذیرفته می‌شود
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        raise RuntimeError(
            "Could not download the dataset from Google Drive. Check your internet "
            "connection and make sure the file is shared with 'Anyone with the link'."
        ) from exc

    if not result or not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        if os.path.isfile(csv_path):
            os.remove(csv_path)
        raise RuntimeError("Google Drive did not return the dataset. Check the sharing permissions.")
    return csv_path


def read_csv(path: str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """CSV را با utf-8-sig می‌خواند، ستون‌های لازم را بررسی و فقط همان‌ها را برمی‌گرداند."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {os.path.basename(path)}: {missing}")
        df = df[required_columns]
    return df