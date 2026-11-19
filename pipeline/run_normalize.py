"""مرحله‌ی ۱: نرمال‌سازی کامنت‌ها و ساخت ستون text_norm."""
from Services.data_service import NORMALIZED_COMMENTS_CSV_PATH
from Services.dataset_service import load_comments
from Services.text_service import normalize_column


def main() -> None:
    df = normalize_column(load_comments())
    df.to_csv(NORMALIZED_COMMENTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"Rows: {len(df)}  saved: {NORMALIZED_COMMENTS_CSV_PATH}")


if __name__ == "__main__":
    main()