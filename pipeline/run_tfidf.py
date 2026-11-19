"""مرحله‌ی تحلیلی: TF-IDF روی متن نرمال‌شده و ذخیره‌ی جدول و نمودارها."""
import os

from Services.data_service import TFIDF_OUTPUT_DIR
from Services.dataset_service import load_normalized_comments
from Services.plot_service import plot_scatter, plot_terms, setup_fonts
from Services.scoring_service import add_score_10
from Services.tfidf_service import term_scores


def main() -> None:
    os.makedirs(TFIDF_OUTPUT_DIR, exist_ok=True)
    setup_fonts()

    df = add_score_10(load_normalized_comments())
    result = term_scores(df)

    result.to_csv(os.path.join(TFIDF_OUTPUT_DIR, "term_scores.csv"), index=False, encoding="utf-8-sig")
    plot_terms(result, df["score_10"].mean(), os.path.join(TFIDF_OUTPUT_DIR, "top_terms.png"))
    plot_scatter(result, os.path.join(TFIDF_OUTPUT_DIR, "terms_scatter.png"))
    print(f"Terms kept: {len(result)}  saved to {TFIDF_OUTPUT_DIR}")


if __name__ == "__main__":
    main()