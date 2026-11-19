"""ساخت جدول نهایی دکترها برای RAG از روی داده‌های آماده‌شده."""
import os

import pandas as pd

from Services.aspect_service import add_aspect_columns, aggregate_aspect_scores
from Services.keywords import ASPECT_KEYWORDS
from Services.scoring_service import PRIOR_WEIGHT, add_score_10, overall_comment_scores


def prepare_comments(comments: pd.DataFrame) -> pd.DataFrame:
    """امتیاز ۰ تا ۱۰ و ستون‌های دسته‌ها را به کامنت‌ها اضافه می‌کند."""
    df = add_score_10(comments)
    return add_aspect_columns(df)


def build_doctor_frame(doctors: pd.DataFrame, prepared_comments: pd.DataFrame,
                       prior_weight: int = PRIOR_WEIGHT) -> pd.DataFrame:
    """یک ردیف برای هر دکتر در doctors (حتی بدون کامنت)."""
    base = doctors[["doctor_id", "name", "specialty", "profile_url", "rating_score_5"]].copy()
    base["program_rating"] = base["rating_score_5"] * 2
    base = base.drop(columns=["rating_score_5"])

    overall = overall_comment_scores(prepared_comments, prior_weight=prior_weight)
    aspects = aggregate_aspect_scores(prepared_comments, prior_weight=prior_weight)

    merged = base.merge(overall, on="doctor_id", how="left").merge(aspects, on="doctor_id", how="left")

    count_cols = ["comment_count"] + [f"{a}_count" for a in ASPECT_KEYWORDS]
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)

    front = ["doctor_id", "name", "specialty", "profile_url",
             "program_rating", "comment_count", "comment_rating"]
    rest = [c for a in ASPECT_KEYWORDS for c in (f"{a}_count", f"{a}_score")]
    return merged[front + rest]


def save_doctor_frame(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")