"""استخراج دسته‌ها از متن کامنت‌ها و تجمیع آن‌ها برای هر دکتر."""
import re

import pandas as pd

from Services.keywords import ASPECT_KEYWORDS, NEGATION_PHRASES, NEGATION_WORDS
from Services.scoring_service import PRIOR_WEIGHT, bayes_mean
from Services.text_service import split_sentences

TOKEN_SPLIT = re.compile(r"[\s،,;:()«»\"']+")


def _padded(sentence: str) -> str:
    """جمله را با فاصله دو طرف برمی‌گرداند تا تطابق فقط روی کلمه‌ی کامل باشد."""
    tokens = [t for t in TOKEN_SPLIT.split(sentence) if t]
    return " " + " ".join(tokens) + " "


def _contains(padded: str, term: str) -> bool:
    return f" {term} " in padded


def _is_negated(padded: str) -> bool:
    return any(_contains(padded, w) for w in NEGATION_WORDS) or \
        any(_contains(padded, p) for p in NEGATION_PHRASES)


def find_aspects(text, keywords: dict[str, list[str]] = ASPECT_KEYWORDS) -> set[str]:
    """دسته‌های یک کامنت. جمله‌ی منفی کنار گذاشته می‌شود (محدودیت: دسته‌های مثبت همان جمله هم حذف می‌شوند)."""
    found = set()
    for sentence in split_sentences(text):
        padded = _padded(sentence)
        if _is_negated(padded):
            continue
        for aspect, terms in keywords.items():
            if any(_contains(padded, term) for term in terms):
                found.add(aspect)
    return found


def add_aspect_columns(df: pd.DataFrame, text_column: str = "text_norm",
                       score_column: str = "score_10",
                       keywords: dict[str, list[str]] = ASPECT_KEYWORDS) -> pd.DataFrame:
    """برای هر دسته: has_<دسته> (۰/۱) و <دسته>_score_10 (امتیاز کامنت یا خالی)."""
    for col in (text_column, score_column):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    out = df.copy()
    per_row = out[text_column].apply(find_aspects)
    for aspect in keywords:
        has = per_row.apply(lambda s: aspect in s).astype(int)
        out[f"has_{aspect}"] = has
        out[f"{aspect}_score_10"] = out[score_column].where(has == 1)
    out["aspect_count"] = per_row.apply(len)
    return out


def aggregate_aspect_scores(df: pd.DataFrame, prior_weight: int = PRIOR_WEIGHT,
                            keywords: dict[str, list[str]] = ASPECT_KEYWORDS) -> pd.DataFrame:
    """برای هر دکتر و هر دسته: <دسته>_count و <دسته>_score (بیزی). بدون کامنت در دسته: NaN."""
    result = pd.DataFrame({"doctor_id": df["doctor_id"].unique()}).set_index("doctor_id")

    for aspect in keywords:
        has_col, score_col = f"has_{aspect}", f"{aspect}_score_10"
        for col in (has_col, score_col):
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}. Run add_aspect_columns first.")

        rows = df[df[has_col] == 1]
        global_mean = rows[score_col].mean()
        grouped = rows.groupby("doctor_id")[score_col].agg(["count", "mean"])

        count = grouped["count"].reindex(result.index).fillna(0).astype(int)
        mean = grouped["mean"].reindex(result.index).fillna(0)
        score = bayes_mean(count, mean, global_mean, prior_weight).where(count > 0).round(2)

        result[f"{aspect}_count"] = count
        result[f"{aspect}_score"] = score

    return result.reset_index()