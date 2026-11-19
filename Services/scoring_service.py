"""امتیازدهی: تبدیل rate به مقیاس ۰ تا ۱۰ و میانگین بیزی."""
import pandas as pd

PRIOR_WEIGHT = 10


def add_score_10(df: pd.DataFrame, source_column: str = "rate", output_column: str = "score_10") -> pd.DataFrame:
    """rate از ۰ تا ۵ است؛ ضرب در ۲ آن را به ۰ تا ۱۰ می‌برد."""
    if source_column not in df.columns:
        raise KeyError(f"Missing column: {source_column}")
    out = df.copy()
    out[output_column] = out[source_column] * 2
    return out


def bayes_mean(count, mean, global_mean: float, prior_weight: int = PRIOR_WEIGHT):
    """میانگین وزنی: تعداد کم را به سمت میانگین کل می‌کشد."""
    return (prior_weight * global_mean + count * mean) / (prior_weight + count)


def overall_comment_scores(df: pd.DataFrame, score_column: str = "score_10",
                           prior_weight: int = PRIOR_WEIGHT) -> pd.DataFrame:
    """برای هر دکتر: comment_count و comment_rating (بیزی)."""
    global_mean = df[score_column].mean()
    stats = df.groupby("doctor_id")[score_column].agg(comment_count="count", mean_score="mean")
    stats["comment_rating"] = bayes_mean(
        stats["comment_count"], stats["mean_score"], global_mean, prior_weight
    ).round(2)
    return stats[["comment_count", "comment_rating"]].reset_index()