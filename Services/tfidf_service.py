"""TF-IDF روی کامنت‌ها و میانگین امتیاز هر کلمه (بدون رسم نمودار)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from Services.scoring_service import PRIOR_WEIGHT


def term_scores(df: pd.DataFrame, text_column: str = "text_norm", score_column: str = "score_10",
                min_df: int = 25, shrink_k: int = PRIOR_WEIGHT * 2) -> pd.DataFrame:
    texts = df[text_column].fillna("")
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=min_df, lowercase=False)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        # sklearn وقتی هیچ کلمه‌ای از min_df نگذرد خطا می‌دهد؛ جدول خالی برمی‌گردانیم
        return pd.DataFrame(columns=["term", "comment_count", "mean_score",
                                     "smoothed_score", "mean_tfidf"])
    terms = np.array(vec.get_feature_names_out())

    presence = (X > 0).astype(float).tocsc()
    count = np.asarray(presence.sum(axis=0)).ravel()
    total_score = presence.T.dot(df[score_column].values)
    mean_score = total_score / count

    global_mean = df[score_column].mean()
    smoothed = (count * mean_score + shrink_k * global_mean) / (count + shrink_k)
    mean_tfidf = np.asarray(X.sum(axis=0)).ravel() / count

    result = pd.DataFrame({
        "term": terms,
        "comment_count": count.astype(int),
        "mean_score": mean_score,
        "smoothed_score": smoothed,
        "mean_tfidf": mean_tfidf,
    })
    return result.sort_values("smoothed_score").reset_index(drop=True)